"""Stage-1 weakly supervised proposition-evidence training entry point.

This command trains evidence heads only. It never trains or emits disease
probabilities; disease labels remain unavailable to the model, loss, and metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from models.clip_tqn import ModelRes, Text_Encoder_Bert
from proposition.data import Stage1SkinCAPDataset, WeakSupervisionArtifacts, load_weak_supervision_v3
from proposition.engine import Stage1Trainer, TrainerConfig
from proposition.losses import MaskedBCELoss, build_pos_weight_artifacts, write_label_distribution
from proposition.models import Stage1EvidenceModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ws.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kb_path", help="Reserved for the external full v3 KB; Stage 1 reads active_csv.")
    parser.add_argument("--labels_npz")
    parser.add_argument("--active_masks_npz")
    parser.add_argument("--active_csv")
    parser.add_argument("--metadata_json")
    parser.add_argument("--train_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--test_csv")
    parser.add_argument("--image_root")
    parser.add_argument("--bert_model_name")
    parser.add_argument("--image_encoder_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--query_chunk_size", type=int)
    parser.add_argument("--pilot_diseases", nargs="*")
    parser.add_argument("--return_attention_samples", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--disease_aliases_json", required=True, help="JSON map/list covering every disease name and alias to mask globally.")
    parser.add_argument("--case_id_column", help="Named case-ID column; omit for legacy SkinCAP filename-derived IDs.")
    parser.add_argument("--image_column", help="Named image column; defaults to legacy CSV column 0.")
    parser.add_argument("--caption_column", help="Named caption column; defaults to legacy CSV column 1.")
    parser.add_argument("--epochs", type=int)
    return parser


def _deep_get(data: Mapping[str, Any], section: str, key: str, default: Any = None) -> Any:
    return data.get(section, {}).get(key, default)


def _value(args: argparse.Namespace, config: Mapping[str, Any], section: str, key: str, default: Any = None) -> Any:
    return getattr(args, key, None) or _deep_get(config, section, key, default)


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_aliases(path: str | None, artifacts: WeakSupervisionArtifacts) -> Mapping[str, list[str]]:
    if not path:
        raise ValueError("--disease_aliases_json is required to prevent disease-name leakage")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("disease_aliases_json must map every full-artifact disease_id to its aliases")
    aliases: dict[str, list[str]] = {}
    for key, values in payload.items():
        if isinstance(values, str) or not isinstance(values, (list, tuple)):
            raise ValueError(f"aliases for {key!r} must be a JSON list")
        aliases[str(key)] = [str(value) for value in values]
    required = {proposition.disease_id for proposition in artifacts.active_propositions}
    missing = sorted(required - set(aliases))
    if missing:
        raise ValueError(f"disease_aliases_json is missing full-artifact disease IDs: {missing!r}")
    return aliases


def _pilot_artifacts(artifacts: WeakSupervisionArtifacts, diseases: list[str] | None) -> WeakSupervisionArtifacts:
    if not diseases:
        return artifacts
    selected = {str(value) for value in diseases}
    positions = np.asarray(
        [index for index, proposition in enumerate(artifacts.active_propositions) if proposition.disease_id in selected and proposition.active_scoring],
        dtype=np.intp,
    )
    if not len(positions):
        raise ValueError("--pilot_diseases did not select any active propositions")
    return replace(
        artifacts,
        proposition_ids=artifacts.proposition_ids[positions],
        label_text=artifacts.label_text[:, positions], label_img=artifacts.label_img[:, positions], p_img=artifacts.p_img[:, positions],
        text_trainable=artifacts.text_trainable[positions], image_trainable=artifacts.image_trainable[positions],
        active_scoring=artifacts.active_scoring[positions],
        active_propositions=tuple(artifacts.active_propositions[index] for index in positions),
    )


def _transform(image_res: int, training: bool):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    steps: list[Any] = [transforms.RandomResizedCrop(image_res, scale=(0.2, 1.0)) if training else transforms.Resize((image_res, image_res))]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose([*steps, transforms.ToTensor(), normalize])


def _loader(dataset: Stage1SkinCAPDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def main(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    weak = config.get("weak_supervision", {})
    artifacts = load_weak_supervision_v3(
        labels_npz=args.labels_npz or weak.get("labels"),
        active_masks_npz=args.active_masks_npz or weak.get("active_masks"),
        active_csv=args.active_csv or weak.get("active_csv"),
        metadata_json=args.metadata_json or weak.get("metadata"),
    )
    # Load aliases from the complete v3 artifact set before restricting a pilot.
    aliases = _load_aliases(args.disease_aliases_json, artifacts)
    artifacts = _pilot_artifacts(artifacts, args.pilot_diseases)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42)); _seed_everything(seed)
    device = args.device or config.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    skincap = config.get("skincap", {}); stage1 = config.get("stage1", {})
    image_root = args.image_root or skincap.get("image_root")
    common = dict(weak_supervision=artifacts, disease_aliases=aliases, image_root=image_root,
                  case_id_column=args.case_id_column, image_column=args.image_column, caption_column=args.caption_column)
    train = Stage1SkinCAPDataset(args.train_csv or skincap.get("train_csv"), transform=_transform(int(skincap.get("image_res", 224)), True), split="train", **common)
    val = Stage1SkinCAPDataset(args.val_csv or skincap.get("val_csv"), transform=_transform(int(skincap.get("image_res", 224)), False), split="val", **common)
    test_path = args.test_csv or skincap.get("test_csv")
    test = Stage1SkinCAPDataset(test_path, transform=_transform(int(skincap.get("image_res", 224)), False), split="test", **common) if test_path else None
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name or stage1["bert_model_name"])
    def tokenize(captions: list[str]):
        return tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=int(stage1.get("max_caption_length", 128)))
    embed_dim = int(stage1.get("embed_dim", 768))
    model = Stage1EvidenceModel(image_encoder=ModelRes(args.image_encoder_name or stage1["image_encoder_name"], embed_dim), text_encoder=Text_Encoder_Bert(args.bert_model_name or stage1["bert_model_name"], embed_dim), weak_supervision=artifacts, query_tokenizer=tokenize, embed_dim=embed_dim, num_heads=int(stage1.get("attention_heads", 8)), query_chunk_size=args.query_chunk_size or stage1.get("query_chunk_size"))
    weights = build_pos_weight_artifacts(artifacts, pos_weight_cap=float(stage1.get("pos_weight_cap", 10.0)))
    output = Path(args.output_dir); write_label_distribution(weights.rows, output / "supervision" / "label_distribution.csv")
    trainer = Stage1Trainer(model=model, tokenizer=tokenize, weak_supervision=artifacts, loss_fn=MaskedBCELoss(weights.text_pos_weight, weights.image_pos_weight), config=TrainerConfig(device=device, early_stopping_patience=int(stage1.get("early_stopping_patience", 10)), learning_rate=float(stage1.get("lr", 1e-4)), weight_decay=float(stage1.get("weight_decay", 1e-4)), config=config), output_dir=output)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    batch_size, eval_batch_size, workers = int(skincap.get("batch_size", 32)), int(skincap.get("eval_batch_size", 64)), int(skincap.get("num_workers", 2))
    val_loader = _loader(val, eval_batch_size, workers, False)
    trainer.fit(_loader(train, batch_size, workers, True), val_loader, max_epochs=args.epochs or int(stage1.get("epochs", 50)), test_dataloader=_loader(test, eval_batch_size, workers, False) if test else None)
    if args.return_attention_samples:
        trainer.save_attention_samples(val_loader, count=args.return_attention_samples)


if __name__ == "__main__":
    main(_parser().parse_args())

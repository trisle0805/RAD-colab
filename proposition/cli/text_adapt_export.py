"""Export frozen Round-B Stage-1 scores after selection, including held-out test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from main_proposition import _load_aliases, _seed_everything, _transform
from models.clip_tqn import ModelRes, Text_Encoder_Bert
from proposition.data import Stage1SkinCAPDataset, load_weak_supervision_v3
from proposition.engine import Stage1Trainer, TrainerConfig
from proposition.engine.text_encoder_adaptation import compare_module_hashes, module_sha256
from proposition.losses import MaskedBCELoss, build_pos_weight_artifacts
from proposition.models import Stage1EvidenceModel

def _loader(dataset: Stage1SkinCAPDataset, batch_size: int, workers: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=generator)

def _image_hashes(model: Stage1EvidenceModel) -> dict[str, str]:
    return {name: module_sha256(getattr(model, name)) for name in (
        "image_encoder", "image_projection", "image_cross_attention", "image_head",
    )}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config", "checkpoint", "labels_npz", "active_masks_npz", "active_csv", "metadata_json",
        "train_csv", "val_csv", "test_csv", "image_root", "disease_aliases_json", "output_dir", "device",
    ):
        parser.add_argument(f"--{name}", required=name in {"checkpoint", "train_csv", "val_csv", "test_csv", "output_dir"})
    parser.add_argument("--seed", type=int)
    return parser


def _anchor_sha256(anchor: torch.Tensor) -> str:
    tensor = anchor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _metrics_payload(result: Any, text_trainable: np.ndarray) -> dict[str, Any]:
    text = result.metrics["text"]
    scores = result.predictions["score_text"]
    labels = result.predictions["label_text"]
    effective = np.broadcast_to(np.asarray(text_trainable, dtype=bool), scores.shape)
    values = scores[effective & np.isfinite(scores)]
    target = (labels == 1) & effective
    negative = (labels != 1) & effective
    return {
        "macro_roc_auc": text["macro_roc_auc"], "macro_pr_auc": text["macro_pr_auc"], "n_eligible_strict": text["n_eligible"],
        "mean_score": float(values.mean()) if values.size else None, "std_score": float(values.std()) if values.size else None,
        "quantiles": {name: float(np.quantile(values, q)) if values.size else None for name, q in (("p01", .01), ("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95), ("p99", .99))},
        "mean_positive_score": float(scores[target].mean()) if np.any(target) else None,
        "mean_negative_score": float(scores[negative].mean()) if np.any(negative) else None,
        "fraction_ge_0_5": float(np.mean(values >= .5)) if values.size else None,
    }


@torch.no_grad()
def _write_query_controls(trainer: Stage1Trainer, loader: DataLoader, result: Any, seed: int) -> None:
    anchor = trainer.model.query_cache.detach().clone()
    controls: dict[str, np.ndarray] = {}
    generator = torch.Generator(device=trainer.device).manual_seed(seed)
    permutation = torch.randperm(anchor.shape[0], generator=generator, device=trainer.device)
    values = [
        ("swap", anchor[permutation]),
        ("zero", torch.zeros_like(anchor)),
        ("mean", anchor.mean(dim=0, keepdim=True).expand_as(anchor)),
        ("random", torch.randn(anchor.shape, generator=generator, device=trainer.device, dtype=anchor.dtype)),
    ]
    for name, queries in values:
        controls[f"query_{name}_score_text"] = trainer._predict_with_queries(loader, queries)["score_text"]
    if not torch.equal(anchor, trainer.model.query_cache):
        raise RuntimeError("query controls mutated the fixed anchor")
    np.savez_compressed(
        trainer.output_dir / "diagnostics" / "query_controls_val.npz",
        case_id=result.predictions["case_id"], proposition_id=np.asarray(trainer.proposition_ids),
        caption_support_binary=(result.predictions["label_text"] == 1).astype(np.int8),
        baseline_score_text=result.predictions["score_text"], **controls,
    )


def main(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config or "configs/ws.yaml").read_text(encoding="utf-8")) or {}
    weak_config, skincap, stage1 = config.get("weak_supervision", {}), config.get("skincap", {}), config.get("stage1", {})
    artifacts = load_weak_supervision_v3(
        labels_npz=args.labels_npz or weak_config.get("labels"), active_masks_npz=args.active_masks_npz or weak_config.get("active_masks"),
        active_csv=args.active_csv or weak_config.get("active_csv"), metadata_json=args.metadata_json or weak_config.get("metadata"),
    )
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    _seed_everything(seed)
    device = args.device or config.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = checkpoint.get("config", {})
    adaptation = run_config.get("bert_adaptation")
    anchor = checkpoint.get("query_anchor")
    anchor_hash = checkpoint.get("query_anchor_sha256")
    if adaptation not in {"frozen", "last1", "last2"} or not isinstance(anchor, torch.Tensor) or not isinstance(anchor_hash, str):
        raise ValueError("checkpoint is not a complete Round-B text-adaptation checkpoint")
    if _anchor_sha256(anchor) != anchor_hash:
        raise ValueError("checkpoint query anchor hash does not match")
    aliases = _load_aliases(args.disease_aliases_json, artifacts)
    common = dict(weak_supervision=artifacts, disease_aliases=aliases, image_root=args.image_root or skincap.get("image_root"))
    image_res = int(skincap.get("image_res", 224))
    datasets = {split: Stage1SkinCAPDataset(getattr(args, f"{split}_csv"), transform=_transform(image_res, False), split=split, **common)
                for split in ("train", "val", "test")}
    tokenizer_model = AutoTokenizer.from_pretrained(stage1["bert_model_name"])

    def tokenize(captions: list[str]) -> Mapping[str, torch.Tensor]:
        return tokenizer_model(captions, return_tensors="pt", padding=True, truncation=True,
                               max_length=int(stage1.get("max_caption_length", 128)))

    embed_dim = int(stage1.get("embed_dim", 768))
    model = Stage1EvidenceModel(
        image_encoder=ModelRes(stage1["image_encoder_name"], embed_dim), text_encoder=Text_Encoder_Bert(stage1["bert_model_name"], embed_dim),
        weak_supervision=artifacts, query_tokenizer=tokenize, embed_dim=embed_dim, num_heads=int(stage1.get("attention_heads", 8)),
        query_chunk_size=stage1.get("query_chunk_size"), training_scope="text_adapt", bert_adaptation=adaptation,
    )
    weights = build_pos_weight_artifacts(artifacts, pos_weight_cap=float(stage1.get("pos_weight_cap", 10.0)), text_target_semantics="caption_support_binary")
    trainer = Stage1Trainer(
        model=model, tokenizer=tokenize, weak_supervision=artifacts,
        loss_fn=MaskedBCELoss(weights.text_pos_weight, weights.image_pos_weight, text_target_semantics="caption_support_binary"),
        config=TrainerConfig(device=str(device), training_scope="text_adapt", text_target_semantics="caption_support_binary",
                             selection_min_present=int(stage1.get("selection_min_present", 5)), selection_min_absent=int(stage1.get("selection_min_absent", 5)), config=run_config),
        output_dir=args.output_dir, inference_only=True,
    )
    trainer.load_checkpoint(args.checkpoint, load_optimizer=False)
    if _anchor_sha256(model.query_cache) != anchor_hash or not torch.equal(model.query_cache.cpu(), anchor.cpu()):
        raise RuntimeError("saved query anchor was not restored exactly")
    before_image_hashes = run_config.get("round_b_image_hashes_before")
    after_image_hashes = _image_hashes(model)
    image_equal = None
    if before_image_hashes is not None:
        image_equal = compare_module_hashes(before_image_hashes, after_image_hashes)
        if not all(image_equal.values()):
            raise RuntimeError("export checkpoint image branch differs from its Round-B baseline")
    loaders = {split: _loader(dataset, int(skincap.get("eval_batch_size", 64)), int(skincap.get("num_workers", 2)), shuffle=False, seed=seed)
               for split, dataset in datasets.items()}
    results = {split: trainer.evaluate(loader, split=split, epoch=trainer.best_epoch or 0) for split, loader in loaders.items()}
    for split, result in results.items():
        trainer._save_predictions(result, f"scores_{split}_best.npz")
        (trainer.output_dir / "metrics" / f"text_evidence_{split}.json").write_text(
            json.dumps(_metrics_payload(result, trainer.text_trainable.detach().cpu().numpy()), indent=2), encoding="utf-8"
        )
    _write_query_controls(trainer, loaders["val"], results["val"], seed)
    (trainer.output_dir / "diagnostics" / "query_anchor_audit.json").write_text(json.dumps({
        "query_anchor_sha256": anchor_hash, "query_anchor_equal": True, "proposition_ids": list(trainer.proposition_ids), "shape": list(anchor.shape),
    }, indent=2), encoding="utf-8")
    (trainer.output_dir / "diagnostics" / "image_branch_freeze_audit.json").write_text(json.dumps({
        "pre_hash": before_image_hashes, "post_hash": after_image_hashes, "equal": image_equal,
    }, indent=2), encoding="utf-8")
    (trainer.output_dir / "export_manifest.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).name), "adaptation_mode": adaptation, "seed": seed,
        "query_anchor_sha256": anchor_hash, "weak_supervision_artifact_hashes": dict(artifacts.artifact_hashes),
        "inference_only": True, "diagnosis_supervision_used": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(_parser().parse_args())
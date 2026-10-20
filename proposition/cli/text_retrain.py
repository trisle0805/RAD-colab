"""Retrain only the Stage-1 textual evidence branch."""

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

from models.clip_tqn import ModelRes, Text_Encoder_Bert
from proposition.data import Stage1SkinCAPDataset, load_weak_supervision_v3
from proposition.engine import Stage1Trainer, TrainerConfig
from proposition.losses import MaskedBCELoss, build_pos_weight_artifacts, write_label_distribution
from proposition.models import Stage1EvidenceModel

from main_proposition import _load_aliases, _seed_everything, _transform


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "init_checkpoint", "labels_npz", "active_masks_npz", "active_csv", "metadata_json",
                 "train_csv", "val_csv", "test_csv", "image_root", "disease_aliases_json", "output_dir", "device"):
        parser.add_argument(f"--{name}", required=name in {"init_checkpoint", "output_dir"})
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--selection_min_present", type=int)
    parser.add_argument("--selection_min_absent", type=int)
    return parser


def _cfg(config: Mapping[str, Any], section: str, key: str, value: Any, default: Any = None) -> Any:
    return value if value is not None else config.get(section, {}).get(key, default)


def _loader(
    dataset: Stage1SkinCAPDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _module_hashes(model: torch.nn.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("image_encoder", "image_projection", "image_cross_attention", "image_head", "text_encoder"):
        digest = hashlib.sha256()
        module = getattr(model, name)
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def _score_equal(before: np.ndarray, after: np.ndarray, tolerance: float = 1e-7) -> bool:
    return bool(np.array_equal(before, after) or np.allclose(before, after, rtol=0.0, atol=tolerance, equal_nan=True))


def _distribution(scores: np.ndarray, labels: np.ndarray, semantics: str) -> dict[str, Any]:
    target = labels == 1 if semantics == "caption_support_binary" else labels
    valid = np.isfinite(scores)
    values = scores[valid]
    effective = valid & (target == 1)
    negatives = valid & (target == 0)
    return {
        "mean_score": float(values.mean()) if values.size else None,
        "std_score": float(values.std()) if values.size else None,
        "score_quantiles": {
            name: float(np.quantile(values, quantile)) if values.size else None
            for name, quantile in (("p01", .01), ("p05", .05), ("p25", .25), ("p50", .50),
                                   ("p75", .75), ("p95", .95), ("p99", .99))
        },
        "mean_score_effective_positive": float(scores[effective].mean()) if np.any(effective) else None,
        "mean_score_effective_negative": float(scores[negatives].mean()) if np.any(negatives) else None,
        "fraction_scores_ge_0_5": float(np.mean(values >= .5)) if values.size else None,
    }


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_single_class_count(
    labels: np.ndarray,
    trainable: np.ndarray,
    semantics: str,
) -> tuple[int, float]:
    target = labels == 1 if semantics == "caption_support_binary" else labels
    counts = []
    for column, enabled in enumerate(trainable):
        if not enabled:
            continue
        positive = int(np.count_nonzero(target[:, column] == 1))
        negative = int(np.count_nonzero(target[:, column] == 0))
        counts.append(positive == 0 or negative == 0)
    count = int(sum(counts))
    return count, count / len(counts) if counts else 0.0


def main(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config or "configs/ws.yaml").read_text(encoding="utf-8")) or {}
    weak = config.get("weak_supervision", {})
    artifacts = load_weak_supervision_v3(
        labels_npz=args.labels_npz or weak.get("labels"),
        active_masks_npz=args.active_masks_npz or weak.get("active_masks"),
        active_csv=args.active_csv or weak.get("active_csv"),
        metadata_json=args.metadata_json or weak.get("metadata"),
    )
    aliases = _load_aliases(args.disease_aliases_json, artifacts)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    _seed_everything(seed)
    device = args.device or config.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly")

    skincap, stage1 = config.get("skincap", {}), config.get("stage1", {})
    image_root = args.image_root or skincap.get("image_root")
    common = dict(weak_supervision=artifacts, disease_aliases=aliases, image_root=image_root)
    image_res = int(skincap.get("image_res", 224))
    train = Stage1SkinCAPDataset(args.train_csv or skincap.get("train_csv"), transform=_transform(image_res, False), split="train", **common)
    val = Stage1SkinCAPDataset(args.val_csv or skincap.get("val_csv"), transform=_transform(image_res, False), split="val", **common)
    test_path = args.test_csv or skincap.get("test_csv")
    test = Stage1SkinCAPDataset(test_path, transform=_transform(image_res, False), split="test", **common) if test_path else None

    tokenizer_model = AutoTokenizer.from_pretrained(stage1["bert_model_name"])
    def tokenize(captions: list[str]) -> Mapping[str, torch.Tensor]:
        return tokenizer_model(captions, return_tensors="pt", padding=True, truncation=True,
                               max_length=int(stage1.get("max_caption_length", 128)))

    embed_dim = int(stage1.get("embed_dim", 768))
    model = Stage1EvidenceModel(
        image_encoder=ModelRes(stage1["image_encoder_name"], embed_dim),
        text_encoder=Text_Encoder_Bert(stage1["bert_model_name"], embed_dim),
        weak_supervision=artifacts,
        query_tokenizer=tokenize,
        embed_dim=embed_dim,
        num_heads=int(stage1.get("attention_heads", 8)),
        query_chunk_size=stage1.get("query_chunk_size"),
        training_scope="text_only",
    )
    weights = build_pos_weight_artifacts(
        artifacts,
        pos_weight_cap=float(stage1.get("pos_weight_cap", 10.0)),
        text_target_semantics="caption_support_binary",
    )
    output = Path(args.output_dir)
    write_label_distribution(weights.rows, output / "supervision" / "label_distribution.csv")
    minimum_present = int(_cfg(config, "stage1", "selection_min_present", args.selection_min_present, 5))
    minimum_absent = int(_cfg(config, "stage1", "selection_min_absent", args.selection_min_absent, 5))
    trainer = Stage1Trainer(
        model=model,
        tokenizer=tokenize,
        weak_supervision=artifacts,
        loss_fn=MaskedBCELoss(
            weights.text_pos_weight, weights.image_pos_weight,
            text_target_semantics="caption_support_binary",
        ),
        config=TrainerConfig(
            device=str(device),
            early_stopping_patience=int(_cfg(config, "stage1", "early_stopping_patience", args.patience, 10)),
            learning_rate=float(stage1.get("lr", 1e-4)),
            weight_decay=float(stage1.get("weight_decay", 1e-4)),
            selection_min_present=minimum_present,
            selection_min_absent=minimum_absent,
            training_scope="text_only",
            text_target_semantics="caption_support_binary",
            config={**config, "training_scope": "text_only", "text_target_semantics": "caption_support_binary"},
        ),
        output_dir=output,
    )
    trainer.load_initial_weights(args.init_checkpoint)
    before_hashes = _module_hashes(model)
    val_loader = _loader(
        val,
        int(skincap.get("eval_batch_size", 64)),
        int(skincap.get("num_workers", 2)),
        shuffle=False,
        seed=seed,
    )
    init_checkpoint_sha256 = _file_sha256(args.init_checkpoint)
    init_checkpoint_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    init_git_commit = init_checkpoint_payload.get("git_commit")
    baseline = trainer.evaluate(val_loader, split="val", epoch=0)
    trainer.fit(
        _loader(
            train,
            int(skincap.get("batch_size", 32)),
            int(skincap.get("num_workers", 2)),
            shuffle=True,
            seed=seed,
        ),
        val_loader,
        max_epochs=int(args.epochs or stage1.get("epochs", 30)),
        test_dataloader=_loader(
            test,
            int(skincap.get("eval_batch_size", 64)),
            int(skincap.get("num_workers", 2)),
            shuffle=False,
            seed=seed,
        ) if test else None,
    )
    if trainer.best_epoch is None:
        raise RuntimeError("text-only retraining produced no eligible validation checkpoint")
    after_hashes = _module_hashes(model)
    hash_flags = {name: before_hashes[name] == after_hashes[name] for name in before_hashes}
    if not all(hash_flags.values()):
        raise RuntimeError("a frozen module changed during text-only retraining")
    final_val = trainer.evaluate(val_loader, split="val", epoch=trainer.best_epoch or 0)
    image_score_max_abs_diff = float(
        np.nanmax(np.abs(baseline.predictions["score_image"] - final_val.predictions["score_image"]))
    )
    if not _score_equal(baseline.predictions["score_image"], final_val.predictions["score_image"]):
        raise RuntimeError("image scores changed during text-only retraining")
    loaders = {
        "train": _loader(
            train,
            int(skincap.get("eval_batch_size", 64)),
            int(skincap.get("num_workers", 2)),
            shuffle=False,
            seed=seed,
        ),
        "val": val_loader,
    }
    if test:
        loaders["test"] = _loader(
            test,
            int(skincap.get("eval_batch_size", 64)),
            int(skincap.get("num_workers", 2)),
            shuffle=False,
            seed=seed,
        )
    trainer.export_best_predictions(loaders)

    summaries: dict[str, Any] = {}
    for split, loader in loaders.items():
        result = trainer.evaluate(loader, split=split, epoch=trainer.best_epoch or 0)
        text_metrics = result.metrics["text"]
        score_mask = trainer.text_trainable.detach().cpu().numpy()
        summaries[split] = {
            "text_macro_roc_auc": text_metrics["macro_roc_auc"],
            "text_macro_pr_auc": text_metrics["macro_pr_auc"],
            "n_eligible_strict": text_metrics["n_eligible"],
            **_distribution(
                result.predictions["score_text"][:, score_mask],
                result.predictions["label_text"][:, score_mask],
                "caption_support_binary",
            ),
        }
        count, fraction = _effective_single_class_count(
            result.predictions["label_text"],
            score_mask,
            "caption_support_binary",
        )
        summaries[split]["n_effectively_single_class"] = count
        summaries[split]["fraction_effectively_single_class"] = fraction
    (output / "metrics" / "text_retrain_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (output / "diagnostics" / "frozen_module_hashes.json").write_text(
        json.dumps({"pre": before_hashes, "post": after_hashes, "equal": hash_flags}, indent=2), encoding="utf-8"
    )
    (output / "diagnostics" / "score_distribution_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    manifest = {
        "training_scope": "text_only",
        "text_target_semantics": "caption_support_binary",
        "selection": {"metric": "validation_text_macro_roc_auc", "min_present": minimum_present, "min_absent_effective": minimum_absent},
        "best_epoch": trainer.best_epoch,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "init_checkpoint_git_commit": init_git_commit,
        "frozen_module_hashes_equal": all(hash_flags.values()),
        "image_scores_unchanged": True,
        "image_score_max_abs_diff": image_score_max_abs_diff,
        "proposition_ids": list(trainer.proposition_ids),
        "artifact_hashes": dict(artifacts.artifact_hashes),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(_parser().parse_args())

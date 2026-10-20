"""Train/validate only the Round-B BioClinicalBERT textual-evidence adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from proposition.engine.text_encoder_adaptation import compare_module_hashes, component_hashes, selected_bert_layers
from proposition.losses import MaskedBCELoss, build_pos_weight_artifacts, write_label_distribution
from proposition.models import Stage1EvidenceModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config", "init_checkpoint", "labels_npz", "active_masks_npz", "active_csv", "metadata_json",
        "train_csv", "val_csv", "image_root", "disease_aliases_json", "output_dir", "device",
    ):
        parser.add_argument(f"--{name}", required=name in {"init_checkpoint", "output_dir"})
    parser.add_argument("--adaptation", choices=("frozen", "last1", "last2"), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--head_lr", type=float)
    parser.add_argument("--bert_lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--selection_min_present", type=int)
    parser.add_argument("--selection_min_absent", type=int)
    return parser


def _value(config: Mapping[str, Any], key: str, supplied: Any, default: Any) -> Any:
    return supplied if supplied is not None else config.get("stage1", {}).get(key, default)


def _loader(dataset: Stage1SkinCAPDataset, batch_size: int, workers: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=generator)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _anchor_sha256(anchor: torch.Tensor) -> str:
    anchor = anchor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(anchor.shape)).encode("utf-8"))
    digest.update(str(anchor.dtype).encode("utf-8"))
    digest.update(anchor.numpy().tobytes())
    return digest.hexdigest()


def _image_hashes(model: Stage1EvidenceModel) -> dict[str, str]:
    from proposition.engine.text_encoder_adaptation import module_sha256

    return {name: module_sha256(getattr(model, name)) for name in (
        "image_encoder", "image_projection", "image_cross_attention", "image_head",
    )}


def _assert_image_invariance(before: np.ndarray, after: np.ndarray, tolerance: float = 1e-7) -> float:
    maximum = float(np.nanmax(np.abs(before - after))) if before.size else 0.0
    if not np.allclose(before, after, rtol=0.0, atol=tolerance, equal_nan=True):
        raise RuntimeError(f"image scores changed during text adaptation (max abs diff={maximum})")
    return maximum

def _parameter_names(model: torch.nn.Module, parameters: list[torch.nn.Parameter]) -> list[str]:
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    return [by_id[id(parameter)] for parameter in parameters]

def _audit_round_b_optimizer(
    model: Stage1EvidenceModel,
    optimizer: torch.optim.Optimizer,
    adaptation: str,
    head_lr: float,
    bert_lr: float,
) -> dict[str, Any]:
    """Fail closed unless the Round-B optimizer has its exact two-group contract."""
    head_parameters = [parameter for name in ("text_cross_attention", "text_head") for parameter in getattr(model, name).parameters()]
    bert_parameters = [parameter for layer in selected_bert_layers(model.text_encoder, adaptation) for parameter in layer.parameters()]
    expected_head_ids, expected_bert_ids = {id(parameter) for parameter in head_parameters}, {id(parameter) for parameter in bert_parameters}
    if expected_head_ids & expected_bert_ids:
        raise RuntimeError("Round-B head and BERT parameter sets overlap")
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("Round-B optimizer must contain exactly head and BERT parameter groups")
    head_group, bert_group = optimizer.param_groups
    head_ids = {id(parameter) for parameter in head_group["params"]}
    bert_ids = {id(parameter) for parameter in bert_group["params"]}
    if head_ids != expected_head_ids or bert_ids != expected_bert_ids:
        raise RuntimeError("Round-B optimizer parameter groups do not match selected heads and BERT layers")
    if head_ids & bert_ids or (head_ids | bert_ids) != (expected_head_ids | expected_bert_ids):
        raise RuntimeError("Round-B optimizer parameter groups overlap or omit expected parameters")
    if any(not parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"]):
        raise RuntimeError("Round-B optimizer contains a frozen parameter")
    if float(head_group["lr"]) != head_lr or float(bert_group["lr"]) != bert_lr:
        raise RuntimeError("Round-B optimizer learning rates do not match head_lr and bert_lr")
    return {
        "head": {"lr": float(head_group["lr"]), "parameter_names": _parameter_names(model, head_parameters), "parameter_count": sum(p.numel() for p in head_parameters)},
        "bert": {"lr": float(bert_group["lr"]), "parameter_names": _parameter_names(model, bert_parameters), "parameter_count": sum(p.numel() for p in bert_parameters), "adaptation": adaptation},
        "groups_overlap": False,
    }


def _persist_round_b_pretraining_audits(
    trainer: Stage1Trainer,
    output: Path,
    *,
    image_before: Mapping[str, str],
    anchor_hash: str,
    optimizer_audit: Mapping[str, Any],
) -> None:
    """Persist the initialization state that finalization must audit against."""
    if isinstance(trainer.config.config, dict):
        trainer.config.config["round_b_image_hashes_before"] = dict(image_before)
        trainer.config.config["round_b_query_anchor_sha256"] = anchor_hash
        trainer.config.config["round_b_optimizer_audit"] = dict(optimizer_audit)
        trainer.write_run_config()
    (output / "diagnostics" / "optimizer_parameter_audit.json").write_text(
        json.dumps(dict(optimizer_audit), indent=2), encoding="utf-8"
    )


def _finalize_round_b(
    trainer: Stage1Trainer,
    model: Stage1EvidenceModel,
    val_loader: DataLoader,
    *,
    baseline: Any,
    anchor: torch.Tensor,
    anchor_hash: str,
    image_before: Mapping[str, str],
    components_before: Mapping[str, str],
    adaptation: str,
    output: Path,
    bert_parameters: list[torch.nn.Parameter],
    init_checkpoint: str | Path,
    seed: int,
    head_lr: float,
    bert_lr: float,
    weight_decay: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Audit and serialize the restored best Round-B checkpoint state."""
    if trainer.best_epoch is None:
        raise RuntimeError("text adaptation produced no strict 5/5 eligible validation checkpoint")
    restored_anchor = model.query_cache.detach()
    if _anchor_sha256(restored_anchor) != anchor_hash or not torch.equal(restored_anchor.cpu(), anchor.cpu()):
        raise RuntimeError("query anchor changed during Round-B adaptation")

    # Stage1Trainer.fit restores best.pt before returning; every audit below is
    # deliberately against that restored best state rather than the latest epoch.
    image_after, components_after = _image_hashes(model), component_hashes(model)
    image_equal = compare_module_hashes(image_before, image_after)
    if not all(image_equal.values()):
        raise RuntimeError("a frozen image module changed during text adaptation")
    final = trainer.evaluate(val_loader, split="val", epoch=trainer.best_epoch)
    image_diff = _assert_image_invariance(baseline.predictions["score_image"], final.predictions["score_image"])

    component_equal = compare_module_hashes(components_before, components_after)
    changed = {name: not equal for name, equal in component_equal.items()}
    allowed = {"text_cross_attention", "text_head"}
    layers = len(getattr(model.text_encoder.bert_model.encoder, "layer"))
    if adaptation == "last1":
        allowed.add(f"bert_encoder_layer_{layers - 1}")
    elif adaptation == "last2":
        allowed.update({f"bert_encoder_layer_{layers - 2}", f"bert_encoder_layer_{layers - 1}"})
    invalid = [name for name, did_change in changed.items() if did_change and name not in allowed]
    if invalid:
        raise RuntimeError(f"frozen text components changed: {invalid}")
    adapted_layers = [name for name in allowed if name.startswith("bert_encoder_layer_")]
    unchanged_adapted_layers = [name for name in adapted_layers if not changed[name]]
    if trainer.epoch > 0 and unchanged_adapted_layers:
        raise RuntimeError(
            "adapted BERT layers did not change after training; verify caption-gradient flow: "
            f"{unchanged_adapted_layers}"
        )
    audit = {"adaptation_mode": adaptation, "pre_hash": dict(components_before), "post_hash": components_after,
             "changed": changed, "allowed_to_change": {name: name in allowed for name in changed},
             "unchanged_adapted_bert_layers": unchanged_adapted_layers,
             "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
             "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "trainable_bert_parameter_count": sum(p.numel() for p in bert_parameters)}
    (output / "diagnostics" / "module_parameter_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "diagnostics" / "image_branch_freeze_audit.json").write_text(json.dumps({
        "pre_hash": dict(image_before), "post_hash": image_after, "equal": image_equal,
        "max_abs_image_score_diff": image_diff,
    }, indent=2), encoding="utf-8")
    (output / "diagnostics" / "query_anchor_audit.json").write_text(json.dumps({
        "query_anchor_sha256_before": anchor_hash, "query_anchor_sha256_after": _anchor_sha256(restored_anchor),
        "query_anchor_equal": True, "proposition_ids": list(trainer.proposition_ids), "shape": list(anchor.shape),
    }, indent=2), encoding="utf-8")
    summary = {"trainable_parameter_count": audit["trainable_parameter_count"], "trainable_bert_parameter_count": audit["trainable_bert_parameter_count"],
               "best_epoch": trainer.best_epoch, "best_validation_text_macro_auc": trainer.best_metric,
               "best_validation_text_bce": trainer.best_loss}
    if isinstance(trainer.config.config, dict):
        trainer.config.config.update({
            "best_epoch": trainer.best_epoch,
            "best_validation_text_macro_auc": trainer.best_metric,
            "best_validation_text_bce": trainer.best_loss,
            "round_b_image_hashes_after": image_after,
            "round_b_query_anchor_sha256_after": _anchor_sha256(restored_anchor),
            "round_b_query_anchor_equal": True,
        })
        trainer.write_run_config()
    (output / "metrics" / "text_adapt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    init_payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    manifest = {**summary, "git_commit": _git_commit(), "adaptation_mode": adaptation, "seed": seed,
                "init_checkpoint_sha256": _sha256_file(init_checkpoint), "init_checkpoint_git_commit_if_available": init_payload.get("git_commit"),
                "weak_supervision_artifact_hashes": dict(trainer.artifacts.artifact_hashes), "query_anchor_sha256": anchor_hash,
                "head_lr": head_lr, "bert_lr": bert_lr, "weight_decay": weight_decay, "max_grad_norm": max_grad_norm,
                "image_branch_unchanged": True, "image_score_max_abs_diff": image_diff,
                "diagnosis_supervision_used": False}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config or "configs/ws.yaml").read_text(encoding="utf-8")) or {}
    weak_config, skincap, stage1 = config.get("weak_supervision", {}), config.get("skincap", {}), config.get("stage1", {})
    artifacts = load_weak_supervision_v3(
        labels_npz=args.labels_npz or weak_config.get("labels"), active_masks_npz=args.active_masks_npz or weak_config.get("active_masks"),
        active_csv=args.active_csv or weak_config.get("active_csv"), metadata_json=args.metadata_json or weak_config.get("metadata"),
    )
    aliases = _load_aliases(args.disease_aliases_json, artifacts)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    _seed_everything(seed)
    device = args.device or config.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    image_root = args.image_root or skincap.get("image_root")
    common = dict(weak_supervision=artifacts, disease_aliases=aliases, image_root=image_root)
    image_res = int(skincap.get("image_res", 224))
    train = Stage1SkinCAPDataset(args.train_csv or skincap.get("train_csv"), transform=_transform(image_res, False), split="train", **common)
    val = Stage1SkinCAPDataset(args.val_csv or skincap.get("val_csv"), transform=_transform(image_res, False), split="val", **common)
    tokenizer_model = AutoTokenizer.from_pretrained(stage1["bert_model_name"])

    def tokenize(captions: list[str]) -> Mapping[str, torch.Tensor]:
        return tokenizer_model(captions, return_tensors="pt", padding=True, truncation=True,
                               max_length=int(stage1.get("max_caption_length", 128)))

    embed_dim = int(stage1.get("embed_dim", 768))
    model = Stage1EvidenceModel(
        image_encoder=ModelRes(stage1["image_encoder_name"], embed_dim),
        text_encoder=Text_Encoder_Bert(stage1["bert_model_name"], embed_dim),
        weak_supervision=artifacts, query_tokenizer=tokenize, embed_dim=embed_dim,
        num_heads=int(stage1.get("attention_heads", 8)), query_chunk_size=stage1.get("query_chunk_size"),
        # Anchor creation must precede unfreezing any BERT layer.
        training_scope="text_only", bert_adaptation="frozen",
    )
    weights = build_pos_weight_artifacts(artifacts, pos_weight_cap=float(stage1.get("pos_weight_cap", 10.0)),
                                         text_target_semantics="caption_support_binary")
    output = Path(args.output_dir)
    (output / "diagnostics").mkdir(parents=True, exist_ok=True)
    write_label_distribution(weights.rows, output / "supervision" / "label_distribution.csv")
    head_lr = float(_value(config, "head_lr", args.head_lr, 1e-4))
    bert_lr = float(_value(config, "bert_lr", args.bert_lr, 1e-5))
    weight_decay = float(_value(config, "weight_decay", args.weight_decay, 1e-4))
    max_grad_norm = float(_value(config, "max_grad_norm", args.max_grad_norm, 1.0))
    bootstrap = Stage1Trainer(
        model=model, tokenizer=tokenize, weak_supervision=artifacts,
        loss_fn=MaskedBCELoss(weights.text_pos_weight, weights.image_pos_weight, text_target_semantics="caption_support_binary"),
        config=TrainerConfig(
            device=str(device), early_stopping_patience=int(_value(config, "early_stopping_patience", args.patience, 10)),
            learning_rate=head_lr, weight_decay=weight_decay, max_grad_norm=max_grad_norm,
            selection_min_present=int(_value(config, "selection_min_present", args.selection_min_present, 5)),
            selection_min_absent=int(_value(config, "selection_min_absent", args.selection_min_absent, 5)),
            training_scope="text_only", text_target_semantics="caption_support_binary",
            config={**config, "training_scope": "text_adapt", "bert_adaptation": args.adaptation, "head_lr": head_lr,
                    "bert_lr": bert_lr, "weight_decay": weight_decay, "max_grad_norm": max_grad_norm, "seed": seed},
        ),
    )
    bootstrap.load_initial_weights(args.init_checkpoint)
    model.text_encoder.eval()
    anchor = model.create_query_anchor()
    anchor_hash = _anchor_sha256(anchor)
    np.savez_compressed(output / "diagnostics" / "query_anchor.npz", query_anchor=anchor.cpu().numpy(),
                        proposition_id=np.asarray(bootstrap.proposition_ids))
    (output / "diagnostics" / "query_anchor_audit.json").write_text(json.dumps({
        "query_anchor_sha256_before": anchor_hash, "proposition_ids": list(bootstrap.proposition_ids),
        "shape": list(anchor.shape), "detached": not anchor.requires_grad,
    }, indent=2), encoding="utf-8")
    model.bert_adaptation = args.adaptation
    model.set_training_scope("text_adapt")
    head_parameters = [parameter for name in ("text_cross_attention", "text_head") for parameter in getattr(model, name).parameters()]
    bert_parameters = [parameter for parameter in model.text_encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW((
        {"params": head_parameters, "lr": head_lr},
        {"params": bert_parameters, "lr": bert_lr},
    ), weight_decay=weight_decay)
    optimizer_audit = _audit_round_b_optimizer(model, optimizer, args.adaptation, head_lr, bert_lr)
    trainer = Stage1Trainer(
        model=model, tokenizer=tokenize, weak_supervision=artifacts,
        loss_fn=MaskedBCELoss(weights.text_pos_weight, weights.image_pos_weight, text_target_semantics="caption_support_binary"),
        optimizer=optimizer,
        config=TrainerConfig(
            device=str(device), early_stopping_patience=int(_value(config, "early_stopping_patience", args.patience, 10)),
            learning_rate=head_lr, weight_decay=weight_decay, max_grad_norm=max_grad_norm,
            selection_min_present=int(_value(config, "selection_min_present", args.selection_min_present, 5)),
            selection_min_absent=int(_value(config, "selection_min_absent", args.selection_min_absent, 5)),
            training_scope="text_adapt", text_target_semantics="caption_support_binary",
            config={**config, "training_scope": "text_adapt", "bert_adaptation": args.adaptation, "head_lr": head_lr,
                    "bert_lr": bert_lr, "weight_decay": weight_decay, "max_grad_norm": max_grad_norm, "seed": seed},
        ), output_dir=output,
    )
    image_before, components_before = _image_hashes(model), component_hashes(model)
    _persist_round_b_pretraining_audits(
        trainer,
        output,
        image_before=image_before,
        anchor_hash=anchor_hash,
        optimizer_audit=optimizer_audit,
    )
    val_loader = _loader(val, int(skincap.get("eval_batch_size", 64)), int(skincap.get("num_workers", 2)), shuffle=False, seed=seed)
    baseline = trainer.evaluate(val_loader, split="val", epoch=0)
    trainer.fit(_loader(train, int(skincap.get("batch_size", 32)), int(skincap.get("num_workers", 2)), shuffle=True, seed=seed),
                val_loader, max_epochs=int(args.epochs or stage1.get("epochs", 30)))
    _finalize_round_b(
        trainer,
        model,
        val_loader,
        baseline=baseline,
        anchor=anchor,
        anchor_hash=anchor_hash,
        image_before=image_before,
        components_before=components_before,
        adaptation=args.adaptation,
        output=output,
        bert_parameters=bert_parameters,
        init_checkpoint=args.init_checkpoint,
        seed=seed,
        head_lr=head_lr,
        bert_lr=bert_lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )


if __name__ == "__main__":
    main(_parser().parse_args())
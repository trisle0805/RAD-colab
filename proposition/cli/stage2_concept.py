"""Train an interpretable learned Stage-2 linear concept head."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from proposition.data.stage2_scores import Stage2Dataset, build_stage2_support, load_stage2_disease_targets, load_stage2_propositions, load_stage2_score_artifact, validate_stage2_splits
from proposition.data.weak_supervision import sha256_file
from proposition.engine.stage2_concept_audit import concept_weight_rows, disease_sparsity_rows, same_disease_contribution_fraction
from proposition.engine.stage2_concept_metrics import bootstrap_topk_confidence_intervals
from proposition.engine.stage2_concept_trainer import Stage2ConceptTrainer, Stage2ConceptTrainerConfig
from proposition.models.stage2_concept_head import Stage2ConceptHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scalar-only learned Stage-2 concept head.")
    for split in ("train", "val"):
        parser.add_argument(f"--{split}_scores", required=True)
        parser.add_argument(f"--{split}_csv", required=True)
    parser.add_argument("--test_scores")
    parser.add_argument("--test_csv")
    parser.add_argument("--evaluate_test", action="store_true", help="Evaluate the selected configuration on the held-out test split.")
    parser.add_argument("--active_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head_mode", choices=("local", "global"), required=True)
    parser.add_argument("--routing_mode", choices=("image_morphology", "full_image_morphology_text_other", "full_fused"), required=True)
    parser.add_argument("--regularization", choices=("none", "l1", "l2"), default="none")
    parser.add_argument("--lambda_reg", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--nonzero_threshold", type=float, default=1e-4)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    args = parser.parse_args()
    if args.evaluate_test and (not args.test_scores or not args.test_csv):
        parser.error("--evaluate_test requires --test_scores and --test_csv")
    return args


def main() -> None:
    args = parse_args()
    if args.head_mode == "local" and args.regularization == "l1":
        raise ValueError("S2-L supports regularization=none or l2; use S2-G for L1 experiments")
    _seed(args.seed)
    splits = ("train", "val", "test") if args.evaluate_test else ("train", "val")
    paths = {f"{split}_scores": Path(getattr(args, f"{split}_scores")) for split in splits}
    paths.update({f"{split}_csv": Path(getattr(args, f"{split}_csv")) for split in splits})
    paths["active_csv"] = Path(args.active_csv)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    artifacts = {split: load_stage2_score_artifact(paths[f"{split}_scores"]) for split in splits}
    validate_stage2_splits(artifacts["train"], artifacts["val"], artifacts.get("test"))
    propositions = load_stage2_propositions(paths["active_csv"], artifacts["train"].proposition_ids)
    routed = {split: build_stage2_support(artifacts[split], propositions, args.routing_mode) for split in splits}
    if any(not np.array_equal(routed["train"].proposition_mask, routed[split].proposition_mask) for split in splits[1:]):
        raise ValueError("routing masks differ across splits")
    targets = {split: load_stage2_disease_targets(paths[f"{split}_csv"], artifacts[split].case_ids) for split in splits}
    disease_ids = targets["train"].disease_ids
    if any(targets[split].disease_ids != disease_ids for split in splits[1:]):
        raise ValueError("disease label columns/order must match across splits")
    disease_index = {disease: index for index, disease in enumerate(disease_ids)}
    proposition_diseases = tuple(item.disease_id for item in propositions)
    model = Stage2ConceptHead(proposition_disease_index=torch.tensor([disease_index[disease] for disease in proposition_diseases]),
                              proposition_polarity=torch.tensor([item.polarity for item in propositions]),
                              proposition_mask=torch.from_numpy(routed["train"].proposition_mask), num_diseases=len(disease_ids), head_mode=args.head_mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "input_contract.json", {"input_hashes": hashes, "routing_mode": args.routing_mode, "proposition_ids": artifacts["train"].proposition_ids.tolist(), "disease_ids": list(disease_ids), "case_orders": {split: artifacts[split].case_ids.tolist() for split in splits}, "proposition_metadata_hash": hashes["active_csv"]})
    trainer = Stage2ConceptTrainer(model=model, disease_ids=disease_ids, proposition_ids=artifacts["train"].proposition_ids.tolist(), routing_mode=args.routing_mode,
                                   input_hashes=hashes, score_source=routed["train"].score_source.tolist(),
                                   config=Stage2ConceptTrainerConfig(device=args.device, learning_rate=args.learning_rate, regularization=args.regularization,
                                   lambda_reg=args.lambda_reg, max_epochs=args.epochs, early_stopping_patience=args.patience, batch_size=args.batch_size, base_seed=args.seed), output_dir=output_dir)
    datasets = {split: Stage2Dataset(artifacts[split].case_ids, routed[split].support_scores, targets[split].targets) for split in splits}
    trainer.fit(datasets["train"], datasets["val"])
    results = {"val": trainer.export_best_split(datasets["val"], "val")}
    if args.evaluate_test:
        results["test"] = trainer.export_best_split(datasets["test"], "test")
        _write_json(output_dir / "metrics" / "test_confidence_intervals.json", bootstrap_topk_confidence_intervals(results["test"]["outputs"]["logits"], datasets["test"].targets, seed=args.seed, n_bootstrap=args.bootstrap_samples))
    _write_audits(output_dir, model, results["val"]["outputs"], disease_ids, artifacts["train"].proposition_ids.tolist(), proposition_diseases, args.nonzero_threshold)
    after = {name: sha256_file(path) for name, path in paths.items()}
    if after != hashes:
        raise RuntimeError("input artifact hashes changed during Stage-2 execution")
    _write_json(output_dir / "stage2_concept_manifest.json", {"head_mode": args.head_mode, "routing_mode": args.routing_mode, "regularization": args.regularization, "lambda_reg": args.lambda_reg, "nonzero_threshold": args.nonzero_threshold, "input_hashes": hashes, "proposition_ids": artifacts["train"].proposition_ids.tolist(), "disease_ids": list(disease_ids), "seed": args.seed, "evaluate_test": args.evaluate_test, "best_epoch": trainer.best_epoch, "best_validation_top_1": trainer.best_top_1_accuracy, "best_validation_cross_entropy": trainer.best_cross_entropy, "git_commit": _git_commit()})
    summary = {"best_epoch": trainer.best_epoch, "val": _summary(results["val"])}
    if args.evaluate_test:
        summary["test"] = _summary(results["test"])
    print(json.dumps(summary, sort_keys=True))


def _write_audits(output_dir: Path, model: Stage2ConceptHead, outputs: dict[str, np.ndarray], disease_ids: tuple[str, ...], proposition_ids: list[str], proposition_disease_ids: tuple[str, ...], threshold: float) -> None:
    weights = outputs["weights"]
    _write_rows(output_dir / "diagnostics" / "concept_weights.csv", concept_weight_rows(weights, disease_ids, proposition_ids, proposition_disease_ids, nonzero_threshold=threshold))
    sparsity = disease_sparsity_rows(weights, disease_ids, proposition_disease_ids, nonzero_threshold=threshold)
    _write_rows(output_dir / "diagnostics" / "disease_sparsity.csv", sparsity)
    fractions, summary = same_disease_contribution_fraction(outputs["input_scores"], weights, disease_ids, proposition_disease_ids)
    if model.head_mode == "local" and not np.allclose([row["same_disease_contribution_fraction"] for row in fractions], 1.0):
        raise AssertionError("S2-L must have a same-disease contribution fraction of 1.0")
    _write_rows(output_dir / "diagnostics" / "same_disease_contribution_fraction.csv", fractions)
    _write_json(output_dir / "diagnostics" / "same_disease_contribution_fraction_summary.json", summary)
    reconstruction = outputs["reconstructed_logits"] - outputs["logits"]
    ranking = np.argsort(-outputs["logits"], axis=1, kind="mergesort")
    rows = np.arange(len(ranking))
    top1, top2 = ranking[:, 0], ranking[:, 1]
    margin = outputs["logits"][rows, top1] - outputs["logits"][rows, top2]
    bias_margin = outputs["bias"][top1] - outputs["bias"][top2]
    delta = outputs["contributions"][rows, top1] - outputs["contributions"][rows, top2]
    margin_error = bias_margin + delta.sum(axis=1) - margin
    _write_json(output_dir / "diagnostics" / "contribution_reconstruction_check.json", {
        "max_abs_logit_error": float(np.abs(reconstruction).max()),
        "max_abs_margin_error": float(np.abs(margin_error).max()),
        "passes": bool(np.allclose(outputs["reconstructed_logits"], outputs["logits"], atol=1e-6) and np.allclose(margin_error, 0.0, atol=1e-6)),
    })


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result["metrics"].items() if key not in {"per_disease", "probabilities", "confusion_matrix"}}

def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")

if __name__ == "__main__":
    main()
"""Run one immutable, frozen-score Stage-2 diagnosis experiment."""

from __future__ import annotations

import argparse
import csv
import subprocess
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from proposition.data.stage2_scores import (
    Stage2Dataset, build_stage2_support, load_stage2_disease_targets,
    load_stage2_propositions, load_stage2_score_artifact, validate_stage2_splits,
)
from proposition.data.weak_supervision import sha256_file
from proposition.engine.stage2_trainer import Stage2Trainer, Stage2TrainerConfig
from proposition.models.stage2_diagnosis import PropositionDiagnosisLayer, validate_routing_disease_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 diagnosis calibration from frozen Stage-1 score NPZs.")
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}_scores", required=True)
        parser.add_argument(f"--{split}_csv", required=True)
    parser.add_argument("--active_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage1_manifest")
    parser.add_argument("--stage1_evidence_hashes")
    parser.add_argument("--mode", required=True, choices=("image_morphology", "full_fused", "full_image_morphology_text_other"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--selection_metric", default="val_loss", choices=("val_loss", "macro_roc_auc", "macro_f1", "micro_f1", "balanced_accuracy"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); _seed(args.seed)
    paths = {f"{split}_scores": Path(getattr(args, f"{split}_scores")) for split in ("train", "val", "test")}
    paths.update({f"{split}_csv": Path(getattr(args, f"{split}_csv")) for split in ("train", "val", "test")})
    paths["active_csv"] = Path(args.active_csv)
    for optional in ("stage1_manifest", "stage1_evidence_hashes"):
        value = getattr(args, optional)
        if value: paths[optional] = Path(value)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    artifacts = {split: load_stage2_score_artifact(paths[f"{split}_scores"]) for split in ("train", "val", "test")}
    validate_stage2_splits(artifacts["train"], artifacts["val"], artifacts["test"])
    propositions = load_stage2_propositions(paths["active_csv"], artifacts["train"].proposition_ids)
    routed = {split: build_stage2_support(artifacts[split], propositions, args.mode) for split in ("train", "val", "test")}
    targets = {split: load_stage2_disease_targets(paths[f"{split}_csv"], artifacts[split].case_ids) for split in ("train", "val", "test")}
    disease_ids = targets["train"].disease_ids
    if any(targets[split].disease_ids != disease_ids for split in ("val", "test")):
        raise ValueError("disease label columns/order must match across train, val, and test CSVs")
    proposition_diseases = tuple(item.disease_id for item in propositions)
    validate_routing_disease_coverage(proposition_disease_ids=proposition_diseases, disease_ids=disease_ids, proposition_mask=torch.from_numpy(routed["train"].proposition_mask))
    disease_index = {disease: index for index, disease in enumerate(disease_ids)}
    model = PropositionDiagnosisLayer(
        proposition_disease_index=torch.tensor([disease_index[disease] for disease in proposition_diseases]),
        proposition_polarity=torch.tensor([item.polarity for item in propositions]),
        proposition_mask=torch.from_numpy(routed["train"].proposition_mask), num_diseases=len(disease_ids),
    )
    config = Stage2TrainerConfig(device=args.device, learning_rate=args.lr, weight_decay=args.weight_decay, max_epochs=args.epochs, early_stopping_patience=args.patience, selection_metric=args.selection_metric, threshold=args.threshold, batch_size=args.batch_size)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "input_contract.json", {"hashes": hashes, "routing_mode": args.mode, "proposition_ids": artifacts["train"].proposition_ids.tolist(), "disease_ids": list(disease_ids)})
    trainer = Stage2Trainer(model=model, disease_ids=disease_ids, proposition_ids=artifacts["train"].proposition_ids.tolist(), routing_mode=args.mode, input_hashes=hashes, score_source=routed["train"].score_source.tolist(), config=config, output_dir=output_dir)
    datasets = {split: Stage2Dataset(artifacts[split].case_ids, routed[split].support_scores, targets[split].targets) for split in ("train", "val", "test")}
    trainer.fit(datasets["train"], datasets["val"])
    results = {split: trainer.export_best_split(dataset, split) for split, dataset in datasets.items()}
    _write_routing(output_dir / "diagnostics" / "proposition_routing.csv", propositions, routed["train"])
    _write_counts(output_dir / "diagnostics" / "disease_proposition_counts.csv", disease_ids, proposition_diseases, routed["train"].proposition_mask)
    _write_calibration(output_dir / "diagnostics" / "calibration_parameters.csv", disease_ids, results["val"]["outputs"])
    after = {name: sha256_file(path) for name, path in paths.items()}
    if after != hashes:
        raise RuntimeError("input artifact hashes changed during Stage-2 execution")
    _write_json(output_dir / "stage2_manifest.json", {"routing_mode": args.mode, "proposition_ids": artifacts["train"].proposition_ids.tolist(), "disease_ids": list(disease_ids), "input_hashes": hashes, "seed": args.seed, "hyperparameters": vars(args), "best_epoch": trainer.best_epoch, "best_validation_metric": trainer.best_metric, "selection_metric": args.selection_metric, "selection_mode": "min" if args.selection_metric == "val_loss" else "max", "stage2_git_commit": _git_commit(), "stage1_provenance": {key: hashes[key] for key in ("stage1_manifest", "stage1_evidence_hashes") if key in hashes}})
    print(json.dumps({"best_epoch": trainer.best_epoch, "best_validation_metric": trainer.best_metric, "test_metrics": {key: value for key, value in results["test"]["metrics"].items() if key != "per_disease"}}, sort_keys=True, default=_json_default))


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_routing(path: Path, propositions: Any, routed: Any) -> None:
    rows = [{"proposition_id": prop.proposition_id, "disease_id": prop.disease_id, "category": prop.category, "polarity": prop.polarity, "proposition_mask": bool(mask), "score_source": str(source)} for prop, mask, source in zip(propositions, routed.proposition_mask, routed.score_source)]
    _write_rows(path, rows)


def _write_counts(path: Path, diseases: Any, proposition_diseases: Any, mask: np.ndarray) -> None:
    _write_rows(path, [{"disease_id": disease, "proposition_count": sum(1 for value, chosen in zip(proposition_diseases, mask) if value == disease and chosen)} for disease in diseases])


def _write_calibration(path: Path, diseases: Any, output: Any) -> None:
    _write_rows(path, [{"disease_id": disease, "gamma": float(gamma), "bias": float(bias)} for disease, gamma, bias in zip(diseases, output["gamma"], output["bias"])])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)

def _git_commit() -> str | None:
    """Return the source revision when the CLI runs inside a Git checkout."""
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    main()
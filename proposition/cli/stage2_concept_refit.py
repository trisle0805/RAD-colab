"""Refit the CV-selected Round-A concept configuration on train+validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from proposition.data.stage2_concept_features import build_stage2_concept_features
from proposition.data.stage2_scores import Stage2Dataset, load_stage2_disease_targets, load_stage2_propositions, load_stage2_score_artifact, validate_stage2_splits
from proposition.data.weak_supervision import sha256_file
from proposition.engine.stage2_concept_cv import Stage2ConceptCandidate, pool_stage2_datasets
from proposition.engine.stage2_concept_ensemble import average_frozen_heads, derive_refit_epochs, stability_rows
from proposition.engine.stage2_concept_refit import refit_fixed_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refit and average a frozen Round-A scalar concept configuration.")
    parser.add_argument("--train_scores", required=True)
    parser.add_argument("--val_scores", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--active_csv", required=True)
    parser.add_argument("--selected_config_json", required=True)
    parser.add_argument("--cv_summary_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("Round-A refit requires exactly three distinct seeds")
    paths = {name: Path(getattr(args, name)) for name in ("train_scores", "val_scores", "train_csv", "val_csv", "active_csv", "selected_config_json", "cv_summary_json")}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    selected = json.loads(paths["selected_config_json"].read_text(encoding="utf-8"))
    candidate_mapping = _selected_candidate_mapping(selected)
    candidate = Stage2ConceptCandidate(**candidate_mapping)
    cv_summary = json.loads(paths["cv_summary_json"].read_text(encoding="utf-8"))
    cv_candidate_mapping = _selected_candidate_mapping(cv_summary)
    if candidate_mapping != cv_candidate_mapping:
        raise ValueError("selected configuration must exactly match the CV-selected candidate")
    cv_hashes = cv_summary.get("input_hashes")
    expected_cv_hashes = {name: hashes[name] for name in ("train_scores", "val_scores", "train_csv", "val_csv", "active_csv")}
    if cv_hashes is not None and {name: cv_hashes.get(name) for name in expected_cv_hashes} != expected_cv_hashes:
        raise ValueError("refit train/validation inputs do not match CV input hashes")
    best_epochs = _selected_best_epochs(cv_summary, candidate.run_id)
    epochs = derive_refit_epochs(best_epochs)

    train_artifact, val_artifact = load_stage2_score_artifact(paths["train_scores"]), load_stage2_score_artifact(paths["val_scores"])
    validate_stage2_splits(train_artifact, val_artifact)
    propositions = load_stage2_propositions(paths["active_csv"], train_artifact.proposition_ids)
    train_targets = load_stage2_disease_targets(paths["train_csv"], train_artifact.case_ids)
    val_targets = load_stage2_disease_targets(paths["val_csv"], val_artifact.case_ids)
    if train_targets.disease_ids != val_targets.disease_ids:
        raise ValueError("train and validation disease label columns/order must match")
    train_features = build_stage2_concept_features(train_artifact, propositions, candidate.routing_mode)
    val_features = build_stage2_concept_features(val_artifact, propositions, candidate.routing_mode)
    if train_features.feature_ids != val_features.feature_ids:
        raise ValueError("routed train and validation feature IDs/order differ")
    pooled, _ = pool_stage2_datasets(
        Stage2Dataset(train_features.case_ids, train_features.support_scores, train_targets.targets),
        Stage2Dataset(val_features.case_ids, val_features.support_scores, val_targets.targets),
    )
    pooled_features = train_features.__class__(pooled.case_ids, pooled.support_scores, train_features.features, candidate.routing_mode)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    heads = []
    seed_histories: dict[str, Any] = {}
    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        head, history = refit_fixed_epochs(pooled, pooled_features, train_targets.disease_ids, candidate, epochs=epochs, seed=seed, device=args.device, output_dir=seed_dir)
        head.save(seed_dir / "effective_concept_head.npz")
        heads.append(head)
        seed_histories[str(seed)] = history
    averaged = average_frozen_heads(heads)
    averaged_path = output / "averaged_concept_head.npz"
    averaged.save(averaged_path)
    _write_rows(output / "stability_audit.csv", stability_rows(heads, averaged, pooled.support_scores))
    manifest = {"candidate": candidate_mapping, "input_hashes": hashes, "fold_best_epochs": best_epochs, "median_best_epoch": float(np.median(best_epochs)),
                "refit_epochs": epochs, "seeds": list(args.seeds), "pooled_case_ids": pooled.case_ids.tolist(), "averaged_head_hash": sha256_file(averaged_path), "seed_histories": seed_histories}
    _write_json(output / "refit_manifest.json", manifest)
    after = {name: sha256_file(path) for name, path in paths.items()}
    if after != hashes:
        raise RuntimeError("refit input artifact hashes changed during execution")
    print(json.dumps({"refit_epochs": epochs, "averaged_head": str(averaged_path)}, sort_keys=True))


def _selected_best_epochs(summary: dict[str, Any], run_id: str) -> list[int]:
    candidates = summary.get("candidate_summaries", [])
    matched = [item for item in candidates if item.get("candidate", {}).get("run_id") == run_id]
    if len(matched) != 1:
        raise ValueError("CV summary does not contain exactly one selected candidate")
    epochs = [int(row["best_epoch"]) for row in matched[0].get("fold_metrics", [])]
    if not epochs:
        raise ValueError("selected candidate has no CV fold best epochs")
    if len(epochs) != int(summary.get("folds", len(epochs))):
        raise ValueError("selected candidate CV fold epoch count is incomplete")
    return epochs


def _selected_candidate_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection", payload)
    selected = selection.get("selected_candidate", selection)
    candidate = selected.get("candidate", selected)
    if not isinstance(candidate, dict):
        raise ValueError("selected configuration must contain a candidate mapping")
    return candidate


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda item: item.item() if isinstance(item, np.generic) else item), encoding="utf-8")


if __name__ == "__main__":
    main()

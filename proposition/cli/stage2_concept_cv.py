"""Run leakage-safe train+validation cross-validation for Round-A concept heads."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from proposition.data.stage2_concept_features import build_stage2_concept_features
from proposition.data.stage2_scores import Stage2Dataset, load_stage2_disease_targets, load_stage2_propositions, load_stage2_score_artifact, validate_stage2_splits
from proposition.data.weak_supervision import sha256_file
from proposition.engine.stage2_concept_cv import (
    aggregate_cv_metrics,
    candidate_to_dict,
    load_stage2_cv_candidates,
    make_stratified_fold_assignments,
    pool_stage2_datasets,
    select_stage2_cv_configuration,
)
from proposition.engine.stage2_concept_trainer import Stage2ConceptTrainer, Stage2ConceptTrainerConfig
from proposition.models.stage2_concept_head import Stage2ConceptHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic train+validation CV for scalar-only Stage-2 concept heads.")
    parser.add_argument("--train_scores", required=True)
    parser.add_argument("--val_scores", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--active_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_config_json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: Path(getattr(args, name)) for name in ("train_scores", "val_scores", "train_csv", "val_csv", "active_csv", "candidate_config_json")}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    train_artifact, val_artifact = load_stage2_score_artifact(paths["train_scores"]), load_stage2_score_artifact(paths["val_scores"])
    validate_stage2_splits(train_artifact, val_artifact)
    propositions = load_stage2_propositions(paths["active_csv"], train_artifact.proposition_ids)
    train_targets = load_stage2_disease_targets(paths["train_csv"], train_artifact.case_ids)
    val_targets = load_stage2_disease_targets(paths["val_csv"], val_artifact.case_ids)
    if train_targets.disease_ids != val_targets.disease_ids:
        raise ValueError("train and validation disease label columns/order must match")
    disease_ids = train_targets.disease_ids
    candidates = load_stage2_cv_candidates(paths["candidate_config_json"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Fold membership depends only on labels and original split membership, never
    # on candidate/routing or any held-out test inputs.
    base_train = Stage2Dataset(train_artifact.case_ids, np.zeros((len(train_artifact.case_ids), 1), dtype=np.float32), train_targets.targets)
    base_val = Stage2Dataset(val_artifact.case_ids, np.zeros((len(val_artifact.case_ids), 1), dtype=np.float32), val_targets.targets)
    pooled_labels, original_splits = pool_stage2_datasets(base_train, base_val)
    assignments = make_stratified_fold_assignments(pooled_labels.case_ids, pooled_labels.targets, disease_ids, n_splits=args.folds, seed=args.seed, original_splits=original_splits)
    _write_rows(output / "cv_fold_assignments.csv", assignments.rows(disease_ids))
    assignment_hash = sha256_file(output / "cv_fold_assignments.csv")

    candidate_summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        train_features = build_stage2_concept_features(train_artifact, propositions, candidate.routing_mode)
        val_features = build_stage2_concept_features(val_artifact, propositions, candidate.routing_mode)
        if train_features.feature_ids != val_features.feature_ids:
            raise ValueError("routed train and validation feature IDs/order differ")
        pooled, _ = pool_stage2_datasets(
            Stage2Dataset(train_features.case_ids, train_features.support_scores, train_targets.targets),
            Stage2Dataset(val_features.case_ids, val_features.support_scores, val_targets.targets),
        )
        feature_disease_indices = _feature_disease_indices(train_features, disease_ids)
        feature_polarities = torch.tensor([item.polarity for item in train_features.features])
        fold_metrics: list[dict[str, Any]] = []
        oof_logits = np.empty((len(pooled.case_ids), len(disease_ids)), dtype=np.float32)
        for fold in range(args.folds):
            validation_mask = assignments.folds == fold
            _seed(args.seed + fold)
            model = Stage2ConceptHead(
                feature_disease_index=feature_disease_indices,
                feature_polarity=feature_polarities,
                feature_mask=torch.ones(len(train_features.features), dtype=torch.bool),
                num_diseases=len(disease_ids), head_mode=candidate.head_mode,
            )
            fold_output = output / "folds" / candidate.run_id / f"fold_{fold}"
            trainer = Stage2ConceptTrainer(
                model=model, disease_ids=disease_ids, proposition_ids=train_features.feature_ids,
                routing_mode=candidate.routing_mode, input_hashes=hashes, score_source=[item.source for item in train_features.features],
                config=Stage2ConceptTrainerConfig(device=args.device, learning_rate=candidate.learning_rate, regularization=candidate.regularization,
                    lambda_reg=candidate.lambda_reg, max_epochs=candidate.max_epochs, early_stopping_patience=candidate.patience,
                    batch_size=candidate.batch_size, base_seed=args.seed + fold), output_dir=fold_output,
            )
            trainer.fit(_subset(pooled, ~validation_mask), _subset(pooled, validation_mask))
            result = trainer.export_best_split(_subset(pooled, validation_mask), "val")
            oof_logits[validation_mask] = result["outputs"]["logits"]
            row = {"candidate_id": candidate.run_id, "fold": fold, "best_epoch": trainer.best_epoch, **_metric_row(result["metrics"])}
            fold_metrics.append(row)
            _write_json(fold_output / "fold_contract.json", {
                "train_case_ids": pooled.case_ids[~validation_mask].tolist(), "validation_case_ids": pooled.case_ids[validation_mask].tolist(),
                "feature_ids": list(train_features.feature_ids), "candidate": candidate_to_dict(candidate),
            })
        np.savez_compressed(output / f"oof_predictions_{candidate.run_id}.npz", case_id=pooled.case_ids, disease_id=np.asarray(disease_ids), fold=assignments.folds,
                            targets=assignments.target_indices, logits=oof_logits, probabilities=_softmax(oof_logits))
        _write_rows(output / f"cv_metrics_{candidate.run_id}.csv", fold_metrics)
        candidate_summaries.append({"candidate": candidate_to_dict(candidate), "cv_metrics": aggregate_cv_metrics(fold_metrics), "fold_metrics": fold_metrics})

    selection = select_stage2_cv_configuration(candidate_summaries)
    selected_run_id = selection["selected_candidate"]["candidate"]["run_id"]
    selected_oof = output / f"oof_predictions_{selected_run_id}.npz"
    canonical_oof = output / "oof_predictions.npz"
    canonical_oof.write_bytes(selected_oof.read_bytes())
    selected_config = {"selected_candidate": selection["selected_candidate"], "fold_assignment_hash": assignment_hash, "input_hashes": hashes,
                       "disease_ids": list(disease_ids), "oof_predictions_hash": sha256_file(canonical_oof)}
    _write_json(output / "selected_config.json", selected_config)
    manifest = {"seed": args.seed, "folds": args.folds, "fold_assignment_hash": assignment_hash, "input_hashes": hashes,
                "disease_ids": list(disease_ids), "candidate_summaries": candidate_summaries, "selection": selection,
                "selected_config_path": "selected_config.json", "oof_predictions_path": "oof_predictions.npz", "oof_predictions_hash": sha256_file(canonical_oof)}
    _write_json(output / "cv_summary.json", manifest)
    after = {name: sha256_file(path) for name, path in paths.items()}
    if after != hashes:
        raise RuntimeError("CV input artifact hashes changed during execution")
    print(json.dumps(selection, sort_keys=True))


def _feature_disease_indices(features: Any, disease_ids: tuple[str, ...]) -> torch.Tensor:
    lookup = {identifier: index for index, identifier in enumerate(disease_ids)}
    return torch.tensor([lookup[feature.disease_id] for feature in features.features])


def _subset(dataset: Stage2Dataset, mask: np.ndarray) -> Stage2Dataset:
    return Stage2Dataset(dataset.case_ids[mask], dataset.support_scores[mask], dataset.targets[mask])


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"per_disease", "probabilities", "confusion_matrix"}}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda item: item.item() if isinstance(item, np.generic) else item), encoding="utf-8")


if __name__ == "__main__":
    main()

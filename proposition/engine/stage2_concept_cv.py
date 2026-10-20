"""Leakage-safe deterministic cross-validation utilities for Stage-2 concept heads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from proposition.data.stage2_scores import Stage2Dataset
from proposition.engine.stage2_concept_metrics import class_indices


@dataclass(frozen=True)
class Stage2ConceptCandidate:
    run_id: str
    head_mode: str
    routing_mode: str
    regularization: str
    lambda_reg: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int


@dataclass(frozen=True)
class FoldAssignments:
    case_ids: np.ndarray
    target_indices: np.ndarray
    folds: np.ndarray
    original_splits: np.ndarray

    def rows(self, disease_ids: Sequence[str]) -> list[dict[str, Any]]:
        return [
            {
                "case_id": str(case_id),
                "disease_id": str(disease_ids[int(target)]),
                "fold": int(fold),
                "original_split": str(original_split),
            }
            for case_id, target, fold, original_split in zip(self.case_ids, self.target_indices, self.folds, self.original_splits)
        ]


def load_stage2_cv_candidates(path: str | Path) -> tuple[Stage2ConceptCandidate, ...]:
    """Load an explicit allow-listed CV candidate configuration JSON."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read candidate configuration {source}: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("candidate configuration must be a non-empty JSON list")
    candidates = tuple(_candidate(item) for item in payload)
    identifiers = [candidate.run_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate run_id values must be unique")
    return candidates


def make_stratified_fold_assignments(
    case_ids: Sequence[str],
    targets: Any,
    disease_ids: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 0,
    original_splits: Sequence[str] | None = None,
) -> FoldAssignments:
    """Assign every pooled case to one deterministic validation fold."""
    identifiers = np.asarray(case_ids, dtype=str)
    target_indices = class_indices(targets, len(disease_ids))
    if n_splits < 2:
        raise ValueError("folds must be at least two")
    if len(identifiers) != len(target_indices) or len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("case IDs must be unique and align with targets")
    counts = np.bincount(target_indices, minlength=len(disease_ids))
    insufficient = [str(disease_ids[index]) for index, count in enumerate(counts) if count < n_splits]
    if insufficient:
        raise ValueError(f"each disease needs at least {n_splits} pooled cases; insufficient={insufficient!r}")
    origin = np.asarray(original_splits if original_splits is not None else ("pooled",) * len(identifiers), dtype=str)
    if origin.shape != identifiers.shape:
        raise ValueError("original_splits must align with case_ids")
    folds = np.full(len(identifiers), -1, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(identifiers)), target_indices)):
        folds[validation] = fold
    if np.any(folds < 0):
        raise RuntimeError("stratified split omitted pooled cases")
    return FoldAssignments(identifiers, target_indices, folds, origin)


def pool_stage2_datasets(
    train: Stage2Dataset,
    validation: Stage2Dataset,
) -> tuple[Stage2Dataset, np.ndarray]:
    """Pool validated train/validation datasets and retain their original split IDs."""
    if train.support_scores.shape[1] != validation.support_scores.shape[1]:
        raise ValueError("train and validation feature widths differ")
    overlap = set(np.asarray(train.case_ids, dtype=str)).intersection(np.asarray(validation.case_ids, dtype=str))
    if overlap:
        raise ValueError(f"train and validation case IDs overlap: {sorted(overlap)[:5]!r}")
    pooled = Stage2Dataset(
        case_ids=np.concatenate((np.asarray(train.case_ids, dtype=str), np.asarray(validation.case_ids, dtype=str))),
        support_scores=np.concatenate((train.support_scores, validation.support_scores), axis=0),
        targets=np.concatenate((train.targets, validation.targets), axis=0),
    )
    splits = np.asarray(("train",) * len(train.case_ids) + ("val",) * len(validation.case_ids), dtype=str)
    return pooled, splits


def select_stage2_cv_configuration(
    summaries: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Select strictly from CV summaries under the Round-A practical-tie policy."""
    if tie_tolerance < 0 or not summaries:
        raise ValueError("non-empty CV summaries and non-negative tie_tolerance are required")
    candidates = [dict(item) for item in summaries]
    for candidate in candidates:
        _reject_test_fields(candidate)
        metrics = candidate.get("cv_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("each CV candidate needs a cv_metrics mapping")
        for key in ("mean_top_1_accuracy", "mean_cross_entropy"):
            if key not in metrics or not np.isfinite(float(metrics[key])):
                raise ValueError(f"each CV candidate needs finite {key}")
        _candidate(candidate.get("candidate", candidate))
    best_top_1 = max(float(item["cv_metrics"]["mean_top_1_accuracy"]) for item in candidates)
    tied = [item for item in candidates if best_top_1 - float(item["cv_metrics"]["mean_top_1_accuracy"]) <= tie_tolerance]

    def preference(item: Mapping[str, Any]) -> tuple[int, int, float, float, str]:
        candidate = _candidate(item.get("candidate", item))
        routing = 1 if candidate.routing_mode == "full_fused" else 0
        local = 1 if candidate.head_mode == "local" else 0
        return (routing, local, candidate.lambda_reg, -float(item["cv_metrics"]["mean_cross_entropy"]), candidate.run_id)

    selected = max(tied, key=preference)
    return {
        "best_mean_cv_top_1": best_top_1,
        "practical_tie_tolerance": tie_tolerance,
        "tied_candidates": [_json_ready(item) for item in tied],
        "selected_candidate": _json_ready(selected),
        "reason": "practical tie: prefer full_fused, then local, then stronger lambda_reg, then lower mean CV cross-entropy",
    }


def aggregate_cv_metrics(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Compute mean/std/min/max for scalar fold metrics."""
    if not fold_metrics:
        raise ValueError("at least one fold metric row is required")
    names = ("top_1_accuracy", "top_3_accuracy", "macro_f1", "balanced_accuracy", "macro_roc_auc", "mrr", "cross_entropy")
    result: dict[str, float | None] = {}
    for name in names:
        values = np.asarray([row.get(name, np.nan) for row in fold_metrics], dtype=float)
        finite = values[np.isfinite(values)]
        for operation, value in (("mean", finite.mean() if len(finite) else np.nan), ("std", finite.std(ddof=0) if len(finite) else np.nan), ("min", finite.min() if len(finite) else np.nan), ("max", finite.max() if len(finite) else np.nan)):
            result[f"{operation}_{name}"] = float(value) if np.isfinite(value) else None
    return result


def candidate_to_dict(candidate: Stage2ConceptCandidate) -> dict[str, Any]:
    return asdict(candidate)


def _candidate(value: Any) -> Stage2ConceptCandidate:
    if not isinstance(value, Mapping):
        raise ValueError("each candidate must be an object")
    _reject_test_fields(value)
    allowed = {"run_id", "head_mode", "routing_mode", "regularization", "lambda_reg", "learning_rate", "batch_size", "max_epochs", "patience"}
    unknown = set(value).difference(allowed)
    missing = allowed.difference(value)
    if unknown or missing:
        raise ValueError(f"candidate fields mismatch; missing={sorted(missing)!r}, unknown={sorted(unknown)!r}")
    candidate = Stage2ConceptCandidate(
        run_id=str(value["run_id"]), head_mode=str(value["head_mode"]), routing_mode=str(value["routing_mode"]),
        regularization=str(value["regularization"]), lambda_reg=float(value["lambda_reg"]), learning_rate=float(value["learning_rate"]),
        batch_size=int(value["batch_size"]), max_epochs=int(value["max_epochs"]), patience=int(value["patience"]),
    )
    if not candidate.run_id or candidate.head_mode not in {"local", "global"} or candidate.routing_mode not in {"image_morphology", "full_fused", "full_image_morphology_text_other", "full_dual_morphology"}:
        raise ValueError("candidate run_id, head_mode, or routing_mode is invalid")
    if candidate.regularization not in {"none", "l1", "l2"} or candidate.lambda_reg < 0 or candidate.learning_rate <= 0 or min(candidate.batch_size, candidate.max_epochs, candidate.patience) < 1:
        raise ValueError("candidate training parameters are invalid")
    if candidate.head_mode == "local" and candidate.regularization == "l1":
        raise ValueError("S2-L supports regularization=none or l2; use S2-G for L1 experiments")
    return candidate


def _reject_test_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if "test" in str(key).lower():
                raise ValueError("test fields are forbidden in CV configuration and selection")
            _reject_test_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_test_fields(nested)


def _json_ready(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=lambda item: item.item() if isinstance(item, np.generic) else item))

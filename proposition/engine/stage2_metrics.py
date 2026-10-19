"""Metrics for multi-label Stage-2 diagnosis prediction."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def multilabel_diagnosis_metrics(
    probabilities: Any,
    targets: Any,
    disease_ids: Sequence[str],
    *,
    threshold: float = 0.5,
    top_k: Iterable[int] = (1, 3),
) -> dict[str, Any]:
    """Calculate deterministic aggregate and per-disease multi-label metrics."""
    scores = np.asarray(probabilities, dtype=float)
    truth = np.asarray(targets)
    if scores.ndim != 2 or truth.shape != scores.shape or scores.shape[1] != len(disease_ids):
        raise ValueError("probabilities and targets must be matching [case, disease] arrays")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("targets must be binary")
    if len(set(disease_ids)) != len(disease_ids):
        raise ValueError("disease_ids must be unique")
    truth = truth.astype(bool)
    predicted = scores >= threshold
    rows: list[dict[str, Any]] = []
    aucs: list[float] = []
    for index, disease_id in enumerate(disease_ids):
        y, p = truth[:, index], predicted[:, index]
        tp, fp, tn, fn = int(np.sum(y & p)), int(np.sum(~y & p)), int(np.sum(~y & ~p)), int(np.sum(y & ~p))
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        specificity = _ratio(tn, tn + fp)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        auc = _roc_auc(y, scores[:, index])
        if auc is not None:
            aucs.append(auc)
        rows.append({
            "disease_id": str(disease_id), "support": int(y.sum()), "prevalence": float(y.mean()),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision,
            "recall": recall, "sensitivity": recall, "specificity": specificity, "f1": f1,
            "balanced_accuracy": (recall + specificity) / 2.0,
            "roc_auc": auc,
            "mean_probability_positive": float(scores[y, index].mean()) if y.any() else None,
            "mean_probability_negative": float(scores[~y, index].mean()) if (~y).any() else None,
        })
    top = _top_k_metrics(scores, truth, tuple(top_k))
    flattened_auc = _roc_auc(truth.reshape(-1), scores.reshape(-1))
    all_tp, all_fp = int(np.sum(truth & predicted)), int(np.sum(~truth & predicted))
    all_fn = int(np.sum(truth & ~predicted))
    aggregate = {
        **top,
        "macro_f1": _mean(row["f1"] for row in rows),
        "micro_f1": _ratio(2 * all_tp, 2 * all_tp + all_fp + all_fn),
        "macro_precision": _mean(row["precision"] for row in rows),
        "macro_recall": _mean(row["recall"] for row in rows),
        "balanced_accuracy": _mean(row["balanced_accuracy"] for row in rows),
        "macro_roc_auc": float(np.mean(aucs)) if aucs else None,
        "micro_roc_auc": flattened_auc,
        "n_auc_eligible": len(aucs),
        "mrr": _mrr(scores, truth),
    }
    return {**aggregate, "per_disease": rows}


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(np.mean(materialized)) if materialized else 0.0


def _roc_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    positives, negatives = int(target.sum()), int((~target).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    return float((ranks[target].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _top_k_metrics(scores: np.ndarray, truth: np.ndarray, top_k: tuple[int, ...]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    valid = truth.any(axis=1)
    for requested in top_k:
        if requested < 1:
            raise ValueError("top_k values must be positive")
        k = min(requested, scores.shape[1])
        top_indices = np.argsort(-scores, axis=1, kind="mergesort")[:, :k]
        hits = [truth[row, indices].any() for row, indices in enumerate(top_indices) if valid[row]]
        result[f"top_{requested}_accuracy"] = float(np.mean(hits)) if hits else None
        result[f"top_{requested}_hit_rate"] = result[f"top_{requested}_accuracy"]
    return result


def _mrr(scores: np.ndarray, truth: np.ndarray) -> float | None:
    reciprocals: list[float] = []
    for row, targets in zip(scores, truth):
        if not targets.any():
            continue
        ranking = np.argsort(-row, kind="mergesort")
        rank = int(np.flatnonzero(targets[ranking])[0]) + 1
        reciprocals.append(1.0 / rank)
    return float(np.mean(reciprocals)) if reciprocals else None
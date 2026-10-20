"""Single-label softmax metrics and validation-only selection for Stage 2."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def class_indices(targets: Any, num_diseases: int) -> np.ndarray:
    values = np.asarray(targets)
    if values.ndim == 1:
        indices = values.astype(np.int64, copy=False)
        if not np.all((indices >= 0) & (indices < num_diseases)):
            raise ValueError("class indices are out of range")
        return indices
    if values.ndim != 2 or values.shape[1] != num_diseases or not np.all(np.isin(values, (0, 1))):
        raise ValueError("targets must be class indices or binary one-hot [case, disease] values")
    if not np.all(values.sum(axis=1) == 1):
        raise ValueError("single-label Stage-2 targets must have exactly one positive disease per case")
    return values.argmax(axis=1).astype(np.int64)


def single_label_diagnosis_metrics(logits: Any, targets: Any, disease_ids: Sequence[str]) -> dict[str, Any]:
    """Calculate metrics exclusively from multinomial-softmax diagnosis outputs."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(disease_ids) or not np.all(np.isfinite(scores)):
        raise ValueError("logits must be finite [case, disease] values")
    if len(set(disease_ids)) != len(disease_ids):
        raise ValueError("disease_ids must be unique")
    truth = class_indices(targets, len(disease_ids))
    probabilities = _softmax(scores)
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((len(disease_ids), len(disease_ids)), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    rows: list[dict[str, Any]] = []
    aucs: list[float] = []
    for index, disease_id in enumerate(disease_ids):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        tn = int(confusion.sum() - tp - fp - fn)
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        auc = _roc_auc(truth == index, probabilities[:, index])
        if auc is not None:
            aucs.append(auc)
        rows.append({"disease_id": str(disease_id), "support": int((truth == index).sum()), "precision": precision,
                     "recall": recall, "f1": f1, "roc_auc": auc, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    ranking = np.argsort(-probabilities, axis=1, kind="mergesort")
    top1, top3 = (ranking[:, :1] == truth[:, None]).any(axis=1), (ranking[:, :min(3, len(disease_ids))] == truth[:, None]).any(axis=1)
    ranks = np.argmax(ranking == truth[:, None], axis=1) + 1
    ce = float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-15, 1.0)).mean())
    return {"top_1_accuracy": float(top1.mean()), "top_3_accuracy": float(top3.mean()),
            "macro_f1": float(np.mean([row["f1"] for row in rows])), "balanced_accuracy": float(np.mean([row["recall"] for row in rows])),
            "macro_roc_auc": float(np.mean(aucs)) if aucs else None, "mrr": float(np.mean(1.0 / ranks)), "cross_entropy": ce,
            "per_disease": rows, "confusion_matrix": confusion, "probabilities": probabilities}


def bootstrap_topk_confidence_intervals(logits: Any, targets: Any, *, seed: int, n_bootstrap: int = 2000) -> dict[str, dict[str, float]]:
    """Seeded post-selection percentile bootstrap CIs for Top-1 and Top-3."""
    scores = np.asarray(logits, dtype=float)
    truth = class_indices(targets, scores.shape[1])
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    ranking = np.argsort(-scores, axis=1, kind="mergesort")
    hits = {"top_1_accuracy": (ranking[:, :1] == truth[:, None]).any(axis=1),
            "top_3_accuracy": (ranking[:, :min(3, scores.shape[1])] == truth[:, None]).any(axis=1)}
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(truth), size=(n_bootstrap, len(truth)))
    return {name: {"estimate": float(values.mean()), "lower_95": float(np.quantile(values[samples].mean(axis=1), .025)),
                   "upper_95": float(np.quantile(values[samples].mean(axis=1), .975))} for name, values in hits.items()}


def select_stage2_configuration(runs: Iterable[Mapping[str, Any]], *, tie_tolerance: float = 0.02) -> dict[str, Any]:
    """Select only from validation fields using the K34 practical-tie policy."""
    candidates = [dict(item) for item in runs]
    if not candidates:
        raise ValueError("at least one validation candidate is required")
    for item in candidates:
        if "test" in item or "test_metrics" in item:
            raise ValueError("test metrics must not be provided to validation-only selection")
        metrics = item.get("validation_metrics", item.get("val_metrics", {}))
        if "top_1_accuracy" not in metrics or "cross_entropy" not in metrics:
            raise ValueError("each candidate needs validation top_1_accuracy and cross_entropy")
        item["_metrics"] = metrics
    best_top1 = max(float(item["_metrics"]["top_1_accuracy"]) for item in candidates)
    tied = [item for item in candidates if best_top1 - float(item["_metrics"]["top_1_accuracy"]) <= tie_tolerance]
    def preference(item: Mapping[str, Any]) -> tuple[float, float, float, str]:
        mode = item.get("head_mode")
        if mode not in {"local", "global"}:
            raise ValueError("candidate head_mode must be local or global")
        reg = str(item.get("regularization", "none"))
        reg_rank = {"none": 0, "l2": 1, "l1": 2}.get(reg)
        if reg_rank is None:
            raise ValueError("unknown regularization")
        return (1.0 if mode == "local" else 0.0, float(reg_rank), float(item.get("lambda_reg", 0.0)), -float(item["_metrics"]["cross_entropy"]))
    selected = sorted(tied, key=lambda item: (*preference(item), str(item.get("run_id", ""))), reverse=True)[0]
    reason = "practical tie: preferred local head, then stronger regularization, then lower validation cross-entropy"
    return {"selected_run": _without_internal(selected), "reason": reason, "best_validation_top_1": best_top1,
            "tied_candidates": [_without_internal(item) for item in tied], "validation_metric_values": [_without_internal(item) for item in candidates]}


def _without_internal(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "_metrics"}

def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)

def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0

def _roc_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    positives, negatives = int(target.sum()), int((~target).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores, start = score[order], 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    return float((ranks[target].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))
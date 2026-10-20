"""Stage-1 proposition evidence metrics, independent of disease labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_VALID = frozenset((0, 1))


def _array(value: Any) -> np.ndarray:
    """Convert NumPy/Torch-like values without requiring torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate(scores: Any, labels: Any, proposition_ids: Sequence[str] | None) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    score_array, label_array = _array(scores).astype(float), _array(labels)
    if score_array.ndim != 2 or label_array.shape != score_array.shape:
        raise ValueError("scores and labels must be aligned [case, proposition] matrices")
    ids = tuple(map(str, proposition_ids or tuple(range(score_array.shape[1]))))
    if len(ids) != score_array.shape[1]:
        raise ValueError("proposition_ids must contain one id per score column")
    return score_array, label_array, ids


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    n_positive, n_negative = int(positives.sum()), int((~positives).sum())
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order] == 1
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels].sum() / sorted_labels.sum())


def modality_metrics(
    scores: Any,
    labels: Any,
    *,
    proposition_ids: Sequence[str] | None = None,
    valid_propositions: Any | None = None,
    threshold: float = 0.5,
    min_present: int = 1,
    min_absent: int = 1,
    target_semantics: str = "ternary_masked",
) -> dict[str, Any]:
    """Score valid weak-label pairs, excluding unknown and single-class pairs.

    Inputs are proposition evidence and weak proposition labels only.  Returned
    values are JSON-serializable, including a row per proposition and macro
    selection metric (macro ROC-AUC across eligible propositions).
    """
    if min_present < 1 or min_absent < 1:
        raise ValueError("min_present and min_absent must both be at least one")
    if target_semantics not in {"ternary_masked", "caption_support_binary"}:
        raise ValueError("unsupported target_semantics")
    score_array, label_array, ids = _validate(scores, labels, proposition_ids)
    if target_semantics == "caption_support_binary":
        label_array = (label_array == 1).astype(np.int8)
    eligible_columns = np.ones(score_array.shape[1], dtype=bool) if valid_propositions is None else _array(valid_propositions).astype(bool)
    if eligible_columns.shape != (score_array.shape[1],):
        raise ValueError("valid_propositions must have one value per proposition")
    rows: list[dict[str, Any]] = []
    for column, proposition_id in enumerate(ids):
        valid = eligible_columns[column] & np.isfinite(score_array[:, column]) & np.isin(label_array[:, column], tuple(_VALID))
        y, s = label_array[valid, column].astype(int), score_array[valid, column]
        n_positive, n_negative = int((y == 1).sum()), int((y == 0).sum())
        row: dict[str, Any] = {
            "proposition_id": proposition_id,
            "n_valid": int(valid.sum()),
            "n_present": n_positive,
            "n_absent": n_negative,
            "eligible": bool(
                n_positive >= min_present
                and n_negative >= min_absent
            ),
        }
        if row["eligible"]:
            predicted = s >= threshold
            tp = int(np.count_nonzero(predicted & (y == 1)))
            fp = int(np.count_nonzero(predicted & (y == 0)))
            fn = int(np.count_nonzero(~predicted & (y == 1)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            row.update({"roc_auc": _rank_auc(y, s), "pr_auc": _average_precision(y, s), "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
        else:
            row.update({"roc_auc": None, "pr_auc": None, "precision": None, "recall": None, "f1": None})
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    macro = lambda key: float(np.mean([row[key] for row in eligible])) if eligible else None
    return {
        "threshold": float(threshold),
        "min_present": int(min_present),
        "min_absent": int(min_absent),
        "target_semantics": target_semantics,

        "n_eligible": len(eligible),

        "macro_roc_auc": macro("roc_auc"),
        "macro_pr_auc": macro("pr_auc"),
        "selection_macro": macro("roc_auc"),

        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),

        "rows": rows,
    }


def masked_binary_cross_entropy(
    logits: Any,
    labels: Any,
    *,
    valid_pairs: Any | None = None,
    target_semantics: str = "ternary_masked",
) -> float | None:
    """Mean BCE over finite, known weak-label pairs; returns None when empty."""
    logit_array, label_array, _ = _validate(logits, labels, None)
    if target_semantics not in {"ternary_masked", "caption_support_binary"}:
        raise ValueError("unsupported target_semantics")
    if target_semantics == "caption_support_binary":
        label_array = (label_array == 1).astype(np.int8)
        valid = np.isfinite(logit_array)
    else:
        valid = np.isfinite(logit_array) & np.isin(label_array, tuple(_VALID))
    if valid_pairs is not None:
        pair_mask = _array(valid_pairs).astype(bool)
        valid &= np.broadcast_to(pair_mask, valid.shape)
    if not valid.any():
        return None
    x, y = logit_array[valid], label_array[valid].astype(float)
    return float(np.mean(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))))


def evaluate_modalities(
    outputs: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    text_target_semantics: str = "ternary_masked",
    image_target_semantics: str = "ternary_masked",
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate text/image evidence mappings and optional masked validation loss."""
    result = {
        modality: modality_metrics(
            outputs[modality],
            labels[modality],
            target_semantics=text_target_semantics if modality == "text" else image_target_semantics,
            **kwargs,
        )
        for modality in ("text", "image")
        if modality in outputs and modality in labels
    }
    for modality, logit_key in (("text", "logits_txt"), ("image", "logits_img")):
        if logit_key in outputs and modality in labels:
            result[modality]["masked_val_loss"] = masked_binary_cross_entropy(
                outputs[logit_key],
                labels[modality],
                target_semantics=text_target_semantics if modality == "text" else image_target_semantics,
            )
    return result

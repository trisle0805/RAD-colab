"""Audits for interpretable Stage-2 learned concept heads."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def concept_weight_rows(weights: Any, disease_ids: Sequence[str], proposition_ids: Sequence[str], proposition_disease_ids: Sequence[str], *, nonzero_threshold: float) -> list[dict[str, Any]]:
    matrix = np.asarray(weights, dtype=float)
    if matrix.shape != (len(disease_ids), len(proposition_ids)) or nonzero_threshold < 0:
        raise ValueError("invalid weight matrix or nonzero_threshold")
    return [{"disease_id": disease, "proposition_id": proposition, "proposition_disease_id": proposition_disease,
             "weight": float(matrix[disease_index, proposition_index]), "nonzero": bool(abs(matrix[disease_index, proposition_index]) >= nonzero_threshold),
             "same_disease": bool(disease == proposition_disease)}
            for disease_index, disease in enumerate(disease_ids) for proposition_index, (proposition, proposition_disease) in enumerate(zip(proposition_ids, proposition_disease_ids))]


def disease_sparsity_rows(weights: Any, disease_ids: Sequence[str], proposition_disease_ids: Sequence[str], *, nonzero_threshold: float) -> list[dict[str, Any]]:
    matrix = np.asarray(weights, dtype=float)
    rows = []
    for index, disease in enumerate(disease_ids):
        nonzero = np.abs(matrix[index]) >= nonzero_threshold
        same = np.asarray([item == disease for item in proposition_disease_ids], dtype=bool)
        rows.append({"disease_id": disease, "total_nonzero_weights": int(nonzero.sum()), "same_disease_nonzero_weights": int((nonzero & same).sum()), "cross_disease_nonzero_weights": int((nonzero & ~same).sum())})
    return rows


def same_disease_contribution_fraction(input_scores: Any, weights: Any, disease_ids: Sequence[str], proposition_disease_ids: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    scores, matrix = np.asarray(input_scores, dtype=float), np.asarray(weights, dtype=float)
    if scores.ndim != 2 or matrix.shape != (len(disease_ids), scores.shape[1]):
        raise ValueError("input_scores and weights must align")
    rows: list[dict[str, Any]] = []
    for index, disease in enumerate(disease_ids):
        absolute = np.abs(scores * matrix[index][None, :]).sum(axis=0)
        denominator = float(absolute.sum())
        same = np.asarray([item == disease for item in proposition_disease_ids], dtype=bool)
        # With no magnitude to apportion, a local-only head satisfies the
        # same-disease constraint vacuously; this keeps its audit invariant 1.
        fraction = float(absolute[same].sum() / denominator) if denominator else 1.0
        rows.append({"disease_id": disease, "same_disease_contribution_fraction": fraction})
    values = np.asarray([item["same_disease_contribution_fraction"] for item in rows], dtype=float)
    return rows, {"macro_mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}
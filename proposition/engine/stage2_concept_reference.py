"""Sklearn reference gate for reproducing the historical scalar probe."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from proposition.engine.stage2_concept_metrics import class_indices, single_label_diagnosis_metrics


def run_reference_probe(train_scores: Any, train_targets: Any, val_scores: Any, val_targets: Any, test_scores: Any, test_targets: Any, disease_ids: Sequence[str], *, seed: int, selected_c: float = 0.1) -> dict[str, Any]:
    """Fit the exact standardized sklearn LogisticRegression replication contract."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("scikit-learn is required for the Stage-2 reference gate") from error
    if selected_c <= 0:
        raise ValueError("selected_c must be positive")
    train_x, val_x, test_x = (np.asarray(value, dtype=float) for value in (train_scores, val_scores, test_scores))
    if any(value.ndim != 2 or value.shape[1] != train_x.shape[1] or not np.all(np.isfinite(value)) for value in (train_x, val_x, test_x)):
        raise ValueError("reference scores must be aligned finite matrices")
    train_y = class_indices(train_targets, len(disease_ids))
    pipeline = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=selected_c, solver="lbfgs", max_iter=5000, random_state=seed))])
    pipeline.fit(train_x, train_y)
    result: dict[str, Any] = {"selected_C": selected_c}
    for split, scores, targets in (("val", val_x, val_targets), ("test", test_x, test_targets)):
        probabilities = pipeline.predict_proba(scores)
        logits = np.log(np.clip(probabilities, 1e-15, 1.0))
        metrics = single_label_diagnosis_metrics(logits, targets, disease_ids)
        result[f"{split}_metrics"] = metrics
        for key in ("top_1_accuracy", "top_3_accuracy"):
            result[f"{split}_{'top1' if key == 'top_1_accuracy' else 'top3'}"] = metrics[key]
    result["macro_f1"] = result["test_metrics"]["macro_f1"]
    result["macro_auc"] = result["test_metrics"]["macro_roc_auc"]
    result["mrr"] = result["test_metrics"]["mrr"]
    result["cross_entropy"] = result["test_metrics"]["cross_entropy"]
    return result


def enforce_reference_gate(result: dict[str, Any], *, expected_val_top1: float, expected_test_top1: float, tolerance: float = 0.01) -> None:
    """Fail loudly when the frozen B reference cannot reproduce historical Top-1."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    deviations = {"val": abs(float(result["val_top1"]) - expected_val_top1), "test": abs(float(result["test_top1"]) - expected_test_top1)}
    if any(value > tolerance for value in deviations.values()):
        raise RuntimeError(f"G-B-L2 reference replication gate failed: deviations={deviations!r}, tolerance={tolerance}")
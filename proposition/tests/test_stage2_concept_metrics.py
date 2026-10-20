from __future__ import annotations

import numpy as np
import pytest

from proposition.engine.stage2_concept_metrics import bootstrap_topk_confidence_intervals, select_stage2_configuration, single_label_diagnosis_metrics


def test_single_label_metrics_use_softmax_rankings_and_one_hot_contract():
    logits = np.asarray([[3., 1., 0.], [0., 3., 1.], [1., 0., 3.]])
    targets = np.eye(3, dtype=np.int8)
    metrics = single_label_diagnosis_metrics(logits, targets, ("a", "b", "c"))
    assert metrics["top_1_accuracy"] == pytest.approx(1.0)
    assert metrics["top_3_accuracy"] == pytest.approx(1.0)
    assert metrics["cross_entropy"] > 0
    assert metrics["probabilities"].sum(axis=1) == pytest.approx(np.ones(3))
    ci = bootstrap_topk_confidence_intervals(logits, targets, seed=7, n_bootstrap=20)
    assert ci["top_1_accuracy"]["estimate"] == pytest.approx(1.0)


def test_validation_selection_prefers_local_then_regularization_then_ce_and_rejects_test():
    runs = [
        {"run_id": "global", "head_mode": "global", "regularization": "l1", "lambda_reg": .1, "validation_metrics": {"top_1_accuracy": .80, "cross_entropy": .1}},
        {"run_id": "local-l2", "head_mode": "local", "regularization": "l2", "lambda_reg": .01, "validation_metrics": {"top_1_accuracy": .79, "cross_entropy": .3}},
        {"run_id": "local-none", "head_mode": "local", "regularization": "none", "lambda_reg": 0, "validation_metrics": {"top_1_accuracy": .80, "cross_entropy": .01}},
    ]
    selected = select_stage2_configuration(runs)
    assert selected["selected_run"]["run_id"] == "local-l2"
    with pytest.raises(ValueError, match="test"):
        select_stage2_configuration([{"head_mode": "local", "validation_metrics": {"top_1_accuracy": 1., "cross_entropy": 0.}, "test": {}}])
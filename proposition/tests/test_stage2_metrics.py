from __future__ import annotations

import numpy as np

from proposition.engine.stage2_metrics import multilabel_diagnosis_metrics


def test_multilabel_metrics_topk_auc_counts_and_mrr_are_deterministic():
    probabilities = np.asarray([[0.9, 0.8, 0.1], [0.2, 0.4, 0.7], [0.7, 0.1, 0.5]])
    targets = np.asarray([[1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
    metrics = multilabel_diagnosis_metrics(probabilities, targets, ("a", "b", "c"), top_k=(1, 3))
    assert metrics["top_1_accuracy"] == 1.0 and metrics["top_3_hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0 and metrics["n_auc_eligible"] == 3
    assert metrics["per_disease"][0]["tp"] == 2
    single = multilabel_diagnosis_metrics(np.asarray([[.1, .5], [.2, .6]]), np.asarray([[0, 1], [0, 1]]), ("a", "b"))
    assert single["macro_roc_auc"] is None and single["n_auc_eligible"] == 0
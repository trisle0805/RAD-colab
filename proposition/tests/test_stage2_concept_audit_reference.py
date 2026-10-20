from __future__ import annotations

import numpy as np
import pytest

from proposition.engine.stage2_concept_audit import disease_sparsity_rows, same_disease_contribution_fraction
from proposition.engine.stage2_concept_reference import enforce_reference_gate, run_reference_probe


def test_audit_counts_nonzero_weights_and_local_fraction_is_one():
    weights = np.asarray([[1., 0., 0.], [0., 0., -2.]])
    rows = disease_sparsity_rows(weights, ("a", "b"), ("a", "b", "b"), nonzero_threshold=1e-4)
    assert rows[0] == {"disease_id": "a", "total_nonzero_weights": 1, "same_disease_nonzero_weights": 1, "cross_disease_nonzero_weights": 0}
    fractions, summary = same_disease_contribution_fraction(np.asarray([[.5, .2, .7]]), weights, ("a", "b"), ("a", "b", "b"))
    assert [row["same_disease_contribution_fraction"] for row in fractions] == pytest.approx([1., 1.])
    assert summary["min"] == pytest.approx(1.)


def test_reference_helper_and_gate_on_synthetic_multiclass_data():
    sklearn = pytest.importorskip("sklearn")
    del sklearn
    train = np.asarray([[1., 0.], [.9, .1], [0., 1.], [.1, .9], [.8, .2], [.2, .8]])
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    result = run_reference_probe(train, labels, train, labels, train, labels, ("a", "b"), seed=1)
    assert result["val_top1"] == pytest.approx(1.)
    enforce_reference_gate(result, expected_val_top1=1., expected_test_top1=1.)
    with pytest.raises(RuntimeError, match="gate failed"):
        enforce_reference_gate(result, expected_val_top1=.5, expected_test_top1=.5)
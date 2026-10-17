import numpy as np

from proposition.engine.diagnostics import (
    proposition_correlation,
    query_ablation,
    query_swap,
    save_attention_artifact,
    within_patient_differentiation,
)
from proposition.engine.metrics import masked_binary_cross_entropy, modality_metrics


def test_metrics_exclude_unknown_and_single_class_pairs():
    result = modality_metrics(
        [[0.1, 0.9], [0.9, 0.8], [0.2, 0.7]],
        [[0, 1], [1, 1], [-1, 1]],
        proposition_ids=["p1", "p2"],
    )
    assert result["n_eligible"] == 1
    assert result["macro_roc_auc"] == 1.0
    assert result["rows"][1]["roc_auc"] is None
    assert masked_binary_cross_entropy([[0.0]], [[-1]]) is None


def test_diagnostics_and_query_helpers_are_deterministic(tmp_path):
    scores = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert within_patient_differentiation(scores)["mean_within_patient_std"] == 0.0
    assert proposition_correlation(scores)["near_identical_fraction"] == 1.0

    def model(batch, queries):
        return np.asarray(queries).sum(axis=-1) + batch

    queries = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert query_swap(model, 0.0, queries)["swap_indices"] == [1, 0]
    first = query_ablation(model, 0.0, queries, mode="random", seed=3)
    second = query_ablation(model, 0.0, queries, mode="random", seed=3)
    assert np.array_equal(first["ablated"], second["ablated"])
    path = save_attention_artifact(scores, tmp_path / "attention.npz", metadata={"z": "x", "a": 1})
    with np.load(path, allow_pickle=False) as saved:
        assert saved.files == ["attention", "metadata_a", "metadata_z"]

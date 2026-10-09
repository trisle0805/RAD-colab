import json

import numpy as np
import pytest

from proposition.visualization.segment_map_analysis import analyze


def test_identical_maps_have_unit_correlations(tmp_path) -> None:
    checklist = tmp_path / "checklist"
    checklist.mkdir()
    (checklist / "proposition_index.json").write_text(
        json.dumps([
            {"proposition_id": "p0", "disease_index": 0},
            {"proposition_id": "p1", "disease_index": 0},
        ]),
        encoding="utf-8",
    )
    # Four image tokens then one caption token; every proposition map is identical.
    attention = np.array(
        [
            [[0.4, 0.3, 0.2, 0.1, 0.0], [0.4, 0.3, 0.2, 0.1, 0.0]],
            [[0.4, 0.3, 0.2, 0.1, 0.0], [0.4, 0.3, 0.2, 0.1, 0.0]],
        ],
        dtype=np.float16,
    )
    evidence = checklist / "test_segment_evidence.npz"
    np.savez_compressed(
        evidence,
        full_attention=attention,
        full_attention_sample_indices=np.array([0, 1]),
        image_token_count=np.array([4, 4], dtype=np.int32),
        pred_proposition=np.array([[0.9], [0.8]], dtype=np.float32),
    )

    metrics = analyze(evidence)

    assert metrics["within_case_same_disease_corr"] == 1.0
    assert metrics["same_prop_across_cases_corr"] == 1.0
    assert metrics["image_max_over_mean"] == pytest.approx(1.6, rel=1e-3)
    assert metrics["image_mass_share"] == 1.0
    assert metrics["num_cases"] == 2
    assert metrics["num_props_used"] == 2
    assert (checklist / "segment_map_analysis.json").is_file()

import json

import numpy as np

from proposition.visualization.proposition_discrimination import (
    compute_proposition_discrimination_from_evidence,
    run,
)


def _evidence() -> dict[str, np.ndarray]:
    return {
        "support": np.asarray([
            [0.9, 0.8, 0.2, 0.7],
            [0.4, 0.6, 0.9, 0.1],
            [0.7, 0.3, 0.8, 0.2],
        ], dtype=np.float32),
        "compatibility": np.ones((3, 4), dtype=np.float32),
        "pred_proposition": np.asarray([
            [0.8, 0.2], [0.1, 0.9], [0.7, 0.3],
        ], dtype=np.float32),
    }


def test_discrimination_metrics_and_json_output(tmp_path) -> None:
    evidence = _evidence()
    diseases = np.asarray([0, 0, 1, 1])
    polarities = np.asarray([1, -1, 1, -1])
    result = compute_proposition_discrimination_from_evidence(evidence, diseases, polarities)

    assert result["num_cases"] == 3
    assert result["top1_pos_share_s_gt_0.5"] == 1.0
    assert result["top1_neg_share_s_gt_0.5"] == 1 / 3
    assert result["other_share_s_gt_0.5"] == 0.5
    assert result["top2_most_common_disease"] == {
        "disease_index": 1, "disease_id": None, "num_cases": 2,
    }

    evidence_path = tmp_path / "test_proposition_evidence.npz"
    np.savez_compressed(evidence_path, **evidence)
    (tmp_path / "proposition_index.json").write_text(json.dumps([
        {"disease_index": 0, "disease_id": "A", "polarity": 1},
        {"disease_index": 0, "disease_id": "A", "polarity": -1},
        {"disease_index": 1, "disease_id": "B", "polarity": 1},
        {"disease_index": 1, "disease_id": "B", "polarity": -1},
    ]), encoding="utf-8")

    output = run(evidence_path)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["top2_most_common_disease"] == {
        "disease_index": 1, "disease_id": "B", "num_cases": 2,
    }
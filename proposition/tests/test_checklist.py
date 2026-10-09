import json

import numpy as np
import torch

from proposition.data.knowledge_base import load_proposition_kb
from proposition.engine.checklist import ChecklistResult, attention_entropy, save_checklist_artifacts


def _kb(tmp_path):
    payload = [
        {
            "disease_id": "A",
            "sourceReference": "ref-a",
            "propositions": [
                {"proposition_id": "a1", "category": "morphology", "canonicalDescription": "red spot", "polarity": 1, "sourceExcerpt": "x"},
                {"proposition_id": "a2", "category": "symptom", "canonicalDescription": "no pain", "polarity": -1, "sourceExcerpt": "y"},
            ],
        },
        {
            "disease_id": "B",
            "sourceReference": "ref-b",
            "propositions": [
                {"proposition_id": "b1", "category": "morphology", "canonicalDescription": "blue spot", "polarity": 1, "sourceExcerpt": "z"},
            ],
        },
    ]
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_proposition_kb(path, ["A", "B"])


def _result():
    attention = torch.tensor([[
        [0.50, 0.20, 0.10, 0.10, 0.10],
        [0.10, 0.50, 0.10, 0.20, 0.10],
        [0.10, 0.10, 0.50, 0.10, 0.20],
    ]])
    return ChecklistResult(
        gt=torch.tensor([[1.0, 0.0]]),
        pred_proposition=torch.tensor([[0.9, 0.1]]),
        pred_label=None,
        support=torch.tensor([[0.8, 0.3, 0.4]]),
        compatibility=torch.tensor([[0.8, 0.7, 0.4]]),
        disease_scores=torch.tensor([[0.75, 0.4]]),
        attention_top_indices=torch.topk(attention, k=2, dim=-1).indices.to(torch.int32),
        attention_top_weights=torch.topk(attention, k=2, dim=-1).values,
        attention_entropy=attention_entropy(attention),
        full_attention_sample_indices=torch.tensor([0]),
        full_attention=attention.to(torch.float16),
        image_token_count=torch.tensor([2], dtype=torch.int32),
        caption_input_ids=torch.tensor([[101, 10, 11]]),
        caption_attention_mask=torch.tensor([[1, 1, 1]]),
    )


def test_attention_entropy_is_normalized() -> None:
    entropy = _result().attention_entropy
    assert entropy.shape == (1, 3)
    assert torch.all((entropy >= 0) & (entropy <= 1))


def test_checklist_artifact_contains_complete_ui_contract(tmp_path) -> None:
    paths = save_checklist_artifacts(
        _result(), tmp_path, _kb(tmp_path), image_paths=["case.png"], top_evidence=2
    )

    with np.load(paths["evidence"]) as artifact:
        assert artifact["full_attention"].shape == (1, 3, 5)
        assert artifact["full_attention"].dtype == np.float16
        assert artifact["full_attention_sample_indices"].tolist() == [0]
        assert artifact["attention_top_indices"].shape == (1, 3, 2)
        assert artifact["caption_input_ids"].shape == (1, 3)
        assert artifact["image_paths"].tolist() == ["case.png"]
    with open(paths["schema"], encoding="utf-8") as handle:
        schema = json.load(handle)
    assert schema["memory_layout"] == {"image": [0, 2], "caption": [2, 5]}
    assert schema["query_level"] == "proposition"

def test_segment_checklist_uses_disease_scores_and_proposition_attention(tmp_path) -> None:
    result = _result()
    result.support = torch.tensor([[0.8, 0.4]])
    result.compatibility = result.support.clone()
    paths = save_checklist_artifacts(
        result,
        tmp_path,
        _kb(tmp_path),
        image_paths=["case.png"],
        top_evidence=2,
        query_level="proposition_segments",
    )

    assert paths["evidence"].endswith("test_segment_evidence.npz")
    assert "proposition_index" in paths
    with np.load(paths["evidence"]) as artifact:
        assert artifact["support"].shape == (1, 2)
        assert artifact["full_attention"].shape == (1, 3, 5)
    with open(paths["schema"], encoding="utf-8") as handle:
        schema = json.load(handle)
    assert schema["query_level"] == "proposition_segments"
    assert schema["attention_axes"] == ["sample", "proposition", "memory_token"]
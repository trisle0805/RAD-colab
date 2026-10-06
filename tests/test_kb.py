import json
from pathlib import Path

import pytest
import torch

from dataset.kb import CATEGORY_ORDER, load_proposition_kb


RAD_ROOT = Path(__file__).resolve().parents[1]
FINAL_KB_PATH = RAD_ROOT / "kb_builder" / "outputs" / "final" / "kb_propositions.json"


def _write_kb(tmp_path: Path, payload: list[dict]) -> Path:
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _disease(disease_id: str, propositions: list[dict]) -> dict:
    return {
        "disease_id": disease_id,
        "sourceReference": f"source for {disease_id}",
        "propositions": propositions,
    }


def _proposition(
    proposition_id: str,
    text: str,
    *,
    polarity: int = 1,
    category: str = "morphology",
) -> dict:
    return {
        "proposition_id": proposition_id,
        "category": category,
        "canonicalDescription": text,
        "polarity": polarity,
        "sourceExcerpt": [f"excerpt for {proposition_id}"],
    }


def test_loader_builds_deterministic_indices_and_deduplicates_text(tmp_path: Path) -> None:
    path = _write_kb(
        tmp_path,
        [
            _disease(
                "alpha",
                [
                    _proposition("alpha_P001", "shared sign"),
                    _proposition("alpha_P002", "invasion", polarity=-1),
                ],
            ),
            _disease(
                "beta",
                [
                    _proposition("beta_P001", "shared sign"),
                    _proposition("beta_P002", "laboratory sign", category="lab"),
                ],
            ),
        ],
    )

    kb = load_proposition_kb(path, ["alpha", "beta"])

    assert kb.disease_ids == ("alpha", "beta")
    assert kb.num_diseases == 2
    assert kb.num_propositions == 4
    assert torch.equal(kb.prop_disease, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(kb.prop_polarity, torch.tensor([1, -1, 1, 1], dtype=torch.int8))
    assert torch.equal(kb.disease_prop_counts, torch.tensor([2, 2]))
    assert kb.negative_alignment_texts == ("absence of invasion",)

    decoded_queries = tuple(kb.unique_texts[index] for index in kb.query_text_index)
    decoded_alignments = tuple(kb.unique_texts[index] for index in kb.align_text_index)
    assert decoded_queries == (
        "shared sign",
        "invasion",
        "shared sign",
        "laboratory sign",
    )
    assert decoded_alignments == (
        "shared sign",
        "absence of invasion",
        "shared sign",
        "laboratory sign",
    )
    assert len(kb.unique_texts) == 4
    assert len(kb.unique_alignment_texts) == 3
    assert kb.disease_query_texts == (
        "shared sign; absence of invasion",
        "shared sign; laboratory sign",
    )


def test_loader_rejects_label_order_mismatch_with_indexed_diff(tmp_path: Path) -> None:
    path = _write_kb(
        tmp_path,
        [
            _disease("alpha", [_proposition("alpha_P001", "a")]),
            _disease("beta", [_proposition("beta_P001", "b")]),
        ],
    )

    with pytest.raises(ValueError, match=r"index 0: CSV='beta', KB='alpha'"):
        load_proposition_kb(path, ["beta", "alpha"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.append(payload[0].copy()), "duplicate disease_id"),
        (
            lambda payload: payload[0]["propositions"].append(
                payload[0]["propositions"][0].copy()
            ),
            "duplicate proposition_id",
        ),
        (lambda payload: payload[0].update(propositions=[]), "non-empty list"),
        (
            lambda payload: payload[0]["propositions"][0].update(polarity=0),
            "polarity must be integer -1 or 1",
        ),
        (
            lambda payload: payload[0]["propositions"][0].update(category="unknown"),
            "unsupported category",
        ),
    ],
)
def test_loader_rejects_invalid_schema(tmp_path: Path, mutation, message: str) -> None:
    payload = [_disease("alpha", [_proposition("alpha_P001", "a")])]
    mutation(payload)
    path = _write_kb(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_proposition_kb(path)


def test_loader_rejects_noncanonical_category_order(tmp_path: Path) -> None:
    path = _write_kb(
        tmp_path,
        [
            _disease(
                "alpha",
                [
                    _proposition("alpha_P001", "lab", category="lab"),
                    _proposition("alpha_P002", "shape", category="morphology"),
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="categories are not in canonical order"):
        load_proposition_kb(path)


def test_proposition_index_inherits_disease_source_reference(tmp_path: Path) -> None:
    path = _write_kb(
        tmp_path,
        [_disease("alpha", [_proposition("alpha_P001", "sign")])],
    )
    kb = load_proposition_kb(path)
    output_path = tmp_path / "checklist" / "proposition_index.json"

    kb.write_proposition_index(output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload == [
        {
            "proposition_id": "alpha_P001",
            "disease_id": "alpha",
            "disease_index": 0,
            "category": "morphology",
            "canonicalDescription": "sign",
            "polarity": 1,
            "sourceExcerpt": ["excerpt for alpha_P001"],
            "sourceReference": "source for alpha",
        }
    ]


def test_final_skincap_kb_integration() -> None:
    kb = load_proposition_kb(FINAL_KB_PATH)

    assert kb.num_diseases == 47
    assert kb.num_propositions == 1436
    assert int((kb.prop_polarity == -1).sum()) == 31
    # Counted directly from the immutable final JSON. kb_stats.json currently
    # reports 1283 and should be regenerated separately by the KB pipeline.
    assert len({record.canonical_description for record in kb.propositions}) == 1284
    assert int(kb.disease_prop_counts.min()) == 15
    assert int(kb.disease_prop_counts.max()) == 51
    assert len(kb.negative_alignment_texts) == 31
    assert all(text.startswith("absence of ") for text in kb.negative_alignment_texts)
    assert {record.category for record in kb.propositions} == set(CATEGORY_ORDER)
    assert all(indices.numel() > 0 for indices in kb.disease_to_prop_indices)

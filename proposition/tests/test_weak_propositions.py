import csv
import json
from pathlib import Path

from proposition.weak.propositions import (
    REVIEW_COLUMNS,
    build_review_rows,
    classify_expression,
    classify_observability,
    export_final_version,
    write_review_csv,
)


def _write_kb(tmp_path: Path) -> Path:
    payload = [
        {
            "disease_id": "alpha",
            "sourceReference": "source",
            "propositions": [
                {
                    "proposition_id": "alpha_P001",
                    "category": "morphology",
                    "canonicalDescription": "papules or nodules",
                    "polarity": 1,
                    "sourceExcerpt": ["source"],
                },
                {
                    "proposition_id": "alpha_P002",
                    "category": "location",
                    "canonicalDescription": "lesions on the face, chest and back",
                    "polarity": 1,
                    "sourceExcerpt": ["source"],
                },
                {
                    "proposition_id": "alpha_P003",
                    "category": "symptom",
                    "canonicalDescription": "itching",
                    "polarity": 1,
                    "sourceExcerpt": ["source"],
                },
                {
                    "proposition_id": "alpha_P004",
                    "category": "lab",
                    "canonicalDescription": "eosinophilia",
                    "polarity": 1,
                    "sourceExcerpt": ["source"],
                },
            ],
        }
    ]
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_observability_keeps_text_only_categories_and_excludes_lab() -> None:
    assert classify_observability("history", "onset in childhood")[0] == 1
    assert classify_observability("symptom", "itching")[0] == 1
    assert classify_observability("lab", "eosinophilia")[0] == 0
    assert classify_observability("location", "renal involvement")[0] == 0


def test_expression_splitting_is_conservative() -> None:
    assert classify_expression("papules or nodules") == (
        "alternatives",
        ["papules", "nodules"],
        True,
    )
    assert classify_expression("lesions on the face, chest and back") == (
        "site_list",
        ["lesions on the face", "lesions on the chest", "lesions on the back"],
        True,
    )
    assert classify_expression("bull's-eye rash with central red spot, central clearing and outer red ring") == (
        "pattern",
        ["bull's-eye rash with central red spot, central clearing and outer red ring"],
        False,
    )


def test_review_and_final_export_preserve_kb_order(tmp_path: Path) -> None:
    kb_path = _write_kb(tmp_path)
    rows = build_review_rows(kb_path)
    assert len(rows) == 4
    assert {row["proposition_id"] for row in rows} == {
        "alpha_P001",
        "alpha_P002",
        "alpha_P003",
        "alpha_P004",
    }
    symptom = next(row for row in rows if row["proposition_id"] == "alpha_P003")
    assert symptom["auto_observable"] == 1
    assert symptom["image_applicable"] == 0

    review_path = write_review_csv(rows, tmp_path / "review.csv")
    with review_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == REVIEW_COLUMNS

    paths = export_final_version(kb_path, review_path, tmp_path / "v1", reviewed_on="2026-10-13")
    final = json.loads(paths["final_json"].read_text(encoding="utf-8"))
    assert [item["proposition_id"] for item in final] == [
        "alpha_P001",
        "alpha_P002",
        "alpha_P003",
    ]
    assert final[0]["hypotheses"] == [
        "The patient has papules.",
        "The patient has nodules.",
    ]
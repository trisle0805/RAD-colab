import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from proposition.data.weak_supervision import UNKNOWN_LABEL, load_weak_supervision_v3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifacts(tmp_path: Path, *, mask_order: tuple[str, ...] = ("p1", "p2")) -> dict[str, Path]:
    labels_path = tmp_path / "labels_raw_v3.npz"
    masks_path = tmp_path / "active_masks_v3.npz"
    csv_path = tmp_path / "active_propositions_v3.csv"
    metadata_path = tmp_path / "metadata_final_v3.json"
    np.savez_compressed(
        labels_path,
        case_id=np.asarray(["case-2", "case-1"]),
        split=np.asarray(["train", "val"]),
        proposition_id=np.asarray(["p1", "p2"]),
        label_text=np.asarray([[1, UNKNOWN_LABEL], [0, 1]], dtype=np.int8),
        label_img=np.asarray([[UNKNOWN_LABEL, 0], [1, UNKNOWN_LABEL]], dtype=np.int8),
        p_img=np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
    )
    mask_values = {
        "p1": (1, 0, 1),
        "p2": (0, 1, 1),
    }
    np.savez_compressed(
        masks_path,
        proposition_id=np.asarray(mask_order),
        text_trainable=np.asarray([mask_values[prop][0] for prop in mask_order], dtype=np.int8),
        image_trainable=np.asarray([mask_values[prop][1] for prop in mask_order], dtype=np.int8),
        active_scoring=np.asarray([mask_values[prop][2] for prop in mask_order], dtype=np.int8),
        min_labels=np.asarray(10, dtype=np.int64),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "proposition_id", "disease_id", "category", "canonicalDescription", "polarity",
                "text_trainable", "image_trainable", "active_scoring",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "proposition_id": "p1", "disease_id": "alpha", "category": "morphology",
                    "canonicalDescription": "papules", "polarity": "1", "text_trainable": "1",
                    "image_trainable": "0", "active_scoring": "1",
                },
                {
                    "proposition_id": "p2", "disease_id": "beta", "category": "symptom",
                    "canonicalDescription": "itching", "polarity": "-1", "text_trainable": "0",
                    "image_trainable": "1", "active_scoring": "1",
                },
            ]
        )
    metadata_path.write_text(json.dumps({"version": "v3"}), encoding="utf-8")
    return {"labels": labels_path, "masks": masks_path, "csv": csv_path, "metadata": metadata_path}


def _load(paths: dict[str, Path], **kwargs):
    return load_weak_supervision_v3(
        labels_npz=paths["labels"],
        active_masks_npz=paths["masks"],
        active_csv=paths["csv"],
        metadata_json=paths["metadata"],
        **kwargs,
    )


def test_loads_v3_labels_by_id_and_preserves_unknown(tmp_path: Path) -> None:
    artifacts = _load(_write_artifacts(tmp_path, mask_order=("p2", "p1")))

    assert artifacts.case_ids.tolist() == ["case-2", "case-1"]
    assert artifacts.proposition_ids.tolist() == ["p1", "p2"]
    assert artifacts.label_text[0, 1] == UNKNOWN_LABEL
    assert artifacts.label_img[0, 0] == UNKNOWN_LABEL
    assert artifacts.text_trainable.tolist() == [True, False]
    assert artifacts.image_trainable.tolist() == [False, True]
    assert artifacts.case_indices(["case-1", "case-2"]).tolist() == [1, 0]
    assert artifacts.active_proposition_ids.tolist() == ["p1", "p2"]


def test_rejects_duplicate_case_or_proposition_ids(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    np.savez_compressed(
        paths["labels"],
        case_id=np.asarray(["case-1", "case-1"]), split=np.asarray(["train", "val"]),
        proposition_id=np.asarray(["p1", "p2"]),
        label_text=np.asarray([[1, 0], [0, 1]], dtype=np.int8),
        label_img=np.asarray([[1, 0], [0, 1]], dtype=np.int8),
        p_img=np.zeros((2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="duplicate case_id"):
        _load(paths)


def test_rejects_missing_proposition_id_between_artifacts(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    np.savez_compressed(
        paths["masks"],
        proposition_id=np.asarray(["p1", "other"]),
        text_trainable=np.asarray([1, 0]), image_trainable=np.asarray([0, 1]),
        active_scoring=np.asarray([1, 1]), min_labels=np.asarray(10),
    )
    with pytest.raises(ValueError, match="proposition_id sets differ"):
        _load(paths)


def test_rejects_unknown_values_outside_present_absent_unknown(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    with np.load(paths["labels"], allow_pickle=False) as current:
        payload = {name: current[name] for name in current.files}
    payload["label_text"] = np.asarray([[2, -1], [0, 1]], dtype=np.int8)
    np.savez_compressed(paths["labels"], **payload)
    with pytest.raises(ValueError, match="PRESENT=1, ABSENT=0, UNKNOWN=-1"):
        _load(paths)


def test_validates_manifest_hashes_when_available(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    paths["metadata"].write_text(
        json.dumps({"sha256": {"labels_raw_v3.npz": _sha256(paths["labels"]), "active_masks_v3.npz": "bad"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _load(paths, verify_hashes=True)

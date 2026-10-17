import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from proposition.data.caption_masking import DiseaseAliasMasker
from proposition.data.stage1_dataset import Stage1SkinCAPDataset
from proposition.data.weak_supervision import load_weak_supervision_v3


def _artifacts(tmp_path: Path):
    labels = tmp_path / "labels.npz"
    masks = tmp_path / "masks.npz"
    metadata = tmp_path / "metadata.json"
    propositions = tmp_path / "propositions.csv"
    np.savez(
        labels,
        case_id=np.asarray(["case-b", "case-a"]),
        split=np.asarray(["val", "train"]),
        proposition_id=np.asarray(["p1"]),
        label_text=np.asarray([[1], [0]], dtype=np.int8),
        label_img=np.asarray([[0], [1]], dtype=np.int8),
        p_img=np.zeros((2, 1), dtype=np.float32),
    )
    np.savez(
        masks,
        proposition_id=np.asarray(["p1"]),
        text_trainable=np.asarray([1]), image_trainable=np.asarray([1]),
        active_scoring=np.asarray([1]), min_labels=np.asarray(1),
    )
    propositions.write_text(
        "proposition_id,disease_id,category,canonicalDescription,polarity,text_trainable,image_trainable,active_scoring\n"
        "p1,disease,morphology,papule,1,1,1,1\n",
        encoding="utf-8",
    )
    metadata.write_text(json.dumps({}), encoding="utf-8")
    return load_weak_supervision_v3(
        labels_npz=labels,
        active_masks_npz=masks,
        active_csv=propositions,
        metadata_json=metadata,
    )


def _skincap_csv(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    image_root.mkdir()
    path = tmp_path / "skincap.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "image", "text", "label_0", "label_1"))
        writer.writeheader()
        writer.writerows((
            {"case_id": "case-a", "image": "case-a.png", "text": "Acne with itchy papules.", "label_0": "1", "label_1": "0"},
            {"case_id": "case-b", "image": "case-b.png", "text": "Eczema is chronic.", "label_0": "0", "label_1": "1"},
        ))
    return path


def test_global_alias_masking_preserves_non_disease_text() -> None:
    masker = DiseaseAliasMasker({"acne": ("acne vulgaris", "AV"), "eczema": ()})
    masked = masker.mask("Acne vulgaris has itchy papules; eczema differs from avocado.")

    assert masked == "[DISEASE] has itchy papules; [DISEASE] differs from avocado."


def test_dataset_masks_global_aliases_and_aligns_weak_labels_by_case_id(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    csv_path = _skincap_csv(tmp_path)
    dataset = Stage1SkinCAPDataset(
        csv_path,
        image_root=tmp_path / "images",
        weak_supervision=artifacts,
        disease_aliases={"acne": (), "eczema": ()},
    )

    record = dataset._records[0]
    assert record["case_id"] == "case-a"
    assert dataset._mask_caption(record["caption"]) == "[DISEASE] with itchy papules."
    item = {key: record[key] for key in ("case_id", "split", "disease_label", "weak_index")}
    assert item == {"case_id": "case-a", "split": "train", "disease_label": (1, 0), "weak_index": 1}
    text_labels, image_labels = dataset.weak_labels(["case-a", "case-b"])
    assert text_labels[:, 0].tolist() == [0, 1]
    assert image_labels[:, 0].tolist() == [1, 0]


def test_dataset_derives_legacy_skincap_case_id_from_image_filename(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts = replace(artifacts, case_ids=np.asarray(["case-b.png", "case-a.png"]))
    image_root = tmp_path / "images"
    csv_path = tmp_path / "legacy_skincap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "caption", "disease_0"))
        writer.writerow(("nested/case-a", "Acne with itchy papules.", "1"))
        writer.writerow(("case-b", "Eczema is chronic.", "0"))
    # SkinCAP artifact IDs use image filenames; legacy CSVs omit a case-ID column.
    # Add suffixes in the CSV input to reproduce actual image-path semantics.
    text = csv_path.read_text(encoding="utf-8").replace("nested/case-a", "nested/case-a.png").replace("case-b,Eczema", "case-b.png,Eczema")
    csv_path.write_text(text, encoding="utf-8")
    dataset = Stage1SkinCAPDataset(
        csv_path, image_root=image_root, weak_supervision=artifacts, disease_aliases=("acne", "eczema")
    )
    assert [record["case_id"] for record in dataset._records] == ["case-a.png", "case-b.png"]


def test_dataset_fails_when_csv_case_has_no_weak_labels(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    csv_path = _skincap_csv(tmp_path)
    with csv_path.open("a", encoding="utf-8") as handle:
        handle.write("missing,case-a.png,benign text,0,0\n")

    with pytest.raises(KeyError, match="no weak labels"):
        Stage1SkinCAPDataset(
            csv_path,
            image_root=tmp_path / "images",
            weak_supervision=artifacts,
            disease_aliases=("acne",),
        )

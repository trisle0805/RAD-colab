from __future__ import annotations

import csv

import numpy as np
import pytest

from proposition.data.stage2_scores import (
    build_stage2_support, load_stage2_disease_targets, load_stage2_propositions,
    load_stage2_score_artifact, validate_stage2_splits,
)


def _artifact(path, *, ids=("a", "b"), proposition_ids=("p1", "p2"), image=(True, False), text=(True, True), active=(True, True), fused=True):
    image_scores = np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    text_scores = np.asarray([[0.4, 0.5], [0.2, 0.1]], dtype=np.float32)
    payload = {"case_id": np.asarray(ids), "proposition_id": np.asarray(proposition_ids), "score_image": image_scores, "score_text": text_scores, "image_trainable": np.asarray(image), "text_trainable": np.asarray(text), "active_scoring": np.asarray(active)}
    if fused: payload["score_fused"] = np.asarray([(image_scores[:, 0] + text_scores[:, 0]) / 2, text_scores[:, 1]]).T
    np.savez(path, **payload)


def _csv(path, *, mismatch=False):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("proposition_id", "disease_id", "category", "polarity", "text_trainable", "image_trainable", "active_scoring")); writer.writeheader()
        writer.writerows((
            {"proposition_id": "p1", "disease_id": "d1", "category": "morphology", "polarity": "1", "text_trainable": "1", "image_trainable": "0" if mismatch else "1", "active_scoring": "1"},
            {"proposition_id": "p2", "disease_id": "d2", "category": "other", "polarity": "-1", "text_trainable": "1", "image_trainable": "0", "active_scoring": "1"},
        ))


def test_load_validates_duplicate_case_and_split_order(tmp_path):
    path = tmp_path / "scores.npz"; _artifact(path, ids=("a", "a"))
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_stage2_score_artifact(path)
    _artifact(path, ids=("a", "b")); train = load_stage2_score_artifact(path)
    other = tmp_path / "other.npz"; _artifact(other, ids=("c", "d"), proposition_ids=("p2", "p1"))
    with pytest.raises(ValueError, match="order"):
        validate_stage2_splits(train, load_stage2_score_artifact(other), load_stage2_score_artifact(other))


def test_metadata_targets_and_routing_contracts(tmp_path):
    path = tmp_path / "scores.npz"; _artifact(path); artifacts = load_stage2_score_artifact(path)
    manifest = tmp_path / "active.csv"; _csv(manifest); props = load_stage2_propositions(manifest, artifacts.proposition_ids)
    assert props[1].polarity == -1
    routing_a = build_stage2_support(artifacts, props, "image_morphology")
    assert routing_a.proposition_mask.tolist() == [True, False] and np.isnan(routing_a.support_scores[:, 1]).all()
    routing_b = build_stage2_support(artifacts, props, "full_fused")
    np.testing.assert_allclose(routing_b.support_scores, [[0.3, 0.5], [0.4, 0.1]])
    routing_c = build_stage2_support(artifacts, props, "full_image_morphology_text_other")
    np.testing.assert_allclose(routing_c.support_scores, [[0.2, 0.5], [0.6, 0.1]])
    _csv(manifest, mismatch=True)
    with pytest.raises(ValueError, match="disagrees"):
        build_stage2_support(artifacts, load_stage2_propositions(manifest, artifacts.proposition_ids), "full_fused")

    labels = tmp_path / "labels.csv"
    labels.write_text("image,label_z,label_a\nfolder/b.png,1,0\nfolder/a.png,0,1\n", encoding="utf-8")
    target = load_stage2_disease_targets(labels, ["a.png", "b.png"])
    assert target.disease_ids == ("a", "z")
    np.testing.assert_array_equal(target.targets, [[1, 0], [0, 1]])


def test_fusion_mismatch_is_rejected(tmp_path):
    path = tmp_path / "scores.npz"; _artifact(path); artifacts = load_stage2_score_artifact(path)
    csv_path = tmp_path / "active.csv"; _csv(csv_path); props = load_stage2_propositions(csv_path, artifacts.proposition_ids)
    bad = artifacts.__class__(**{**artifacts.__dict__, "score_fused": np.zeros_like(artifacts.score_image)})
    with pytest.raises(ValueError, match="does not match"):
        build_stage2_support(bad, props, "full_fused")
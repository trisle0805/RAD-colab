from __future__ import annotations

import csv
import json
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from proposition.cli import stage2_concept_cv, stage2_concept_evaluate
from proposition.engine.stage2_concept_ensemble import FrozenConceptLinearHead


def _scores(path, case_ids: tuple[str, ...]) -> None:
    row_count = len(case_ids)
    image = np.asarray([[(index % 2) * .7 + .1, .9 - (index % 2) * .7] for index in range(row_count)], dtype=np.float32)
    text = np.asarray([[(index % 2) * .6 + .2, .8 - (index % 2) * .5] for index in range(row_count)], dtype=np.float32)
    np.savez(path, case_id=np.asarray(case_ids), proposition_id=np.asarray(("p1", "p2")), score_image=image, score_text=text,
             score_fused=(image + text) / 2, image_trainable=np.asarray((True, True)), text_trainable=np.asarray((True, True)), active_scoring=np.asarray((True, True)))


def _labels(path, case_ids: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "label_d1", "label_d2"))
        writer.writeheader()
        for index, case_id in enumerate(case_ids):
            writer.writerow({"case_id": case_id, "label_d1": int(index % 2 == 0), "label_d2": int(index % 2 == 1)})


def _active(path) -> None:
    path.write_text("proposition_id,disease_id,category,polarity,text_trainable,image_trainable,active_scoring\n"
                    "p1,d1,morphology,1,1,1,1\n"
                    "p2,d2,other,1,1,1,1\n", encoding="utf-8")


def test_cv_cli_writes_selected_contract_and_complete_oof(tmp_path, monkeypatch):
    train_ids = tuple(f"train-{index}" for index in range(6))
    val_ids = tuple(f"val-{index}" for index in range(4))
    _scores(tmp_path / "train.npz", train_ids)
    _scores(tmp_path / "val.npz", val_ids)
    _labels(tmp_path / "train.csv", train_ids)
    _labels(tmp_path / "val.csv", val_ids)
    _active(tmp_path / "active.csv")
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(json.dumps([{"run_id": "candidate", "head_mode": "global", "routing_mode": "full_fused", "regularization": "none", "lambda_reg": 0.0, "learning_rate": .01, "batch_size": 2, "max_epochs": 1, "patience": 1}]), encoding="utf-8")
    output = tmp_path / "cv"
    monkeypatch.setattr(sys, "argv", ["stage2_concept_cv", "--train_scores", str(tmp_path / "train.npz"), "--val_scores", str(tmp_path / "val.npz"),
        "--train_csv", str(tmp_path / "train.csv"), "--val_csv", str(tmp_path / "val.csv"), "--active_csv", str(tmp_path / "active.csv"),
        "--candidate_config_json", str(candidate_path), "--output_dir", str(output), "--folds", "5"])
    stage2_concept_cv.main()
    assert (output / "selected_config.json").is_file()
    oof = np.load(output / "oof_predictions.npz")
    assert sorted(oof["case_id"].tolist()) == sorted(train_ids + val_ids)
    assert np.allclose(oof["probabilities"].sum(axis=1), 1.0)
    summary = json.loads((output / "cv_summary.json").read_text(encoding="utf-8"))
    assert summary["oof_predictions_path"] == "oof_predictions.npz"


def test_frozen_evaluator_has_no_training_inputs_and_writes_channel_metadata(tmp_path, monkeypatch):
    case_ids = ("test-a", "test-b")
    _scores(tmp_path / "test.npz", case_ids)
    _labels(tmp_path / "test.csv", case_ids)
    _active(tmp_path / "active.csv")
    head = FrozenConceptLinearHead(
        feature_ids=("p1::image", "p1::text", "p2::fused"), base_proposition_ids=("p1", "p1", "p2"), feature_disease_ids=("d1", "d1", "d2"),
        feature_categories=("morphology", "morphology", "other"), feature_polarities=np.asarray((1, 1, 1)), feature_sources=("image", "text", "fused"), disease_ids=("d1", "d2"),
        effective_weights=np.asarray(((1.0, 1.0, 0.0), (0.0, 0.0, 1.0))), bias=np.zeros(2), head_mode="local", routing_mode="full_dual_morphology", regularization="l2", lambda_reg=.1, refit_epochs=3, seeds=(42, 43, 44),
    )
    frozen = tmp_path / "head.npz"
    head.save(frozen)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["stage2_concept_evaluate", "--frozen_head", str(frozen), "--test_scores", str(tmp_path / "test.npz"),
        "--test_csv", str(tmp_path / "test.csv"), "--active_csv", str(tmp_path / "active.csv"), "--output_dir", str(output), "--bootstrap_samples", "4"])
    stage2_concept_evaluate.main()
    contributions = np.load(output / "contributions" / "test_contributions.npz")
    assert contributions["feature_category"].tolist() == ["morphology", "morphology", "other"]
    assert (output / "diagnostics" / "contribution_reconstruction_check_test.json").is_file()

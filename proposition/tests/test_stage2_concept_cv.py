from __future__ import annotations

import json

import numpy as np
import pytest

from proposition.data.stage2_scores import Stage2Dataset
from proposition.engine.stage2_concept_cv import (
    load_stage2_cv_candidates,
    make_stratified_fold_assignments,
    pool_stage2_datasets,
    select_stage2_cv_configuration,
)


def test_stratified_assignments_are_deterministic_and_cover_each_case_once():
    case_ids = [f"case-{index}" for index in range(15)]
    targets = np.eye(3, dtype=np.int8)[np.repeat(np.arange(3), 5)]

    first = make_stratified_fold_assignments(case_ids, targets, ("d1", "d2", "d3"), n_splits=5, seed=12, original_splits=["train"] * 15)
    repeated = make_stratified_fold_assignments(case_ids, targets, ("d1", "d2", "d3"), n_splits=5, seed=12, original_splits=["train"] * 15)

    np.testing.assert_array_equal(first.folds, repeated.folds)
    assert sorted(first.folds.tolist()) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert len({row["case_id"] for row in first.rows(("d1", "d2", "d3"))}) == 15
    with pytest.raises(ValueError, match="at least 5"):
        make_stratified_fold_assignments(case_ids[:-1], targets[:-1], ("d1", "d2", "d3"), n_splits=5, seed=12)


def test_pool_rejects_case_overlap_and_retains_origin():
    train = Stage2Dataset(np.asarray(("a",)), np.asarray(((0.1,),)), np.asarray(((1, 0),)))
    validation = Stage2Dataset(np.asarray(("b",)), np.asarray(((0.2,),)), np.asarray(((0, 1),)))
    pooled, origins = pool_stage2_datasets(train, validation)
    assert pooled.case_ids.tolist() == ["a", "b"]
    assert origins.tolist() == ["train", "val"]
    with pytest.raises(ValueError, match="overlap"):
        pool_stage2_datasets(train, Stage2Dataset(np.asarray(("a",)), np.asarray(((0.2,),)), np.asarray(((0, 1),))))


def test_cv_selector_applies_round_a_policy_and_rejects_test_metrics():
    summaries = [
        {"candidate": {"run_id": "global-d", "head_mode": "global", "routing_mode": "full_dual_morphology", "regularization": "l1", "lambda_reg": 1e-4, "learning_rate": .01, "batch_size": 8, "max_epochs": 10, "patience": 2}, "cv_metrics": {"mean_top_1_accuracy": .70, "mean_cross_entropy": .4}},
        {"candidate": {"run_id": "local-b", "head_mode": "local", "routing_mode": "full_fused", "regularization": "l2", "lambda_reg": .01, "learning_rate": .01, "batch_size": 8, "max_epochs": 10, "patience": 2}, "cv_metrics": {"mean_top_1_accuracy": .69, "mean_cross_entropy": .5}},
    ]
    result = select_stage2_cv_configuration(summaries)
    assert result["selected_candidate"]["candidate"]["run_id"] == "local-b"
    with pytest.raises(ValueError, match="test fields"):
        select_stage2_cv_configuration([{**summaries[0], "test_metrics": {}}])


def test_candidate_json_is_allow_list_and_rejects_test_paths(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"run_id": "g", "head_mode": "global", "routing_mode": "full_fused", "regularization": "l1", "lambda_reg": 0.01, "learning_rate": 0.1, "batch_size": 3, "max_epochs": 4, "patience": 2}]), encoding="utf-8")
    assert load_stage2_cv_candidates(good)[0].run_id == "g"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"run_id": "g", "head_mode": "global", "routing_mode": "full_fused", "regularization": "l1", "lambda_reg": 0.01, "learning_rate": 0.1, "batch_size": 3, "max_epochs": 4, "patience": 2, "test_path": "forbidden"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="test fields"):
        load_stage2_cv_candidates(bad)

import csv
import json
import random

import numpy as np
import pytest
import torch

from proposition.engine.artifacts import (
    append_metrics_history,
    capture_rng_state,
    final_metrics,
    restore_rng_state,
    save_checkpoint,
    save_final_metrics,
    save_validation_artifacts,
)
from proposition.engine.trainer import EpochResult


def _result(with_label: bool = True) -> EpochResult:
    gt = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    proposition = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]
    )
    label = proposition.roll(1, dims=1) if with_label else None
    return EpochResult(
        losses={"total": 1.2, "proposition": 0.7},
        gt=gt,
        pred_proposition=proposition,
        pred_label=label,
    )


def test_final_metrics_contains_both_metric_tiers() -> None:
    result = _result()
    metrics = final_metrics(result.gt.numpy(), result.pred_proposition.numpy())

    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["top1_accuracy"] == pytest.approx(1.0)
    assert "mrr_at_3" in metrics
    assert "macro_auc" in metrics
    assert "mAP" in metrics
    assert set(metrics["rad_legacy_metrics"]) == {
        "mean_auc", "mean_ap", "mean_accuracy", "mean_f1",
        "mean_precision", "mean_recall", "subset_accuracy",
    }


def test_validation_uses_only_proposition_macro_f1_and_writes_optional_label(tmp_path) -> None:
    summary = save_validation_artifacts(_result(), tmp_path, epoch=2)

    assert summary["validation_score"] == summary["proposition_macro_f1"]
    assert "label_macro_f1" in summary
    path = tmp_path / "predictions" / "val_epoch_002.npz"
    with np.load(path) as predictions:
        assert set(predictions.files) == {"gt", "pred_proposition", "pred_label"}


def test_disabled_label_branch_is_omitted_from_artifacts(tmp_path) -> None:
    result = _result(with_label=False)
    summary = save_validation_artifacts(result, tmp_path, epoch=1)
    final = save_final_metrics(
        result, tmp_path, best_epoch=1, best_validation_score=0.75
    )

    assert not any(key.startswith("label_") for key in summary)
    assert "label_aux" not in final
    with np.load(tmp_path / "predictions" / "test_best.npz") as predictions:
        assert set(predictions.files) == {"gt", "pred_proposition"}
    with (tmp_path / "final_metrics.json").open(encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert "label_aux" not in persisted


def test_metrics_history_appends_and_updates_latest(tmp_path) -> None:
    append_metrics_history(tmp_path, {"epoch": 1, "validation_score": 0.4})
    append_metrics_history(tmp_path, {"epoch": 2, "validation_score": 0.5})

    with (tmp_path / "metrics" / "metrics_history.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    with (tmp_path / "metrics" / "latest_metrics.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["epoch"] == 2


def test_rng_capture_restore_reproduces_all_cpu_streams() -> None:
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    generator = torch.Generator().manual_seed(4)
    state = capture_rng_state(generator)

    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )
    restore_rng_state(state, generator)
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )
    assert actual == expected


def test_atomic_checkpoint_contains_rng_state(tmp_path) -> None:
    generator = torch.Generator().manual_seed(8)
    path = tmp_path / "checkpoints" / "latest.pt"
    save_checkpoint(path, {"epoch": 3, "rng_state": capture_rng_state(generator)})

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["epoch"] == 3
    assert set(checkpoint["rng_state"]) == {
        "python", "numpy", "torch_cpu", "torch_cuda", "pecl_generator"
    }
    assert not path.with_suffix(".pt.tmp").exists()

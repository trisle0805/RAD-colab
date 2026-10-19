from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from proposition.data.stage2_scores import Stage2Dataset
from proposition.engine.stage2_trainer import Stage2Trainer, Stage2TrainerConfig
from proposition.models.stage2_diagnosis import PropositionDiagnosisLayer


def _dataset() -> Stage2Dataset:
    return Stage2Dataset(np.asarray(["a", "b", "c", "d"]), np.asarray([[.9, .2], [.8, .1], [.2, .8], [.1, .9]], dtype=np.float32), np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8))


def _trainer(tmp_path, *, selection_metric="val_loss"):
    model = PropositionDiagnosisLayer(proposition_disease_index=torch.tensor([0, 1]), proposition_polarity=torch.tensor([1, 1]), proposition_mask=torch.tensor([True, True]), num_diseases=2)
    return Stage2Trainer(model=model, disease_ids=("d1", "d2"), proposition_ids=("p1", "p2"), routing_mode="full_fused", input_hashes={"input": "hash"}, score_source=("fused", "fused"), config=Stage2TrainerConfig(max_epochs=3, early_stopping_patience=2, learning_rate=.1, batch_size=2, selection_metric=selection_metric), output_dir=tmp_path)


def test_trainer_optimizes_only_calibration_selects_validation_and_exports(tmp_path):
    trainer = _trainer(tmp_path)
    assert {id(p) for group in trainer.optimizer.param_groups for p in group["params"]} == {id(trainer.model.raw_gamma), id(trainer.model.bias)}
    history = trainer.fit(_dataset(), _dataset())
    assert history and trainer.best_epoch is not None
    trainer.export_best_split(_dataset(), "test")
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "predictions" / "test_best.npz").exists()
    assert (tmp_path / "diagnostics" / "contributions_test.npz").exists()
    trainer.load_checkpoint(tmp_path / "checkpoints" / "best.pt")
    trainer.input_hashes = {"input": "changed"}
    with pytest.raises(ValueError, match="hashes"):
        trainer.load_checkpoint(tmp_path / "checkpoints" / "best.pt")

def test_trainer_selects_lowest_validation_bce_loss_by_default(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    assert trainer.config.selection_metric == "val_loss"
    losses = iter((0.8, 0.4, 0.6))
    monkeypatch.setattr(trainer, "_train_epoch", lambda dataset: 0.0)
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda dataset: {"loss": next(losses), "metrics": {"macro_roc_auc": 0.5}},
    )

    trainer.fit(_dataset(), _dataset())

    assert trainer.best_epoch == 2
    assert trainer.best_metric == pytest.approx(0.4)
    assert [record["selection_metric"] for record in trainer.history] == [0.8, 0.4, 0.6]

def test_trainer_selects_highest_optional_validation_metric(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, selection_metric="macro_f1")
    assert trainer.best_metric == float("-inf")
    metrics = iter((0.2, 0.7, 0.4))
    monkeypatch.setattr(trainer, "_train_epoch", lambda dataset: 0.0)
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda dataset: {"loss": 1.0, "metrics": {"macro_f1": next(metrics)}},
    )

    trainer.fit(_dataset(), _dataset())

    assert trainer.best_epoch == 2
    assert trainer.best_metric == pytest.approx(0.7)
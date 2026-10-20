from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from proposition.data.stage2_scores import Stage2Dataset
from proposition.engine.stage2_concept_trainer import Stage2ConceptTrainer, Stage2ConceptTrainerConfig
from proposition.models.stage2_concept_head import Stage2ConceptHead

def _dataset() -> Stage2Dataset:
    return Stage2Dataset(np.asarray(["a", "b", "c", "d", "e", "f"]), np.asarray([[.9, .1], [.8, .2], [.7, .3], [.3, .7], [.2, .8], [.1, .9]], dtype=np.float32), np.eye(2, dtype=np.int8)[[0, 0, 0, 1, 1, 1]])

def _trainer(tmp_path, *, seed=0) -> Stage2ConceptTrainer:
    model = Stage2ConceptHead(proposition_disease_index=torch.tensor([0, 1]), proposition_polarity=torch.tensor([1, 1]), proposition_mask=torch.tensor([True, True]), num_diseases=2, head_mode="global")
    return Stage2ConceptTrainer(model=model, disease_ids=("d1", "d2"), proposition_ids=("p1", "p2"), routing_mode="full_fused", input_hashes={"scores": "hash"}, score_source=("fused", "fused"), config=Stage2ConceptTrainerConfig(max_epochs=3, early_stopping_patience=3, learning_rate=.1, batch_size=2, base_seed=seed), output_dir=tmp_path)


def test_concept_trainer_uses_cross_entropy_and_exports_compact_margin_contributions(tmp_path):
    dataset = Stage2Dataset(np.asarray(["a", "b", "c", "d"]), np.asarray([[.9, .1], [.8, .2], [.2, .8], [.1, .9]], dtype=np.float32), np.eye(2, dtype=np.int8)[[0, 0, 1, 1]])
    model = Stage2ConceptHead(proposition_disease_index=torch.tensor([0, 1]), proposition_polarity=torch.tensor([1, 1]), proposition_mask=torch.tensor([True, True]), num_diseases=2, head_mode="global")
    trainer = Stage2ConceptTrainer(model=model, disease_ids=("d1", "d2"), proposition_ids=("p1", "p2"), routing_mode="full_fused", input_hashes={"scores": "hash"}, score_source=("fused", "fused"), config=Stage2ConceptTrainerConfig(max_epochs=3, early_stopping_patience=2, learning_rate=.1, batch_size=2), output_dir=tmp_path)
    trainer.fit(dataset, dataset)
    result = trainer.export_best_split(dataset, "test")
    assert result["metrics"]["top_1_accuracy"] >= 0.5
    artifact = np.load(tmp_path / "contributions" / "test_contributions.npz")
    assert {"top1_index", "top2_index", "contribution_top1", "contribution_top2", "delta_contribution", "bias"}.issubset(artifact.files)
    rows = np.arange(len(dataset.case_ids))
    reconstructed_margin = artifact["bias"][artifact["top1_index"]] - artifact["bias"][artifact["top2_index"]] + artifact["delta_contribution"].sum(axis=1)
    assert reconstructed_margin == pytest.approx(artifact["top1_logit"] - artifact["top2_logit"])

def test_equal_validation_top1_lower_cross_entropy_replaces_best_checkpoint(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    dataset = _dataset()
    outcomes = iter(((0.5, 0.8), (0.5, 0.4), (0.5, 0.6)))

    monkeypatch.setattr(trainer, "_train_epoch", lambda dataset, *, epoch: float(epoch))
    def evaluate(_dataset):
        top_1, cross_entropy = next(outcomes)
        return {"loss": cross_entropy, "metrics": {"top_1_accuracy": top_1}}
    monkeypatch.setattr(trainer, "evaluate", evaluate)

    trainer.fit(dataset, dataset)

    assert trainer.best_epoch == 2
    assert trainer.best_top_1_accuracy == pytest.approx(0.5)
    assert trainer.best_cross_entropy == pytest.approx(0.4)
    checkpoint = torch.load(tmp_path / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_top_1_accuracy"] == pytest.approx(0.5)
    assert checkpoint["best_cross_entropy"] == pytest.approx(0.4)

def test_training_permutation_is_seeded_by_epoch(tmp_path):
    trainer = _trainer(tmp_path, seed=37)
    first = trainer._training_permutation(6, 1)
    repeated = trainer._training_permutation(6, 1)
    second_epoch = trainer._training_permutation(6, 2)
    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(38)

    assert torch.equal(first, repeated)
    assert torch.equal(first, torch.randperm(6, generator=expected_generator))
    assert not torch.equal(first, second_epoch)
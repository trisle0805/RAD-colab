from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from proposition.data.stage2_scores import Stage2Dataset
from proposition.engine.stage2_concept_trainer import Stage2ConceptTrainer, Stage2ConceptTrainerConfig
from proposition.models.stage2_concept_head import Stage2ConceptHead


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
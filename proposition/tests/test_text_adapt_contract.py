from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from proposition.engine.trainer import Stage1Trainer, TrainerConfig
from proposition.losses import MaskedBCELoss
from proposition.tests.test_stage1_trainer import EvidenceModel, artifacts, tokenizer

def _round_b_trainer(model: EvidenceModel, output_dir: Path) -> Stage1Trainer:
    return Stage1Trainer(
        model=model, tokenizer=tokenizer, weak_supervision=artifacts(),
        loss_fn=MaskedBCELoss(torch.ones(2), torch.ones(2), text_target_semantics="caption_support_binary"),
        config=TrainerConfig(training_scope="text_adapt", text_target_semantics="caption_support_binary"),
        output_dir=output_dir,
    )


def test_text_adapt_optimizer_requires_all_and_only_trainable_parameters() -> None:
    model = EvidenceModel()
    trainer = Stage1Trainer(
        model=model, tokenizer=tokenizer, weak_supervision=artifacts(),
        loss_fn=MaskedBCELoss(torch.ones(2), torch.ones(2), text_target_semantics="caption_support_binary"),
        config=TrainerConfig(training_scope="text_adapt", text_target_semantics="caption_support_binary"),
    )
    assert trainer.config.training_scope == "text_adapt"


def test_round_b_checkpoint_requires_and_restores_anchor(tmp_path: Path) -> None:
    model = EvidenceModel()
    model.query_cache = torch.tensor([[3.0], [4.0]])
    def restore_query_anchor(anchor: torch.Tensor) -> None:
        model.query_cache = anchor.detach().clone()
    model.restore_query_anchor = restore_query_anchor
    trainer = _round_b_trainer(model, tmp_path)
    trainer.save_checkpoint("round_b.pt")
    checkpoint = torch.load(tmp_path / "checkpoints" / "round_b.pt", map_location="cpu", weights_only=False)
    assert torch.equal(checkpoint["query_anchor"], model.query_cache)
    assert isinstance(checkpoint["query_anchor_sha256"], str)
    model.query_cache.zero_()
    trainer.load_checkpoint(tmp_path / "checkpoints" / "round_b.pt", load_optimizer=False)
    assert torch.equal(model.query_cache, torch.tensor([[3.0], [4.0]]))

def test_round_b_checkpoint_without_anchor_fails_loudly(tmp_path: Path) -> None:
    model = EvidenceModel()
    model.query_cache = torch.empty((0, 1))
    trainer = _round_b_trainer(model, tmp_path)
    with pytest.raises(RuntimeError, match="fixed query anchor"):
        trainer.save_checkpoint("round_b.pt")
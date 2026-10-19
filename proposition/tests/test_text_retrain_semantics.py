from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from proposition.data.weak_supervision import ActiveProposition, WeakSupervisionArtifacts
from proposition.engine.metrics import modality_metrics
from proposition.losses import MaskedBCELoss, build_pos_weight_artifacts
from proposition.models import Stage1EvidenceModel
from torch import nn


def _artifacts(labels: np.ndarray) -> WeakSupervisionArtifacts:
    proposition_ids = np.asarray(["p1"])
    row = ActiveProposition("p1", "d1", "category", "description", 1, True, True, True)
    return WeakSupervisionArtifacts(
        case_ids=np.asarray([f"case-{index}" for index in range(len(labels))]),
        splits=np.asarray(["train"] * len(labels)),
        proposition_ids=proposition_ids,
        label_text=labels.reshape(-1, 1).astype(np.int8),
        label_img=labels.reshape(-1, 1).astype(np.int8),
        p_img=np.zeros((len(labels), 1), dtype=np.float32),
        text_trainable=np.asarray([True]),
        image_trainable=np.asarray([True]),
        active_scoring=np.asarray([True]),
        min_labels=1,
        active_propositions=(row,),
        metadata={},
        artifact_hashes={},
    )


def test_caption_support_targets_and_effective_weights() -> None:
    artifacts = _artifacts(np.asarray([1, 1, 0, -1, -1]))
    weights = build_pos_weight_artifacts(
        artifacts,
        text_target_semantics="caption_support_binary",
    )
    row = weights.rows[0]
    assert (row.n_present, row.n_absent, row.n_unknown) == (2, 1, 2)
    assert (row.n_effective_positive, row.n_effective_negative) == (2, 3)
    assert row.pos_weight == pytest.approx(1.5)


def test_unknown_text_changes_only_binary_support_loss() -> None:
    ternary = MaskedBCELoss(torch.ones(1), torch.ones(1))
    binary = MaskedBCELoss(
        torch.ones(1),
        torch.ones(1),
        text_target_semantics="caption_support_binary",
    )
    common = (torch.zeros((1, 1)), torch.zeros((1, 1)), torch.ones(1), torch.ones(1))
    low = ternary(common[0], common[1], torch.tensor([[-1]]), torch.tensor([[-1]]), common[2], common[3])
    high = ternary(torch.tensor([[10.0]]), common[1], torch.tensor([[-1]]), torch.tensor([[-1]]), common[2], common[3])
    assert low.item() == high.item() == 0.0
    low = binary(common[0], common[1], torch.tensor([[-1]]), torch.tensor([[-1]]), common[2], common[3], include_image=False)
    high = binary(torch.tensor([[10.0]]), common[1], torch.tensor([[-1]]), torch.tensor([[-1]]), common[2], common[3], include_image=False)
    assert low.item() != high.item()


def test_text_metrics_use_effective_unknown_negatives() -> None:
    result = modality_metrics(
        [[0.9], [0.8], [0.1], [0.2]],
        [[1], [-1], [0], [-1]],
        target_semantics="caption_support_binary",
        min_present=1,
        min_absent=3,
    )
    assert result["n_eligible"] == 1
    assert result["rows"][0]["n_absent"] == 3


class _Encoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.layer = nn.Linear(dimension, dimension)

    def forward(self, value):
        return value, value


def test_text_only_scope_contains_only_text_modules() -> None:
    artifacts = _artifacts(np.asarray([0]))

    def tokenize(descriptions):
        return {"input_ids": torch.ones((len(descriptions), 1), dtype=torch.long)}

    model = Stage1EvidenceModel(
        image_encoder=_Encoder(4),
        text_encoder=_Encoder(4),
        weak_supervision=artifacts,
        query_tokenizer=tokenize,
        embed_dim=4,
        num_heads=2,
        training_scope="text_only",
    )
    trainable = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    expected = {
        id(parameter)
        for module in (model.text_cross_attention, model.text_head)
        for parameter in module.parameters()
    }
    assert trainable == expected
    assert not model.image_cross_attention.training
    assert not model.image_head.training
    assert not model.image_projection.training

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from proposition.cli.text_adapt import _audit_round_b_optimizer
from proposition.data.weak_supervision import ActiveProposition, WeakSupervisionArtifacts
from proposition.models import Stage1EvidenceModel


class TinyImage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


class TinyBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(16, 8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.pooler = nn.Linear(8, 8)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        values = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            values = torch.tanh(layer(values))
        return SimpleNamespace(last_hidden_state=values)


class TinyTextWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert_model = TinyBert()
        self.mlp_embed = nn.Linear(8, 8)
        self.logit_scale = nn.Parameter(torch.ones(()))


def _artifacts() -> WeakSupervisionArtifacts:
    identifiers = np.asarray(["p1", "p2"])
    propositions = tuple(
        ActiveProposition(identifier, "d", "c", f"canonical {identifier}", 1, True, True, True)
        for identifier in identifiers
    )
    return WeakSupervisionArtifacts(
        case_ids=np.asarray(["case"]), splits=np.asarray(["train"]), proposition_ids=identifiers,
        label_text=np.zeros((1, 2), dtype=np.int8), label_img=np.zeros((1, 2), dtype=np.int8),
        p_img=np.zeros((1, 2), dtype=np.float32), text_trainable=np.ones(2, dtype=bool),
        image_trainable=np.ones(2, dtype=bool), active_scoring=np.ones(2, dtype=bool), min_labels=1,
        active_propositions=propositions, metadata={}, artifact_hashes={},
    )


def _tokenize(captions: tuple[str, ...] | list[str]) -> dict[str, torch.Tensor]:
    ids = torch.arange(1, len(captions) + 1).unsqueeze(1).repeat(1, 3)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _model(mode: str) -> Stage1EvidenceModel:
    model = Stage1EvidenceModel(
        image_encoder=TinyImage(), text_encoder=TinyTextWrapper(), weak_supervision=_artifacts(),
        query_tokenizer=_tokenize, embed_dim=8, num_heads=2, training_scope="text_only",
    )
    model.create_query_anchor()
    model.bert_adaptation = mode
    model.set_training_scope("text_adapt")
    return model.train()


def _inputs() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return torch.randn(2, 4, 3), _tokenize(["caption one", "caption two"])


@pytest.mark.parametrize("mode,expected", [("frozen", []), ("last1", [2]), ("last2", [1, 2])])
def test_round_b_gradient_flow_is_exact(mode: str, expected: list[int]) -> None:
    model = _model(mode)
    outputs = model(*_inputs())
    outputs["logits_txt"].sum().backward()
    for index, layer in enumerate(model.text_encoder.bert_model.encoder.layer):
        gradients = [parameter.grad for parameter in layer.parameters()]
        assert all(gradient is None for gradient in gradients) if index not in expected else any(
            gradient is not None and torch.count_nonzero(gradient) for gradient in gradients
        )
    assert any(parameter.grad is not None for parameter in model.text_cross_attention.parameters())
    assert any(parameter.grad is not None for parameter in model.text_head.parameters())
    assert all(parameter.grad is None for parameter in model.image_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.image_projection.parameters())
    assert all(parameter.grad is None for parameter in model.image_cross_attention.parameters())
    assert all(parameter.grad is None for parameter in model.image_head.parameters())


@pytest.mark.parametrize("mode,expected", [("last1", [2]), ("last2", [1, 2])])
def test_model_train_enforces_round_b_module_modes(mode: str, expected: list[int]) -> None:
    model = _model(mode)
    assert not model.image_encoder.training
    assert not model.image_projection.training
    assert not model.image_cross_attention.training
    assert not model.image_head.training
    assert model.text_cross_attention.training and model.text_head.training
    for index, layer in enumerate(model.text_encoder.bert_model.encoder.layer):
        assert layer.training is (index in expected)
    model.eval()
    assert all(not layer.training for layer in model.text_encoder.bert_model.encoder.layer)


def test_text_adapt_cannot_lazily_recreate_query_anchor() -> None:
    model = _model("last1")
    model.query_cache = torch.empty_like(model.query_cache[:0])
    with pytest.raises(RuntimeError, match="fixed query anchor"):
        model(*_inputs())


def test_adapted_text_changes_but_fixed_anchor_keeps_image_scores_identical() -> None:
    model = _model("last1").eval()
    images, tokens = _inputs()
    before = model(images, tokens)
    with torch.no_grad():
        model.text_encoder.bert_model.encoder.layer[-1].weight.add_(0.5)
    after = model(images, tokens)
    assert not torch.allclose(before["scores_txt"], after["scores_txt"])
    assert torch.equal(before["scores_img"], after["scores_img"])


def test_round_b_optimizer_audit_requires_exact_groups_and_learning_rates() -> None:
    model = _model("last2")
    heads = [parameter for name in ("text_cross_attention", "text_head") for parameter in getattr(model, name).parameters()]
    bert = [parameter for layer in model.text_encoder.bert_model.encoder.layer[-2:] for parameter in layer.parameters()]
    optimizer = torch.optim.AdamW(({"params": heads, "lr": 3e-4}, {"params": bert, "lr": 2e-5}))
    audit = _audit_round_b_optimizer(model, optimizer, "last2", 3e-4, 2e-5)
    assert audit["head"]["lr"] == 3e-4
    assert audit["bert"]["lr"] == 2e-5
    assert len(audit["head"]["parameter_names"]) == len(heads)
    assert len(audit["bert"]["parameter_names"]) == len(bert)
    optimizer.param_groups[1]["lr"] = 1e-4
    with pytest.raises(RuntimeError, match="learning rates"):
        _audit_round_b_optimizer(model, optimizer, "last2", 3e-4, 2e-5)
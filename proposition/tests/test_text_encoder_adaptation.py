from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from proposition.engine.text_encoder_adaptation import (
    configure_text_encoder_adaptation,
    enforce_text_encoder_modes,
    selected_bert_layers,
)


class TinyBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(10, 4)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.pooler = nn.Linear(4, 4)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        result = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            result = layer(result)
        return SimpleNamespace(last_hidden_state=result)


class Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert_model = TinyBert()
        self.mlp_embed = nn.Linear(4, 4)
        self.logit_scale = nn.Parameter(torch.ones(()))


@pytest.mark.parametrize("mode,expected", [("frozen", []), ("last1", [2]), ("last2", [1, 2])])
def test_configure_text_encoder_adaptation_unfreezes_exact_final_layers(mode: str, expected: list[int]) -> None:
    wrapper = Wrapper()
    selected = configure_text_encoder_adaptation(wrapper, mode)
    assert selected == [wrapper.bert_model.encoder.layer[index] for index in expected]
    for index, layer in enumerate(wrapper.bert_model.encoder.layer):
        assert all(parameter.requires_grad is (index in expected) for parameter in layer.parameters())
    assert all(not parameter.requires_grad for parameter in wrapper.bert_model.embeddings.parameters())
    assert all(not parameter.requires_grad for parameter in wrapper.bert_model.pooler.parameters())
    assert all(not parameter.requires_grad for parameter in wrapper.mlp_embed.parameters())
    assert wrapper.logit_scale.requires_grad is False


def test_selected_layers_have_train_mode_but_frozen_layers_stay_eval() -> None:
    wrapper = Wrapper()
    selected = configure_text_encoder_adaptation(wrapper, "last2")
    enforce_text_encoder_modes(wrapper, selected, True)
    assert wrapper.bert_model.encoder.layer[0].training is False
    assert wrapper.bert_model.encoder.layer[1].training is True
    assert wrapper.bert_model.encoder.layer[2].training is True
    enforce_text_encoder_modes(wrapper, selected, False)
    assert all(not layer.training for layer in wrapper.bert_model.encoder.layer)


def test_invalid_adaptation_mode_or_missing_stack_fails_loudly() -> None:
    with pytest.raises(ValueError, match="bert_adaptation"):
        selected_bert_layers(Wrapper(), "all")
    with pytest.raises(ValueError, match="encoder.layer"):
        configure_text_encoder_adaptation(nn.Linear(1, 1), "last1")
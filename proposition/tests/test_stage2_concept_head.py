from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from proposition.models.stage2_concept_head import Stage2ConceptHead


def _model(mode: str) -> Stage2ConceptHead:
    return Stage2ConceptHead(proposition_disease_index=torch.tensor([0, 0, 1]), proposition_polarity=torch.tensor([1, -1, 1]), proposition_mask=torch.tensor([True, True, True]), num_diseases=2, head_mode=mode)


def test_local_head_is_polarity_adjusted_nonnegative_and_disease_local():
    model = _model("local")
    with torch.no_grad():
        model.raw_weights.copy_(torch.tensor([[0.0, 1.0, 20.0], [20.0, 20.0, 0.0]]))
        model.bias.copy_(torch.tensor([.1, -.2]))
    outputs = model.contribution_outputs(torch.tensor([[.2, .3, .6]]))
    assert torch.allclose(outputs["input_scores"], torch.tensor([[.2, .7, .6]]))
    assert torch.all(outputs["weights"] >= 0)
    assert outputs["weights"][0, 2] == 0 and torch.all(outputs["weights"][1, :2] == 0)
    assert torch.allclose(outputs["reconstructed_logits"], outputs["logits"])
    assert torch.all(outputs["contributions"][0, 0, 2] == 0)


def test_global_head_uses_original_scores_and_signed_terms():
    model = _model("global")
    with torch.no_grad():
        model.raw_weights.copy_(torch.tensor([[1.0, -2.0, 0.0], [0.0, 0.0, 1.0]]))
    outputs = model.contribution_outputs(torch.tensor([[.2, .3, .6]]))
    assert torch.allclose(outputs["input_scores"], torch.tensor([[.2, .3, .6]]))
    assert outputs["contributions"][0, 0, 1] < 0
    assert torch.allclose(outputs["reconstructed_logits"], outputs["logits"])
    assert model.regularization_penalty("l1").item() == pytest.approx(4.0)
    assert model.regularization_penalty("l2").item() == pytest.approx(6.0)


def test_local_head_rejects_a_disease_without_selected_proposition():
    with pytest.raises(ValueError, match="without propositions"):
        Stage2ConceptHead(proposition_disease_index=torch.tensor([0]), proposition_polarity=torch.tensor([1]), proposition_mask=torch.tensor([True]), num_diseases=2, head_mode="local")
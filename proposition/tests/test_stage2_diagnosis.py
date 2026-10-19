from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from proposition.models.stage2_diagnosis import PropositionDiagnosisLayer


def test_diagnosis_applies_polarity_mean_gamma_and_contributions():
    model = PropositionDiagnosisLayer(proposition_disease_index=torch.tensor([0, 0, 1]), proposition_polarity=torch.tensor([1, -1, 1]), proposition_mask=torch.tensor([True, True, True]), num_diseases=2, gamma_init=2.0, bias_init=0.1)
    outputs = model.contribution_outputs(torch.tensor([[0.2, 0.3, 0.6]]))
    assert torch.all(outputs["gamma"] > 0)
    assert torch.allclose(outputs["gamma"], torch.tensor([2.0, 2.0]), atol=1e-5)
    assert torch.allclose(outputs["compatibility_scores"], torch.tensor([[0.2, 0.7, 0.6]]))
    assert torch.allclose(outputs["disease_evidence"], torch.tensor([[0.45, 0.6]]))
    assert torch.allclose(outputs["disease_logit_contributions"] + outputs["bias"], outputs["logits"])
    assert torch.allclose(outputs["probabilities"], torch.sigmoid(outputs["logits"]))
    assert not torch.allclose(outputs["probabilities"].sum(1), torch.ones(1))


def test_diagnosis_rejects_zero_proposition_disease():
    with pytest.raises(ValueError, match="without propositions"):
        PropositionDiagnosisLayer(proposition_disease_index=torch.tensor([0]), proposition_polarity=torch.tensor([1]), proposition_mask=torch.tensor([True]), num_diseases=2)
"""Small, encoder-free multi-label diagnosis calibration layer."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class PropositionDiagnosisLayer(nn.Module):
    """Aggregate routed proposition support and calibrate each disease logit."""

    def __init__(
        self,
        *,
        proposition_disease_index: Tensor,
        proposition_polarity: Tensor,
        proposition_mask: Tensor,
        num_diseases: int,
        gamma_init: float = 1.0,
        bias_init: float = 0.0,
        gamma_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_diseases < 1 or gamma_init <= gamma_eps or gamma_eps <= 0:
            raise ValueError("num_diseases must be positive and gamma_init must exceed gamma_eps")
        disease_index = torch.as_tensor(proposition_disease_index, dtype=torch.long).reshape(-1)
        polarity = torch.as_tensor(proposition_polarity, dtype=torch.float32).reshape(-1)
        mask = torch.as_tensor(proposition_mask, dtype=torch.bool).reshape(-1)
        if not (len(disease_index) == len(polarity) == len(mask)):
            raise ValueError("proposition structure tensors must have identical length")
        if torch.any((disease_index < 0) | (disease_index >= num_diseases)):
            raise ValueError("proposition disease indices are out of range")
        if torch.any((polarity != -1) & (polarity != 1)):
            raise ValueError("proposition polarity must be -1 or 1")
        counts = torch.bincount(disease_index[mask], minlength=num_diseases)
        if torch.any(counts == 0):
            raise ValueError(f"selected routing leaves diseases without propositions: {torch.where(counts == 0)[0].tolist()!r}")
        aggregation = torch.zeros((len(disease_index), num_diseases), dtype=torch.float32)
        selected = torch.where(mask)[0]
        aggregation[selected, disease_index[selected]] = 1.0 / counts[disease_index[selected]].float()
        self.register_buffer("proposition_disease_index", disease_index)
        self.register_buffer("proposition_polarity", polarity)
        self.register_buffer("proposition_mask", mask)
        self.register_buffer("proposition_counts", counts)
        self.register_buffer("aggregation_matrix", aggregation)
        raw_init = math.log(math.expm1(gamma_init - gamma_eps))
        self.raw_gamma = nn.Parameter(torch.full((num_diseases,), raw_init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.full((num_diseases,), bias_init, dtype=torch.float32))
        self.gamma_eps = float(gamma_eps)

    @property
    def gamma(self) -> Tensor:
        return functional.softplus(self.raw_gamma) + self.gamma_eps

    def forward(self, support_scores: Tensor) -> dict[str, Tensor]:
        if support_scores.ndim != 2 or support_scores.shape[1] != self.proposition_mask.numel():
            raise ValueError("support_scores must have shape [batch, proposition]")
        included = support_scores[:, self.proposition_mask]
        if not torch.isfinite(included).all():
            raise ValueError("included proposition support scores must be finite")
        if torch.any(included < -1e-6) or torch.any(included > 1.0 + 1e-6):
            raise ValueError("included proposition support scores must lie in [0, 1]")
        support = support_scores.to(dtype=self.raw_gamma.dtype)
        compatibility = torch.where(
            self.proposition_polarity.unsqueeze(0) > 0,
            support,
            1.0 - support,
        )
        compatibility = torch.where(self.proposition_mask.unsqueeze(0), compatibility, torch.zeros_like(compatibility))
        evidence = compatibility @ self.aggregation_matrix
        gamma = self.gamma
        logits = evidence * gamma.unsqueeze(0) + self.bias.unsqueeze(0)
        return {
            "support_scores": support,
            "compatibility_scores": compatibility,
            "disease_evidence": evidence,
            "gamma": gamma,
            "bias": self.bias,
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
        }

    def contribution_outputs(self, support_scores: Tensor) -> dict[str, Tensor]:
        """Return deterministic proposition terms that sum with bias to logits."""
        outputs = self(support_scores)
        disease_index = self.proposition_disease_index
        denominators = self.proposition_counts[disease_index].to(outputs["compatibility_scores"].dtype)
        mean = outputs["compatibility_scores"] / denominators.unsqueeze(0)
        contributions = mean * outputs["gamma"][disease_index].unsqueeze(0)
        contributions = torch.where(self.proposition_mask.unsqueeze(0), contributions, torch.zeros_like(contributions))
        disease_contributions = torch.zeros_like(outputs["logits"])
        disease_contributions.index_add_(1, disease_index, contributions)
        outputs["mean_contributions"] = mean
        outputs["logit_contributions"] = contributions
        outputs["disease_logit_contributions"] = disease_contributions
        return outputs


def validate_routing_disease_coverage(
    *,
    proposition_disease_ids: Sequence[str],
    disease_ids: Sequence[str],
    proposition_mask: Tensor,
) -> None:
    """Report external disease IDs when selected routing omits a disease."""
    index = {disease: position for position, disease in enumerate(disease_ids)}
    if len(index) != len(disease_ids):
        raise ValueError("disease IDs must be unique")
    try:
        mapped = torch.tensor([index[disease] for disease in proposition_disease_ids], dtype=torch.long)
    except KeyError as error:
        raise ValueError(f"proposition refers to unknown disease {error.args[0]!r}") from error
    mask = torch.as_tensor(proposition_mask, dtype=torch.bool)
    counts = torch.bincount(mapped[mask], minlength=len(disease_ids))
    missing = [disease for disease, count in zip(disease_ids, counts.tolist()) if count == 0]
    if missing:
        raise ValueError(f"selected routing leaves diseases without propositions: {missing!r}")
"""Interpretable Stage-2 linear heads over frozen proposition scores only."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

ConceptHeadMode = Literal["local", "global"]


class Stage2ConceptHead(nn.Module):
    """A linear disease classifier whose only inputs are routed scalar scores.

    ``local`` uses polarity-adjusted evidence and non-negative weights restricted
    to propositions owned by each disease.  ``global`` uses original scores and
    a signed dense disease-by-proposition weight matrix.
    """

    def __init__(
        self,
        *,
        proposition_disease_index: Tensor,
        proposition_polarity: Tensor,
        proposition_mask: Tensor,
        num_diseases: int,
        head_mode: ConceptHeadMode,
    ) -> None:
        super().__init__()
        if head_mode not in {"local", "global"}:
            raise ValueError("head_mode must be 'local' or 'global'")
        if num_diseases < 2:
            raise ValueError("num_diseases must be at least two for single-label diagnosis")
        disease_index = torch.as_tensor(proposition_disease_index, dtype=torch.long).reshape(-1)
        polarity = torch.as_tensor(proposition_polarity, dtype=torch.float32).reshape(-1)
        mask = torch.as_tensor(proposition_mask, dtype=torch.bool).reshape(-1)
        if not (len(disease_index) == len(polarity) == len(mask)):
            raise ValueError("proposition structure tensors must have identical length")
        if torch.any((disease_index < 0) | (disease_index >= num_diseases)):
            raise ValueError("proposition disease indices are out of range")
        if torch.any((polarity != -1) & (polarity != 1)):
            raise ValueError("proposition polarity must be -1 or 1")

        local_connectivity = torch.zeros((num_diseases, len(disease_index)), dtype=torch.bool)
        local_connectivity[disease_index, torch.arange(len(disease_index))] = True
        local_connectivity &= mask.unsqueeze(0)
        if head_mode == "local" and torch.any(~local_connectivity.any(dim=1)):
            missing = torch.where(~local_connectivity.any(dim=1))[0].tolist()
            raise ValueError(f"selected routing leaves diseases without propositions: {missing!r}")

        self.head_mode = head_mode
        self.num_diseases = int(num_diseases)
        self.register_buffer("proposition_disease_index", disease_index)
        self.register_buffer("proposition_polarity", polarity)
        self.register_buffer("proposition_mask", mask)
        self.register_buffer("local_connectivity", local_connectivity)
        self.raw_weights = nn.Parameter(torch.zeros((num_diseases, len(disease_index)), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_diseases, dtype=torch.float32))

    @property
    def effective_weights(self) -> Tensor:
        """Return the auditable ``[disease, proposition]`` linear weights."""
        if self.head_mode == "local":
            return functional.softplus(self.raw_weights) * self.local_connectivity
        return self.raw_weights * self.proposition_mask.unsqueeze(0)

    def _input_scores(self, support_scores: Tensor) -> Tensor:
        if support_scores.ndim != 2 or support_scores.shape[1] != self.proposition_mask.numel():
            raise ValueError("support_scores must have shape [batch, proposition]")
        included = support_scores[:, self.proposition_mask]
        if not torch.isfinite(included).all():
            raise ValueError("included proposition support scores must be finite")
        if torch.any(included < -1e-6) or torch.any(included > 1.0 + 1e-6):
            raise ValueError("included proposition support scores must lie in [0, 1]")
        scores = torch.nan_to_num(support_scores, nan=0.0).to(dtype=self.raw_weights.dtype)
        if self.head_mode == "local":
            scores = torch.where(self.proposition_polarity.unsqueeze(0) > 0, scores, 1.0 - scores)
        return scores * self.proposition_mask.unsqueeze(0)

    def forward(self, support_scores: Tensor) -> dict[str, Tensor]:
        input_scores = self._input_scores(support_scores)
        weights = self.effective_weights
        logits = input_scores @ weights.transpose(0, 1) + self.bias.unsqueeze(0)
        return {
            "support_scores": torch.nan_to_num(support_scores, nan=0.0).to(dtype=self.raw_weights.dtype),
            "input_scores": input_scores,
            "weights": weights,
            "bias": self.bias,
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
        }

    def contribution_outputs(self, support_scores: Tensor) -> dict[str, Tensor]:
        """Return exact per-proposition terms satisfying bias + sum = logit."""
        outputs = self(support_scores)
        contributions = outputs["input_scores"][:, None, :] * outputs["weights"][None, :, :]
        outputs["contributions"] = contributions
        outputs["reconstructed_logits"] = self.bias.unsqueeze(0) + contributions.sum(dim=2)
        return outputs

    def regularization_penalty(self, regularization: str) -> Tensor:
        """Return the production penalty over effective, reachable weights."""
        weights = self.effective_weights
        if regularization == "none":
            return weights.new_zeros(())
        if regularization == "l1":
            return weights.abs().sum()
        if regularization == "l2":
            return weights.square().sum()
        raise ValueError("regularization must be 'none', 'l1', or 'l2'")
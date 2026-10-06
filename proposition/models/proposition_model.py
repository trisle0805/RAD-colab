"""Proposition-based prediction modules.

The modules in this file operate only on encoded tensors. Tokenization, image/text
encoding, KB loading, and loss computation belong to the training pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


AggregationMode = Literal["mean", "weighted"]
QueryLevel = Literal["proposition", "disease"]


@dataclass
class AttentionOutput:
    context: Tensor
    attention: Tensor | None


@dataclass
class PropositionOutput:
    logits: Tensor
    probabilities: Tensor
    support: Tensor
    compatibility: Tensor
    disease_scores: Tensor
    attention: Tensor | None


@dataclass
class LabelBranchOutput:
    logits: Tensor
    probabilities: Tensor
    attention: Tensor | None


@dataclass
class ModelOutput:
    proposition: PropositionOutput
    label: LabelBranchOutput | None = None


class CosineCrossAttention(nn.Module):
    """Single-head cosine cross-attention without residual or FFN blocks."""

    def __init__(self, embed_dim: int, tau_att: float = 0.1) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if tau_att <= 0:
            raise ValueError("tau_att must be positive")

        self.embed_dim = embed_dim
        self.tau_att = float(tau_att)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
        *,
        return_attention: bool = True,
        query_chunk_size: int | None = None,
    ) -> AttentionOutput:
        """Attend from ``query`` to ``memory``.

        ``query`` may have shape ``(Q, d)`` (shared across the batch) or
        ``(B, Q, d)``. ``memory_padding_mask`` has shape ``(B, L)`` and uses
        ``True`` for positions that must not receive attention.
        """

        if memory.ndim != 3 or memory.shape[-1] != self.embed_dim:
            raise ValueError(f"memory must have shape (B, L, {self.embed_dim})")
        batch_size, memory_length, _ = memory.shape

        if query.ndim == 2:
            if query.shape[-1] != self.embed_dim:
                raise ValueError(f"query must have final dimension {self.embed_dim}")
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
        elif query.ndim == 3:
            if query.shape[0] != batch_size or query.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"batched query must have shape (B, Q, {self.embed_dim})"
                )
        else:
            raise ValueError("query must have shape (Q, d) or (B, Q, d)")

        mask: Tensor | None = None
        if memory_padding_mask is not None:
            if memory_padding_mask.shape != (batch_size, memory_length):
                raise ValueError("memory_padding_mask must have shape (B, L)")
            mask = memory_padding_mask.to(device=memory.device, dtype=torch.bool)
            if mask.all(dim=1).any():
                raise ValueError("each sample must contain at least one unmasked memory token")

        query_count = query.shape[1]
        if query_chunk_size is None:
            query_chunk_size = query_count
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when provided")

        projected_keys = F.normalize(self.k_proj(memory), p=2, dim=-1)
        projected_values = self.v_proj(memory)
        contexts: list[Tensor] = []
        attentions: list[Tensor] = []

        for start in range(0, query_count, query_chunk_size):
            query_chunk = query[:, start : start + query_chunk_size]
            projected_queries = F.normalize(self.q_proj(query_chunk), p=2, dim=-1)
            logits = torch.matmul(projected_queries, projected_keys.transpose(-2, -1))
            logits = logits / self.tau_att
            if mask is not None:
                logits = logits.masked_fill(mask.unsqueeze(1), float("-inf"))
            attention = torch.softmax(logits, dim=-1)
            contexts.append(torch.matmul(attention, projected_values))
            if return_attention:
                attentions.append(attention)

        context = torch.cat(contexts, dim=1)
        full_attention = torch.cat(attentions, dim=1) if return_attention else None
        return AttentionOutput(context=context, attention=full_attention)


class SharedScorer(nn.Module):
    """Shared two-layer scorer for query/evidence pairs."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.embed_dim = embed_dim
        self.layers = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, context: Tensor, query: Tensor) -> Tensor:
        if context.ndim != 3 or context.shape[-1] != self.embed_dim:
            raise ValueError(f"context must have shape (B, Q, {self.embed_dim})")
        if query.ndim == 2:
            if query.shape != context.shape[1:]:
                raise ValueError("shared query must have shape (Q, d) matching context")
            query = query.unsqueeze(0).expand(context.shape[0], -1, -1)
        elif query.shape != context.shape:
            raise ValueError("batched query must have the same shape as context")
        return self.layers(torch.cat((context, query), dim=-1)).squeeze(-1)


class DiseaseAggregator(nn.Module):
    """Aggregate proposition compatibility independently within each disease."""

    def __init__(
        self,
        prop_disease: Tensor,
        num_diseases: int,
        mode: AggregationMode = "mean",
    ) -> None:
        super().__init__()
        if num_diseases <= 0:
            raise ValueError("num_diseases must be positive")
        if prop_disease.ndim != 1 or prop_disease.numel() == 0:
            raise ValueError("prop_disease must be a non-empty one-dimensional tensor")
        if mode not in ("mean", "weighted"):
            raise ValueError("mode must be 'mean' or 'weighted'")

        indices = prop_disease.to(dtype=torch.long)
        if int(indices.min()) < 0 or int(indices.max()) >= num_diseases:
            raise ValueError("prop_disease contains an out-of-range disease index")
        counts = torch.bincount(indices, minlength=num_diseases)
        if (counts == 0).any():
            raise ValueError("every disease must own at least one proposition")

        self.num_diseases = num_diseases
        self.mode = mode
        self.register_buffer("prop_disease", indices.clone())
        self.register_buffer("disease_prop_counts", counts)
        if mode == "weighted":
            self.aggregation_logits = nn.Parameter(torch.zeros(indices.numel()))
        else:
            self.register_parameter("aggregation_logits", None)

    def weights(self) -> Tensor:
        if self.aggregation_logits is None:
            return self.disease_prop_counts[self.prop_disease].reciprocal().to(torch.float32)

        weights = torch.empty_like(self.aggregation_logits)
        for disease_index in range(self.num_diseases):
            disease_mask = self.prop_disease == disease_index
            weights[disease_mask] = torch.softmax(
                self.aggregation_logits[disease_mask], dim=0
            )
        return weights

    def forward(self, compatibility: Tensor) -> Tensor:
        if compatibility.ndim != 2 or compatibility.shape[1] != self.prop_disease.numel():
            raise ValueError("compatibility must have shape (B, J) matching prop_disease")

        result = compatibility.new_zeros((compatibility.shape[0], self.num_diseases))
        disease_indices = self.prop_disease.unsqueeze(0).expand(compatibility.shape[0], -1)
        if self.mode == "mean":
            result.scatter_add_(1, disease_indices, compatibility)
            return result / self.disease_prop_counts.to(
                device=compatibility.device, dtype=compatibility.dtype
            ).unsqueeze(0)

        weights = self.weights().to(dtype=compatibility.dtype)
        result.scatter_add_(1, disease_indices, compatibility * weights.unsqueeze(0))
        return result


class PositiveCalibration(nn.Module):
    """Per-disease affine calibration with a strictly positive slope."""

    def __init__(
        self,
        num_diseases: int,
        initial_slope: float = 10.0,
        initial_bias: float = -5.0,
    ) -> None:
        super().__init__()
        if num_diseases <= 0:
            raise ValueError("num_diseases must be positive")
        if initial_slope <= 0:
            raise ValueError("initial_slope must be positive")
        inverse_softplus = initial_slope + math.log(-math.expm1(-initial_slope))
        self.rho = nn.Parameter(torch.full((num_diseases,), inverse_softplus))
        self.bias = nn.Parameter(torch.full((num_diseases,), float(initial_bias)))

    @property
    def slope(self) -> Tensor:
        return F.softplus(self.rho)

    def forward(self, disease_scores: Tensor) -> Tensor:
        if disease_scores.ndim != 2 or disease_scores.shape[1] != self.rho.numel():
            raise ValueError("disease_scores must have shape (B, C)")
        return disease_scores * self.slope.unsqueeze(0) + self.bias.unsqueeze(0)


class PropositionPathway(nn.Module):
    """Main prediction pathway for proposition- or disease-level queries."""

    def __init__(
        self,
        embed_dim: int,
        num_diseases: int,
        prop_disease: Tensor | None,
        prop_polarity: Tensor | None,
        *,
        tau_att: float = 0.1,
        aggregation: AggregationMode = "mean",
        query_level: QueryLevel = "proposition",
    ) -> None:
        super().__init__()
        if query_level not in ("proposition", "disease"):
            raise ValueError("query_level must be 'proposition' or 'disease'")
        self.num_diseases = num_diseases
        self.query_level = query_level
        self.attention = CosineCrossAttention(embed_dim, tau_att)
        self.scorer = SharedScorer(embed_dim)
        self.calibration = PositiveCalibration(num_diseases)

        if query_level == "proposition":
            if prop_disease is None or prop_polarity is None:
                raise ValueError("proposition-level prediction requires KB index tensors")
            if prop_polarity.ndim != 1 or prop_polarity.shape != prop_disease.shape:
                raise ValueError("prop_polarity must match prop_disease")
            polarity = prop_polarity.to(dtype=torch.int8)
            if not torch.all((polarity == 1) | (polarity == -1)):
                raise ValueError("prop_polarity values must be -1 or 1")
            self.register_buffer("prop_polarity", polarity.clone())
            self.aggregator: DiseaseAggregator | None = DiseaseAggregator(
                prop_disease, num_diseases, aggregation
            )
        else:
            self.register_buffer("prop_polarity", None)
            self.aggregator = None

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
        *,
        return_attention: bool = False,
        query_chunk_size: int | None = None,
    ) -> PropositionOutput:
        if self.query_level == "disease" and query.shape[-2] != self.num_diseases:
            raise ValueError("disease-level query count must equal num_diseases")

        attended = self.attention(
            query,
            memory,
            memory_padding_mask,
            return_attention=return_attention,
            query_chunk_size=query_chunk_size,
        )
        raw_logits = self.scorer(attended.context, query)
        support = torch.sigmoid(raw_logits)

        if self.query_level == "proposition":
            assert self.aggregator is not None and self.prop_polarity is not None
            polarity = self.prop_polarity.to(device=support.device)
            compatibility = torch.where(polarity.unsqueeze(0) == 1, support, 1.0 - support)
            disease_scores = self.aggregator(compatibility)
        else:
            compatibility = support
            disease_scores = support

        logits = self.calibration(disease_scores)
        return PropositionOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            support=support,
            compatibility=compatibility,
            disease_scores=disease_scores,
            attention=attended.attention,
        )

    def aggregation_weights(self) -> Tensor | None:
        if self.aggregator is None or self.aggregator.mode != "weighted":
            return None
        return self.aggregator.weights()


class AuxiliaryLabelBranch(nn.Module):
    """Independent disease-label query branch used only as auxiliary supervision."""

    def __init__(self, embed_dim: int, tau_att: float = 0.1) -> None:
        super().__init__()
        self.attention = CosineCrossAttention(embed_dim, tau_att)
        self.scorer = SharedScorer(embed_dim)

    def forward(
        self,
        label_query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
        *,
        return_attention: bool = False,
        query_chunk_size: int | None = None,
    ) -> LabelBranchOutput:
        attended = self.attention(
            label_query,
            memory,
            memory_padding_mask,
            return_attention=return_attention,
            query_chunk_size=query_chunk_size,
        )
        logits = self.scorer(attended.context, label_query)
        return LabelBranchOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            attention=attended.attention,
        )


class PropositionModel(nn.Module):
    """Container for the main pathway and optional auxiliary label branch."""

    def __init__(
        self,
        embed_dim: int,
        num_diseases: int,
        prop_disease: Tensor | None,
        prop_polarity: Tensor | None,
        *,
        tau_att: float = 0.1,
        aggregation: AggregationMode = "mean",
        query_level: QueryLevel = "proposition",
        use_label_branch: bool = True,
    ) -> None:
        super().__init__()
        self.pathway = PropositionPathway(
            embed_dim,
            num_diseases,
            prop_disease,
            prop_polarity,
            tau_att=tau_att,
            aggregation=aggregation,
            query_level=query_level,
        )
        self.label_branch = (
            AuxiliaryLabelBranch(embed_dim, tau_att) if use_label_branch else None
        )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
        *,
        label_query: Tensor | None = None,
        return_attention: bool = False,
        query_chunk_size: int | None = None,
    ) -> ModelOutput:
        proposition_output = self.pathway(
            query,
            memory,
            memory_padding_mask,
            return_attention=return_attention,
            query_chunk_size=query_chunk_size,
        )

        label_output: LabelBranchOutput | None = None
        if self.label_branch is not None:
            if label_query is None:
                raise ValueError("label_query is required when the label branch is enabled")
            label_output = self.label_branch(
                label_query,
                memory,
                memory_padding_mask,
                return_attention=return_attention,
                query_chunk_size=query_chunk_size,
            )
        elif label_query is not None:
            raise ValueError("label_query was provided but the label branch is disabled")

        return ModelOutput(proposition=proposition_output, label=label_output)

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
QueryLevel = Literal["proposition", "disease", "proposition_segments"]


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

        query, mask = self._validate_inputs(query, memory, memory_padding_mask)
        batch_size = memory.shape[0]

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
            attention = self.attention_weights(
                query_chunk, memory, mask, projected_keys=projected_keys
            )
            contexts.append(torch.matmul(attention, projected_values))
            if return_attention:
                attentions.append(attention)

        context = torch.cat(contexts, dim=1)
        full_attention = torch.cat(attentions, dim=1) if return_attention else None
        return AttentionOutput(context=context, attention=full_attention)

    def _validate_inputs(
        self,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
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
        if memory_padding_mask is None:
            return query, None
        if memory_padding_mask.shape != (batch_size, memory_length):
            raise ValueError("memory_padding_mask must have shape (B, L)")
        mask = memory_padding_mask.to(device=memory.device, dtype=torch.bool)
        if mask.all(dim=1).any():
            raise ValueError("each sample must contain at least one unmasked memory token")
        return query, mask

    def attention_weights(
        self,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
        *,
        projected_keys: Tensor | None = None,
    ) -> Tensor:
        """Return normalized attention weights without applying the value projection.

        This is intentionally shared by the legacy forward path and the segment
        pathway, whose disease evidence must be formed only after averaging maps.
        """

        query, mask = self._validate_inputs(query, memory, memory_padding_mask)
        keys = (
            F.normalize(self.k_proj(memory), p=2, dim=-1)
            if projected_keys is None
            else projected_keys
        )
        if keys.shape != memory.shape:
            raise ValueError("projected_keys must have shape matching memory")
        projected_queries = F.normalize(self.q_proj(query), p=2, dim=-1)
        logits = torch.matmul(projected_queries, keys.transpose(-2, -1)) / self.tau_att
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1), float("-inf"))
        return torch.softmax(logits, dim=-1)


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


class SegmentQueryPathway(nn.Module):
    """Disease prediction from token queries with post-hoc proposition maps."""

    def __init__(
        self,
        embed_dim: int,
        num_diseases: int,
        prop_disease: Tensor,
        prop_align_index: Tensor,
        num_align_texts: int,
        *,
        tau_att: float = 0.1,
    ) -> None:
        super().__init__()
        if prop_disease.ndim != 1 or prop_disease.numel() == 0:
            raise ValueError("prop_disease must be a non-empty one-dimensional tensor")
        if prop_align_index.shape != prop_disease.shape:
            raise ValueError("prop_align_index must match prop_disease")
        if num_align_texts <= 0:
            raise ValueError("num_align_texts must be positive")
        prop_disease = prop_disease.to(dtype=torch.long)
        prop_align_index = prop_align_index.to(dtype=torch.long)
        if int(prop_disease.min()) < 0 or int(prop_disease.max()) >= num_diseases:
            raise ValueError("prop_disease contains an out-of-range disease index")
        if int(prop_align_index.min()) < 0 or int(prop_align_index.max()) >= num_align_texts:
            raise ValueError("prop_align_index contains an out-of-range alignment index")
        counts = torch.bincount(prop_disease, minlength=num_diseases)
        if (counts == 0).any():
            raise ValueError("every disease must own at least one proposition")

        aggregation = torch.zeros(num_diseases, prop_disease.numel(), dtype=torch.float32)
        aggregation[prop_disease, torch.arange(prop_disease.numel())] = 1.0 / counts[
            prop_disease
        ].to(torch.float32)
        self.embed_dim = embed_dim
        self.num_diseases = num_diseases
        self.num_align_texts = num_align_texts
        self.attention = CosineCrossAttention(embed_dim, tau_att)
        self.scorer = SharedScorer(embed_dim)
        self.calibration = PositiveCalibration(num_diseases)
        self.register_buffer("prop_disease", prop_disease.clone())
        self.register_buffer("prop_align_index", prop_align_index.clone())
        self.register_buffer("P_dis", aggregation)

    def forward(
        self,
        query: Tensor,
        query_token_mask: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
        *,
        return_attention: bool = False,
        query_chunk_size: int | None = None,
    ) -> PropositionOutput:
        if query.ndim != 3 or query.shape[0] != self.num_align_texts or query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"segment query must have shape ({self.num_align_texts}, T, {self.embed_dim})"
            )
        if query_token_mask.shape != query.shape[:2]:
            raise ValueError("query_token_mask must have shape (U, T) matching query")
        content_mask = query_token_mask.to(device=query.device, dtype=torch.bool)
        if not content_mask.any(dim=1).all():
            raise ValueError("each alignment text must have at least one content token")
        if memory.ndim != 3 or memory.shape[-1] != self.embed_dim:
            raise ValueError(f"memory must have shape (B, L, {self.embed_dim})")

        owner, _ = content_mask.nonzero(as_tuple=True)
        token_query = query[content_mask]
        token_count = token_query.shape[0]
        if query_chunk_size is None:
            query_chunk_size = token_count
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when provided")

        batch_size, memory_length, _ = memory.shape
        projected_keys = F.normalize(self.attention.k_proj(memory), p=2, dim=-1)
        text_attention = memory.new_zeros((batch_size, self.num_align_texts, memory_length))
        for start in range(0, token_count, query_chunk_size):
            end = min(start + query_chunk_size, token_count)
            token_attention = self.attention.attention_weights(
                token_query[start:end],
                memory,
                memory_padding_mask,
                projected_keys=projected_keys,
            )
            text_attention.index_add_(1, owner[start:end], token_attention)
        token_counts = content_mask.sum(dim=1).to(device=memory.device, dtype=memory.dtype)
        text_attention = text_attention / token_counts.view(1, -1, 1)

        proposition_attention = text_attention[:, self.prop_align_index]
        disease_attention = torch.einsum(
            "cj,bjl->bcl", self.P_dis.to(dtype=memory.dtype), proposition_attention
        )
        context = torch.matmul(disease_attention, self.attention.v_proj(memory))
        text_query = torch.zeros(
            self.num_align_texts, self.embed_dim, dtype=query.dtype, device=query.device
        )
        text_query.index_add_(0, owner, token_query)
        text_query = text_query / content_mask.sum(dim=1).to(query.dtype).unsqueeze(1)
        proposition_query = text_query[self.prop_align_index]
        disease_query = torch.matmul(self.P_dis.to(dtype=query.dtype), proposition_query)
        support = torch.sigmoid(self.scorer(context, disease_query))
        logits = self.calibration(support)
        return PropositionOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            support=support,
            compatibility=support,
            disease_scores=support,
            attention=proposition_attention if return_attention else None,
        )

    def aggregation_weights(self) -> Tensor | None:
        return None


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
        prop_align_index: Tensor | None = None,
        num_align_texts: int | None = None,
        use_label_branch: bool = True,
    ) -> None:
        super().__init__()
        self.query_level = query_level
        if query_level == "proposition_segments":
            if prop_disease is None or prop_align_index is None or num_align_texts is None:
                raise ValueError("segment queries require prop_disease, prop_align_index, and num_align_texts")
            if aggregation != "mean":
                raise ValueError("--agg weighted is not supported with proposition_segments")
            self.pathway: PropositionPathway | SegmentQueryPathway = SegmentQueryPathway(
                embed_dim,
                num_diseases,
                prop_disease,
                prop_align_index,
                num_align_texts,
                tau_att=tau_att,
            )
        else:
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
        query_token_mask: Tensor | None = None,
        return_attention: bool = False,
        query_chunk_size: int | None = None,
    ) -> ModelOutput:
        if self.query_level == "proposition_segments":
            if query_token_mask is None:
                raise ValueError("query_token_mask is required with proposition_segments")
            assert isinstance(self.pathway, SegmentQueryPathway)
            proposition_output = self.pathway(
                query,
                query_token_mask,
                memory,
                memory_padding_mask,
                return_attention=return_attention,
                query_chunk_size=query_chunk_size,
            )
        else:
            if query_token_mask is not None:
                raise ValueError("query_token_mask is only supported with proposition_segments")
            assert isinstance(self.pathway, PropositionPathway)
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

"""Small cross-attention block used by Stage-1 proposition evidence heads."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn


class PropositionCrossAttention(nn.Module):
    """Attend proposition queries over one modality's feature sequence.

    The block intentionally contains only multi-head attention, a query residual,
    and layer normalization. It has no proposition-ID-specific parameters.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.attention: Final[nn.MultiheadAttention] = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        *,
        memory_key_padding_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return evidence features for every batch/query pair.

        ``queries`` is ``[J, D]`` and ``memory`` is ``[B, N, D]``. A true
        entry in ``memory_key_padding_mask`` excludes that text-memory token.
        """
        if queries.ndim != 2:
            raise ValueError(f"queries must have shape [J, D], got {tuple(queries.shape)}")
        if memory.ndim != 3:
            raise ValueError(f"memory must have shape [B, N, D], got {tuple(memory.shape)}")
        batch_size, memory_length, dimension = memory.shape
        if queries.shape[1] != dimension:
            raise ValueError("query and memory embedding dimensions must match")
        if memory_key_padding_mask is not None and memory_key_padding_mask.shape != (batch_size, memory_length):
            raise ValueError("memory_key_padding_mask must have shape [B, N]")

        batch_queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, weights = self.attention(
            batch_queries,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attention,
        )
        output = self.normalization(batch_queries + attended)
        if return_attention:
            return output, weights
        return output

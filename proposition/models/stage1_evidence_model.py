"""Current-only dual-modality Stage-1 proposition evidence model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from proposition.data.weak_supervision import WeakSupervisionArtifacts

from .cross_attention import PropositionCrossAttention


class Stage1EvidenceModel(nn.Module):
    """Score proposition evidence from frozen image and caption encoders.

    Active queries originate exclusively from each proposition's
    ``canonicalDescription``. Outputs are evidence scores, never disease scores.
    """

    def __init__(
        self,
        *,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        weak_supervision: WeakSupervisionArtifacts,
        query_tokenizer: Callable[[Sequence[str]], Mapping[str, Tensor]],
        embed_dim: int = 768,
        num_heads: int = 8,
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if query_chunk_size is not None and query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive when supplied")
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.query_tokenizer = query_tokenizer
        self.query_chunk_size = query_chunk_size
        self.image_cross_attention = PropositionCrossAttention(embed_dim, num_heads)
        self.text_cross_attention = PropositionCrossAttention(embed_dim, num_heads)
        self.image_head = nn.Linear(embed_dim, 1)
        self.text_head = nn.Linear(embed_dim, 1)

        active_indices = weak_supervision.active_indices
        if not len(active_indices):
            raise ValueError("weak-supervision artifacts have no active propositions")
        descriptions = tuple(
            weak_supervision.active_propositions[index].canonical_description for index in active_indices
        )
        if any(not description for description in descriptions):
            raise ValueError("active proposition canonicalDescription must not be empty")
        self.active_proposition_ids = tuple(weak_supervision.proposition_ids[active_indices].tolist())
        self.active_descriptions = descriptions
        self.register_buffer("text_trainable", torch.as_tensor(weak_supervision.text_trainable[active_indices], dtype=torch.bool))
        self.register_buffer("image_trainable", torch.as_tensor(weak_supervision.image_trainable[active_indices], dtype=torch.bool))
        self.register_buffer("active_scoring", torch.ones(len(active_indices), dtype=torch.bool))
        self.register_buffer("query_cache", torch.empty((0, embed_dim)), persistent=False)
        self._freeze_encoder(self.image_encoder)
        self._freeze_encoder(self.text_encoder)

    @staticmethod
    def _freeze_encoder(encoder: nn.Module) -> None:
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "Stage1EvidenceModel":
        super().train(mode)
        # Keep frozen BERT dropout and ResNet BatchNorm deterministic/frozen.
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def refresh_query_cache(self) -> None:
        """Encode canonical descriptions once, using the frozen text encoder."""
        tokens = self.query_tokenizer(self.active_descriptions)
        device = next(self.text_encoder.parameters(), self.query_cache).device
        tokens = {name: value.to(device) for name, value in tokens.items()}
        query_output = self._encode_text(tokens)
        self.query_cache = self._pooled_features(query_output).detach()

    def _queries(self, device: torch.device) -> Tensor:
        if self.query_cache.numel() == 0:
            self.refresh_query_cache()
        return self.query_cache.to(device)

    @staticmethod
    def _pooled_features(output: Any) -> Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0, :]
        if isinstance(output, Tensor):
            return output
        raise TypeError("encoder output must be a Tensor, tuple, or model output with last_hidden_state")

    @staticmethod
    def _sequence_features(output: Any) -> Tensor:
        if isinstance(output, tuple):
            return output[1] if len(output) > 1 else output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, Tensor):
            return output
        raise TypeError("encoder output must be a Tensor, tuple, or model output with last_hidden_state")

    def _encode_text(self, tokens: Mapping[str, Tensor]) -> Any:
        if hasattr(self.text_encoder, "encode_text"):
            return self.text_encoder.encode_text(tokens)
        return self.text_encoder(**tokens)

    def _encode_image(self, images: Tensor) -> Any:
        return self.image_encoder(images)

    def _run_attention(
        self,
        attention: PropositionCrossAttention,
        head: nn.Linear,
        queries: Tensor,
        memory: Tensor,
        padding_mask: Tensor | None,
        return_attention: bool,
    ) -> tuple[Tensor, list[Tensor] | None]:
        logits, attention_chunks = [], []
        chunk_size = self.query_chunk_size or len(queries)
        for query_chunk in queries.split(chunk_size):
            result = attention(
                query_chunk, memory, memory_key_padding_mask=padding_mask, return_attention=return_attention
            )
            features, weights = result if return_attention else (result, None)
            logits.append(head(features).squeeze(-1))
            if weights is not None:
                attention_chunks.append(weights)
        return torch.cat(logits, dim=1), attention_chunks if return_attention else None

    def forward(
        self,
        images: Tensor,
        caption_tokens: Mapping[str, Tensor],
        *,
        return_attention: bool = False,
        query_embeddings: Tensor | None = None,
    ) -> dict[str, Tensor | None | dict[str, Tensor]]:
        """Compute modality evidence and fixed-fusion support for active queries.

        ``query_embeddings`` is an evaluation-only diagnostic override used for
        query-swap and ablation controls; normal training always uses the cache.
        """
        with torch.no_grad():
            image_memory = self._sequence_features(self._encode_image(images))
            text_memory = self._sequence_features(self._encode_text(caption_tokens))
        queries = self._queries(image_memory.device) if query_embeddings is None else query_embeddings.to(image_memory.device)
        if queries.ndim != 2 or queries.shape[0] != len(self.active_proposition_ids):
            raise ValueError("query_embeddings must be [active_proposition, embed_dim]")
        if queries.shape[-1] != image_memory.shape[-1] or queries.shape[-1] != text_memory.shape[-1]:
            raise ValueError("encoder features and cached query embeddings must share embed_dim")
        text_padding = caption_tokens.get("attention_mask")
        if text_padding is not None:
            text_padding = ~text_padding.to(dtype=torch.bool, device=text_memory.device)
        logits_img, image_attention = self._run_attention(
            self.image_cross_attention, self.image_head, queries, image_memory, None, return_attention
        )
        logits_txt, text_attention = self._run_attention(
            self.text_cross_attention, self.text_head, queries, text_memory, text_padding, return_attention
        )
        scores_img, scores_txt = torch.sigmoid(logits_img), torch.sigmoid(logits_txt)
        fused = torch.where(
            self.image_trainable.unsqueeze(0) & self.text_trainable.unsqueeze(0),
            (scores_img + scores_txt) / 2,
            torch.where(self.text_trainable.unsqueeze(0), scores_txt, scores_img),
        )
        fused = fused.masked_fill(~self.active_scoring.unsqueeze(0), float("nan"))
        output: dict[str, Tensor | None | dict[str, Tensor]] = {
            "logits_img": logits_img,
            "logits_txt": logits_txt,
            "scores_img": scores_img,
            "scores_txt": scores_txt,
            "support_scores": fused,
        }
        if return_attention:
            output["attention"] = {
                "image": torch.cat(image_attention or [], dim=1),
                "text": torch.cat(text_attention or [], dim=1),
            }
        return output

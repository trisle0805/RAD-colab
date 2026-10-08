"""End-to-end adapters connecting RAD encoders to the proposition core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn

from proposition.data.knowledge_base import PropositionKnowledgeBase
from proposition.engine.trainer import EncodedBatch


@dataclass(frozen=True)
class TokenizedKnowledge:
    """Tokenized static knowledge texts and measured length metadata."""

    unique: Mapping[str, Tensor]
    alignment: Mapping[str, Tensor]
    disease: Mapping[str, Tensor]
    max_length: int
    measured_max_length: int
    disease_lengths: tuple[int, ...]


def _to_device(tokens: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in tokens.items()}


def _token_lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    encoded = tokenizer(list(texts), add_special_tokens=True, padding=False, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]


def _tokenize_fixed(tokenizer: Any, texts: Sequence[str], max_length: int) -> Mapping[str, Tensor]:
    tokens = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding="max_length",
        truncation=False,
        max_length=max_length,
        return_tensors="pt",
    )
    if tokens["input_ids"].shape[1] > max_length:
        raise ValueError("tokenizer returned a sequence longer than the selected max_length")
    return tokens


def prepare_knowledge_tokens(
    tokenizer: Any,
    kb: PropositionKnowledgeBase,
    *,
    multiple: int = 8,
    disease_max_length: int = 512,
) -> TokenizedKnowledge:
    """Measure without truncation, then tokenize all static knowledge text."""

    if multiple <= 0 or disease_max_length <= 0:
        raise ValueError("token length limits must be positive")
    measured_lengths = _token_lengths(tokenizer, kb.unique_texts)
    measured_max = max(measured_lengths)
    max_length = ((measured_max + multiple - 1) // multiple) * multiple
    unique = _tokenize_fixed(tokenizer, kb.unique_texts, max_length)
    alignment = _tokenize_fixed(tokenizer, kb.unique_alignment_texts, max_length)

    disease_lengths = tuple(_token_lengths(tokenizer, kb.disease_query_texts))
    disease = tokenizer(
        list(kb.disease_query_texts),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=disease_max_length,
        return_tensors="pt",
    )
    return TokenizedKnowledge(
        unique=unique,
        alignment=alignment,
        disease=disease,
        max_length=max_length,
        measured_max_length=measured_max,
        disease_lengths=disease_lengths,
    )


class EncodedBatchStream(Iterable[EncodedBatch]):
    """Lazily encode raw SkinCAP batches while preserving gradients."""

    def __init__(
        self,
        raw_batches: Iterable[Mapping[str, Any]],
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        kb: PropositionKnowledgeBase,
        knowledge_tokens: TokenizedKnowledge,
        device: torch.device,
        *,
        caption_max_length: int = 512,
        query_level: str = "proposition",
        use_label_branch: bool = True,
        apply_fourier_augmentation: bool = False,
        frozen_unique_pooled: Tensor | None = None,
    ) -> None:
        self.raw_batches = raw_batches
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.kb = kb
        self.knowledge_tokens = knowledge_tokens
        self.device = device
        self.caption_max_length = caption_max_length
        self.query_level = query_level
        self.use_label_branch = use_label_branch
        self.apply_fourier_augmentation = apply_fourier_augmentation
        self.frozen_unique_pooled = frozen_unique_pooled

    def __iter__(self) -> Iterator[EncodedBatch]:
        for raw in self.raw_batches:
            images = raw["image"].to(self.device, non_blocking=True)
            if self.apply_fourier_augmentation:
                # RAD applies this augmentation only to training batches.
                from engine.train_rad import fourier_aug
                images = fourier_aug(images)
            labels = torch.as_tensor(raw["label"], device=self.device, dtype=torch.float32)
            captions = self.tokenizer(
                list(raw["entity"]),
                padding="max_length",
                truncation=True,
                max_length=self.caption_max_length,
                return_tensors="pt",
            )
            captions = _to_device(captions, self.device)
            image_tokens, image_pooled = self.image_encoder(images)
            text_pooled, text_tokens = self.text_encoder.encode_text(captions)
            memory = torch.cat((image_tokens, text_tokens), dim=1)
            image_mask = torch.zeros(
                images.shape[0], image_tokens.shape[1], dtype=torch.bool, device=self.device
            )
            memory_mask = torch.cat((image_mask, captions["attention_mask"].eq(0)), dim=1)

            if self.frozen_unique_pooled is None:
                unique_pooled, _ = self.text_encoder.encode_text(
                    _to_device(self.knowledge_tokens.unique, self.device)
                )
            else:
                unique_pooled = self.frozen_unique_pooled.to(self.device)
            align_to_unique = torch.empty(len(self.kb.unique_alignment_texts), dtype=torch.long)
            align_to_unique[self.kb.align_unique_index] = self.kb.align_text_index
            alignment_pooled = unique_pooled[align_to_unique.to(self.device)]
            if self.query_level == "proposition":
                query = unique_pooled[self.kb.query_text_index.to(self.device)]
            elif self.query_level == "disease":
                query, _ = self.text_encoder.encode_text(
                    _to_device(self.knowledge_tokens.disease, self.device)
                )
            else:
                raise ValueError("query_level must be 'proposition' or 'disease'")

            label_query = None
            if self.use_label_branch:
                label_tokens = self.tokenizer(
                    list(self.kb.disease_ids),
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                label_query, _ = self.text_encoder.encode_text(
                    _to_device(label_tokens, self.device)
                )

            yield EncodedBatch(
                memory=memory,
                query=query,
                labels=labels,
                text_pooled=text_pooled,
                image_pooled=image_pooled,
                alignment_prototypes=alignment_pooled,
                memory_padding_mask=memory_mask,
                label_query=label_query,
                image_token_count=torch.full(
                    (images.shape[0],), image_tokens.shape[1], dtype=torch.int32, device=self.device
                ),
                caption_input_ids=captions["input_ids"],
                caption_attention_mask=captions["attention_mask"],
            )

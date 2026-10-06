"""Proposition-enhanced contrastive loss (PECL).

This module consumes one prototype per unique polarity-aware alignment text. The
knowledge-base loader supplies the disease-to-prototype mapping, thereby keeping
text deduplication and disease membership deterministic and outside the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class PECLLossOutput:
    """Unweighted modality losses and their weighted total."""

    total: Tensor
    text: Tensor
    image: Tensor


class PECLLoss(nn.Module):
    """Polarity-aware proposition contrastive loss for text and image features."""

    def __init__(
        self,
        disease_to_prototypes: tuple[Tensor, ...] | list[Tensor],
        num_prototypes: int,
        *,
        negative_ratio: int = 5,
        text_temperature: float = 0.5,
        image_temperature: float = 2.0,
        text_weight: float = 0.1,
        image_weight: float = 0.001,
    ) -> None:
        super().__init__()
        if not disease_to_prototypes:
            raise ValueError("disease_to_prototypes must not be empty")
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be non-negative")
        if text_temperature <= 0 or image_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if text_weight < 0 or image_weight < 0:
            raise ValueError("modality weights must be non-negative")

        membership = torch.zeros(
            (len(disease_to_prototypes), num_prototypes), dtype=torch.bool
        )
        for disease_index, indices in enumerate(disease_to_prototypes):
            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError("every disease must map to a non-empty 1D index tensor")
            indices = indices.to(dtype=torch.long)
            if int(indices.min()) < 0 or int(indices.max()) >= num_prototypes:
                raise ValueError("prototype mapping contains an out-of-range index")
            membership[disease_index, indices] = True

        used = membership.any(dim=0)
        if not used.all():
            raise ValueError("every prototype must belong to at least one disease")

        self.num_diseases = membership.shape[0]
        self.num_prototypes = num_prototypes
        self.negative_ratio = negative_ratio
        self.text_temperature = float(text_temperature)
        self.image_temperature = float(image_temperature)
        self.text_weight = float(text_weight)
        self.image_weight = float(image_weight)
        self.register_buffer("disease_prototype_membership", membership)

    def _validate_common(self, features: Tensor, prototypes: Tensor, labels: Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape (B, d)")
        if prototypes.ndim != 2 or prototypes.shape[0] != self.num_prototypes:
            raise ValueError("prototypes must have shape (A, d) matching the mapping")
        if features.shape[1] != prototypes.shape[1]:
            raise ValueError("features and prototypes must have the same embedding dimension")
        if labels.shape != (features.shape[0], self.num_diseases):
            raise ValueError("labels must have shape (B, C) matching the mapping")
        if labels.device != features.device or prototypes.device != features.device:
            raise ValueError("features, prototypes, and labels must be on the same device")

    def _prototype_masks(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        positive_diseases = labels > 0
        membership = self.disease_prototype_membership.to(dtype=labels.dtype)
        positive = (positive_diseases.to(labels.dtype) @ membership) > 0
        negative_diseases = ~positive_diseases
        negative_candidates = (
            negative_diseases.to(labels.dtype) @ membership
        ) > 0
        negative_candidates &= ~positive
        return positive, negative_candidates

    def modality_loss(
        self,
        features: Tensor,
        prototypes: Tensor,
        labels: Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None = None,
        detach_prototypes: bool = False,
    ) -> Tensor:
        """Compute one modality loss with a full-batch denominator."""

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._validate_common(features, prototypes, labels)
        if detach_prototypes:
            prototypes = prototypes.detach()

        similarities = F.normalize(features, p=2, dim=-1) @ F.normalize(
            prototypes, p=2, dim=-1
        ).transpose(0, 1)
        logits = similarities / temperature
        positive_masks, negative_masks = self._prototype_masks(labels)
        # Preserve a zero-gradient autograd path when the batch has no positives.
        loss_sum = logits.sum() * 0.0

        for sample_index in range(features.shape[0]):
            positive_indices = positive_masks[sample_index].nonzero(as_tuple=False).flatten()
            if positive_indices.numel() == 0:
                continue

            positive_loss = -F.logsigmoid(logits[sample_index, positive_indices]).mean()
            sample_loss = positive_loss

            negative_indices = negative_masks[sample_index].nonzero(as_tuple=False).flatten()
            negative_count = min(
                self.negative_ratio * positive_indices.numel(),
                negative_indices.numel(),
            )
            if negative_count > 0:
                permutation = torch.randperm(
                    negative_indices.numel(),
                    generator=generator,
                    device=negative_indices.device,
                )[:negative_count]
                sampled_negative_indices = negative_indices[permutation]
                negative_loss = -F.logsigmoid(
                    -logits[sample_index, sampled_negative_indices]
                ).mean()
                sample_loss = sample_loss + negative_loss

            loss_sum = loss_sum + sample_loss

        return loss_sum / features.shape[0]

    def forward(
        self,
        text_features: Tensor,
        image_features: Tensor,
        prototypes: Tensor,
        labels: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> PECLLossOutput:
        """Compute text PECL, image PECL, and the weighted total.

        Text prototypes preserve gradients. Image prototypes are detached while
        image features continue to receive gradients.
        """

        text_loss = self.modality_loss(
            text_features,
            prototypes,
            labels,
            temperature=self.text_temperature,
            generator=generator,
            detach_prototypes=False,
        )
        image_loss = self.modality_loss(
            image_features,
            prototypes,
            labels,
            temperature=self.image_temperature,
            generator=generator,
            detach_prototypes=True,
        )
        total = self.text_weight * text_loss + self.image_weight * image_loss
        return PECLLossOutput(total=total, text=text_loss, image=image_loss)

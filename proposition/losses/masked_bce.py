"""Masked proposition-level BCE loss for Stage 1 weak supervision."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from proposition.data.weak_supervision import UNKNOWN_LABEL, WeakSupervisionArtifacts

Modality = Literal["text", "image"]


@dataclass(frozen=True)
class LabelDistributionRow:
    """Training-label counts and class weight for one proposition/modality."""

    proposition_id: str
    modality: Modality
    n_present: int
    n_absent: int
    n_unknown: int
    pos_weight: float
    single_class: bool


@dataclass(frozen=True)
class PosWeightArtifacts:
    """Per-proposition training statistics used by :class:`MaskedBCELoss`."""

    text_pos_weight: Tensor
    image_pos_weight: Tensor
    rows: tuple[LabelDistributionRow, ...]


def build_pos_weight_artifacts(
    artifacts: WeakSupervisionArtifacts,
    *,
    pos_weight_cap: float = 10.0,
    train_split: str = "train",
) -> PosWeightArtifacts:
    """Derive modality-specific weights exclusively from valid training labels.

    Only active-scoring propositions are returned, in the same order as
    ``artifacts.active_indices`` used by the model. A modality/proposition is
    eligible only when it is trainable for that modality. Unknown labels and
    every non-training row are excluded. Single-class targets deliberately
    retain a weight of one so that positive-only propositions cannot silently
    receive zero loss.
    """
    if not np.isfinite(pos_weight_cap) or pos_weight_cap <= 0:
        raise ValueError("pos_weight_cap must be finite and greater than zero")

    train_rows = artifacts.splits == train_split
    specifications: tuple[tuple[Modality, np.ndarray, np.ndarray], ...] = (
        ("text", artifacts.label_text, artifacts.text_trainable),
        ("image", artifacts.label_img, artifacts.image_trainable),
    )
    weights: dict[Modality, list[float]] = {"text": [], "image": []}
    rows: list[LabelDistributionRow] = []
    active_indices = artifacts.active_indices

    for modality, labels, trainable in specifications:
        for column in active_indices.tolist():
            proposition_id = artifacts.proposition_ids[column]
            if trainable[column]:
                values = labels[train_rows, column]
                n_present = int(np.count_nonzero(values == 1))
                n_absent = int(np.count_nonzero(values == 0))
                n_unknown = int(np.count_nonzero(values == UNKNOWN_LABEL))
                single_class = n_present == 0 or n_absent == 0
                weight = 1.0 if single_class else min(n_absent / n_present, pos_weight_cap)
            else:
                n_present = n_absent = n_unknown = 0
                single_class = True
                weight = 1.0
            weights[modality].append(float(weight))
            rows.append(
                LabelDistributionRow(
                    proposition_id=str(proposition_id),
                    modality=modality,
                    n_present=n_present,
                    n_absent=n_absent,
                    n_unknown=n_unknown,
                    pos_weight=float(weight),
                    single_class=single_class,
                )
            )

    return PosWeightArtifacts(
        text_pos_weight=torch.tensor(weights["text"], dtype=torch.float32),
        image_pos_weight=torch.tensor(weights["image"], dtype=torch.float32),
        rows=tuple(rows),
    )


def write_label_distribution(rows: Sequence[LabelDistributionRow], path: str | Path) -> None:
    """Write the required, stable audit table for loss target distributions."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "proposition_id",
                "modality",
                "N_present",
                "N_absent",
                "N_unknown",
                "pos_weight",
                "single_class",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "proposition_id": row.proposition_id,
                    "modality": row.modality,
                    "N_present": row.n_present,
                    "N_absent": row.n_absent,
                    "N_unknown": row.n_unknown,
                    "pos_weight": row.pos_weight,
                    "single_class": row.single_class,
                }
            )


class MaskedBCELoss(nn.Module):
    """Sum independently normalized masked BCE losses for text and image labels.

    Targets must be weak proposition labels (``0``, ``1``, or ``UNKNOWN_LABEL``);
    disease labels are intentionally absent from this interface. Modality masks
    must already encode active and trainable propositions for the current run.
    """

    def __init__(self, text_pos_weight: Tensor, image_pos_weight: Tensor) -> None:
        super().__init__()
        self.register_buffer("text_pos_weight", _validate_pos_weights(text_pos_weight, "text"))
        self.register_buffer("image_pos_weight", _validate_pos_weights(image_pos_weight, "image"))

    def forward(
        self,
        text_logits: Tensor,
        image_logits: Tensor,
        text_targets: Tensor,
        image_targets: Tensor,
        text_modality_mask: Tensor,
        image_modality_mask: Tensor,
    ) -> Tensor:
        """Return Stage-1 loss, with zero contribution for an empty branch."""
        text_loss = self._branch_loss(
            text_logits, text_targets, text_modality_mask, self.text_pos_weight, "text"
        )
        image_loss = self._branch_loss(
            image_logits, image_targets, image_modality_mask, self.image_pos_weight, "image"
        )
        return text_loss + image_loss

    @staticmethod
    def _branch_loss(
        logits: Tensor,
        targets: Tensor,
        modality_mask: Tensor,
        pos_weight: Tensor,
        modality: Modality,
    ) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(f"{modality}_logits must have shape [batch, proposition]")
        if targets.shape != logits.shape:
            raise ValueError(f"{modality}_targets must match {modality}_logits shape")
        if pos_weight.ndim != 1 or pos_weight.numel() != logits.shape[1]:
            raise ValueError(f"{modality} pos_weight must have one value per proposition")

        mask = _expand_modality_mask(modality_mask, logits, modality)
        valid = mask & ((targets == 0) | (targets == 1))
        if not torch.any(valid):
            return logits.sum() * 0.0

        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.to(dtype=logits.dtype),
            pos_weight=pos_weight.to(device=logits.device, dtype=logits.dtype),
            reduction="none",
        )
        return element_loss.masked_select(valid).mean()


def _validate_pos_weights(value: Tensor, modality: Modality) -> Tensor:
    weights = torch.as_tensor(value, dtype=torch.float32)
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError(f"{modality}_pos_weight must be a non-empty one-dimensional tensor")
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError(f"{modality}_pos_weight values must be finite and greater than zero")
    return weights


def _expand_modality_mask(mask: Tensor, logits: Tensor, modality: Modality) -> Tensor:
    if mask.ndim == 1:
        if mask.shape[0] != logits.shape[1]:
            raise ValueError(f"{modality}_modality_mask must have one value per proposition")
        return mask.to(device=logits.device, dtype=torch.bool).unsqueeze(0).expand_as(logits)
    if mask.shape != logits.shape:
        raise ValueError(f"{modality}_modality_mask must be [proposition] or match logits")
    return mask.to(device=logits.device, dtype=torch.bool)

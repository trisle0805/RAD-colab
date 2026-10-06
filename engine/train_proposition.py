"""Training and evaluation core for proposition-based prediction.

The orchestration entry point owns tokenization and encoder execution. This file
combines already encoded tensors, computes the documented objective, and offers
small reusable epoch helpers without changing the RAD baseline engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from factory.pecl_loss import PECLLoss, PECLLossOutput
from models.proposition_model import ModelOutput, PropositionModel


@dataclass
class EncodedBatch:
    """Encoded tensors needed by the proposition model and PECL."""

    memory: Tensor
    query: Tensor
    labels: Tensor
    text_pooled: Tensor
    image_pooled: Tensor
    alignment_prototypes: Tensor
    memory_padding_mask: Tensor | None = None
    label_query: Tensor | None = None


@dataclass
class LossBreakdown:
    """Differentiable objective components for one batch."""

    total: Tensor
    proposition: Tensor
    pecl: Tensor
    pecl_text: Tensor
    pecl_image: Tensor
    label: Tensor | None
    model_output: ModelOutput


@dataclass
class EpochResult:
    """Sample-weighted epoch losses and concatenated predictions."""

    losses: dict[str, float]
    gt: Tensor
    pred_proposition: Tensor
    pred_label: Tensor | None


def compute_objective(
    model: PropositionModel,
    pecl_loss: PECLLoss | None,
    batch: EncodedBatch,
    *,
    lambda_label: float = 1.0,
    beta_pecl: float = 1.0,
    pecl_generator: torch.Generator | None = None,
    return_attention: bool = False,
    query_chunk_size: int | None = None,
) -> LossBreakdown:
    """Run the prediction pathways and compute the Chapter 4 objective."""

    if lambda_label < 0 or beta_pecl < 0:
        raise ValueError("loss coefficients must be non-negative")
    labels = batch.labels.to(dtype=batch.memory.dtype)
    output = model(
        batch.query,
        batch.memory,
        batch.memory_padding_mask,
        label_query=batch.label_query,
        return_attention=return_attention,
        query_chunk_size=query_chunk_size,
    )
    if output.proposition.logits.shape != labels.shape:
        raise ValueError("proposition logits and labels must have identical shape")

    proposition_loss = F.binary_cross_entropy_with_logits(
        output.proposition.logits, labels
    )

    label_loss: Tensor | None = None
    if output.label is not None:
        if output.label.logits.shape != labels.shape:
            raise ValueError("label-branch logits and labels must have identical shape")
        label_loss = F.binary_cross_entropy_with_logits(output.label.logits, labels)
    elif lambda_label != 0:
        raise ValueError("lambda_label must be zero when the label branch is disabled")

    if pecl_loss is None:
        pecl_output = PECLLossOutput(
            total=proposition_loss.new_zeros(()),
            text=proposition_loss.new_zeros(()),
            image=proposition_loss.new_zeros(()),
        )
    else:
        pecl_output = pecl_loss(
            batch.text_pooled,
            batch.image_pooled,
            batch.alignment_prototypes,
            labels,
            generator=pecl_generator,
        )

    total = proposition_loss + beta_pecl * pecl_output.total
    if label_loss is not None:
        total = total + lambda_label * label_loss

    return LossBreakdown(
        total=total,
        proposition=proposition_loss,
        pecl=pecl_output.total,
        pecl_text=pecl_output.text,
        pecl_image=pecl_output.image,
        label=label_loss,
        model_output=output,
    )


def _accumulate_loss(
    sums: dict[str, float], breakdown: LossBreakdown, batch_size: int
) -> None:
    values = {
        "total": breakdown.total,
        "proposition": breakdown.proposition,
        "pecl": breakdown.pecl,
        "pecl_text": breakdown.pecl_text,
        "pecl_image": breakdown.pecl_image,
    }
    if breakdown.label is not None:
        values["label"] = breakdown.label
    for name, value in values.items():
        sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size


def _finalize_epoch(
    sums: Mapping[str, float],
    sample_count: int,
    ground_truth: list[Tensor],
    proposition_predictions: list[Tensor],
    label_predictions: list[Tensor],
) -> EpochResult:
    if sample_count == 0:
        raise ValueError("data loader produced no samples")
    return EpochResult(
        losses={name: value / sample_count for name, value in sums.items()},
        gt=torch.cat(ground_truth, dim=0),
        pred_proposition=torch.cat(proposition_predictions, dim=0),
        pred_label=torch.cat(label_predictions, dim=0) if label_predictions else None,
    )


def train_one_epoch(
    model: PropositionModel,
    pecl_loss: PECLLoss | None,
    encoded_batches: Iterable[EncodedBatch],
    optimizer: torch.optim.Optimizer,
    *,
    accumulation_steps: int = 1,
    lambda_label: float = 1.0,
    beta_pecl: float = 1.0,
    pecl_generator: torch.Generator | None = None,
    query_chunk_size: int | None = None,
) -> EpochResult:
    """Train for one epoch over encoded batches with gradient accumulation."""

    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    sample_count = 0
    gt: list[Tensor] = []
    pred_proposition: list[Tensor] = []
    pred_label: list[Tensor] = []
    pending_steps = 0

    for batch in encoded_batches:
        breakdown = compute_objective(
            model,
            pecl_loss,
            batch,
            lambda_label=lambda_label,
            beta_pecl=beta_pecl,
            pecl_generator=pecl_generator,
            query_chunk_size=query_chunk_size,
        )
        (breakdown.total / accumulation_steps).backward()
        pending_steps += 1
        if pending_steps == accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending_steps = 0

        batch_size = batch.labels.shape[0]
        sample_count += batch_size
        _accumulate_loss(sums, breakdown, batch_size)
        gt.append(batch.labels.detach().cpu())
        pred_proposition.append(
            breakdown.model_output.proposition.probabilities.detach().cpu()
        )
        if breakdown.model_output.label is not None:
            pred_label.append(breakdown.model_output.label.probabilities.detach().cpu())

    if pending_steps:
        # Correct the final short accumulation window to preserve mean gradients.
        correction = accumulation_steps / pending_steps
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return _finalize_epoch(sums, sample_count, gt, pred_proposition, pred_label)


@torch.no_grad()
def evaluate(
    model: PropositionModel,
    pecl_loss: PECLLoss | None,
    encoded_batches: Iterable[EncodedBatch],
    *,
    lambda_label: float = 1.0,
    beta_pecl: float = 1.0,
    pecl_generator: torch.Generator | None = None,
    return_attention: bool = False,
    query_chunk_size: int | None = None,
) -> EpochResult:
    """Evaluate without changing model parameters or mixing prediction branches."""

    model.eval()
    sums: dict[str, float] = {}
    sample_count = 0
    gt: list[Tensor] = []
    pred_proposition: list[Tensor] = []
    pred_label: list[Tensor] = []

    for batch in encoded_batches:
        breakdown = compute_objective(
            model,
            pecl_loss,
            batch,
            lambda_label=lambda_label,
            beta_pecl=beta_pecl,
            pecl_generator=pecl_generator,
            return_attention=return_attention,
            query_chunk_size=query_chunk_size,
        )
        batch_size = batch.labels.shape[0]
        sample_count += batch_size
        _accumulate_loss(sums, breakdown, batch_size)
        gt.append(batch.labels.detach().cpu())
        pred_proposition.append(
            breakdown.model_output.proposition.probabilities.detach().cpu()
        )
        if breakdown.model_output.label is not None:
            pred_label.append(breakdown.model_output.label.probabilities.detach().cpu())

    return _finalize_epoch(sums, sample_count, gt, pred_proposition, pred_label)

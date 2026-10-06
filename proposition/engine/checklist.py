"""Detailed test-time evidence artifacts for proposition-model inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from proposition.data.knowledge_base import PropositionKnowledgeBase
from proposition.engine.trainer import EncodedBatch, compute_objective
from proposition.losses.pecl import PECLLoss
from proposition.models.proposition_model import PropositionModel


@dataclass
class ChecklistResult:
    """Evidence arrays collected from a deterministic test inference pass."""

    gt: Tensor
    pred_proposition: Tensor
    pred_label: Tensor | None
    support: Tensor
    compatibility: Tensor
    disease_scores: Tensor
    attention_top_indices: Tensor
    attention_top_weights: Tensor
    attention_entropy: Tensor
    full_attention_sample_indices: Tensor
    full_attention: Tensor
    image_token_count: Tensor
    caption_input_ids: Tensor
    caption_attention_mask: Tensor


def _atomic_json(path: Path, payload: Mapping[str, Any] | list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def attention_entropy(attention: Tensor) -> Tensor:
    """Return normalized entropy per sample/query for an attention distribution."""

    if attention.ndim != 3 or attention.shape[-1] <= 1:
        raise ValueError("attention must have shape (N, Q, L) with L > 1")
    safe = attention.clamp_min(torch.finfo(attention.dtype).tiny)
    entropy = -(safe * safe.log()).sum(dim=-1)
    return entropy / float(np.log(attention.shape[-1]))


@torch.no_grad()
def collect_checklist(
    model: PropositionModel,
    pecl_loss: PECLLoss | None,
    encoded_batches: Iterable[EncodedBatch],
    *,
    lambda_label: float = 1.0,
    beta_pecl: float = 1.0,
    pecl_generator: torch.Generator | None = None,
    query_chunk_size: int | None = None,
    top_evidence: int = 10,
    full_attention_sample_indices: Tensor | None = None,
) -> ChecklistResult:
    """Collect every test sample's prediction and explicit main-path attention."""

    model.eval()
    collected: dict[str, list[Tensor]] = {
        "gt": [],
        "pred_proposition": [],
        "support": [],
        "compatibility": [],
        "disease_scores": [],
        "attention_top_indices": [],
        "attention_top_weights": [],
        "attention_entropy": [],
        "image_token_count": [],
        "caption_input_ids": [],
        "caption_attention_mask": [],
    }
    full_attention: list[Tensor] = []
    full_attention_indices: list[Tensor] = []
    label_predictions: list[Tensor] = []
    sample_offset = 0

    for batch in encoded_batches:
        breakdown = compute_objective(
            model,
            pecl_loss,
            batch,
            lambda_label=lambda_label,
            beta_pecl=beta_pecl,
            pecl_generator=pecl_generator,
            return_attention=True,
            query_chunk_size=query_chunk_size,
        )
        output = breakdown.model_output.proposition
        if output.attention is None:
            raise RuntimeError("main proposition attention was requested but not returned")
        if (
            batch.image_token_count is None
            or batch.caption_input_ids is None
            or batch.caption_attention_mask is None
        ):
            raise ValueError("checklist collection requires pipeline token metadata")

        collected["gt"].append(batch.labels.detach().cpu())
        collected["pred_proposition"].append(output.probabilities.detach().cpu())
        collected["support"].append(output.support.detach().cpu())
        collected["compatibility"].append(output.compatibility.detach().cpu())
        attention = output.attention
        k = min(top_evidence, attention.shape[-1])
        top_weights, top_indices = torch.topk(attention, k=k, dim=-1)
        collected["disease_scores"].append(output.disease_scores.detach().cpu())
        collected["attention_top_indices"].append(top_indices.to(torch.int32).detach().cpu())
        collected["attention_top_weights"].append(top_weights.detach().cpu())
        collected["attention_entropy"].append(attention_entropy(attention).detach().cpu())
        if full_attention_sample_indices is not None:
            batch_indices = torch.arange(
                sample_offset, sample_offset + attention.shape[0], device=attention.device
            )
            selected = torch.isin(batch_indices, full_attention_sample_indices.to(attention.device))
            if selected.any():
                full_attention.append(attention[selected].to(torch.float16).detach().cpu())
                full_attention_indices.append(batch_indices[selected].to(torch.int64).detach().cpu())
        sample_offset += attention.shape[0]
        collected["image_token_count"].append(batch.image_token_count.detach().cpu())
        collected["caption_input_ids"].append(batch.caption_input_ids.detach().cpu())
        collected["caption_attention_mask"].append(batch.caption_attention_mask.detach().cpu())
        if breakdown.model_output.label is not None:
            label_predictions.append(breakdown.model_output.label.probabilities.detach().cpu())

    if not collected["gt"]:
        raise ValueError("encoded_batches produced no test samples")
    return ChecklistResult(
        gt=torch.cat(collected["gt"], dim=0),
        pred_proposition=torch.cat(collected["pred_proposition"], dim=0),
        pred_label=torch.cat(label_predictions, dim=0) if label_predictions else None,
        support=torch.cat(collected["support"], dim=0),
        compatibility=torch.cat(collected["compatibility"], dim=0),
        disease_scores=torch.cat(collected["disease_scores"], dim=0),
        attention_top_indices=torch.cat(collected["attention_top_indices"], dim=0),
        attention_top_weights=torch.cat(collected["attention_top_weights"], dim=0),
        attention_entropy=torch.cat(collected["attention_entropy"], dim=0),
        full_attention_sample_indices=(
            torch.cat(full_attention_indices, dim=0)
            if full_attention_indices
            else torch.empty(0, dtype=torch.int64)
        ),
        full_attention=(
            torch.cat(full_attention, dim=0)
            if full_attention
            else torch.empty((0, 0, 0), dtype=torch.float16)
        ),
        image_token_count=torch.cat(collected["image_token_count"], dim=0),
        caption_input_ids=torch.cat(collected["caption_input_ids"], dim=0),
        caption_attention_mask=torch.cat(collected["caption_attention_mask"], dim=0),
    )


def save_checklist_artifacts(
    result: ChecklistResult,
    output_dir: str | Path,
    kb: PropositionKnowledgeBase,
    *,
    image_paths: Sequence[str],
    top_evidence: int = 10,
    aggregation_weights: Tensor | None = None,
) -> dict[str, str]:
    """Save compact UI-ready test evidence and its machine-readable schema."""

    if top_evidence <= 0:
        raise ValueError("top_evidence must be positive")
    output = Path(output_dir) / "checklist"
    sample_count, query_count = result.support.shape
    memory_length = result.caption_input_ids.shape[1] + int(result.image_token_count[0])
    if len(image_paths) != sample_count:
        raise ValueError("image_paths length must match checklist sample count")
    if result.image_token_count.shape != (sample_count,):
        raise ValueError("image_token_count must have one value per sample")
    if not torch.all(result.image_token_count == result.image_token_count[0]):
        raise ValueError("all SkinCAP samples must have the same image patch-token count")
    image_token_count = int(result.image_token_count[0])
    caption_length = result.caption_input_ids.shape[1]
    if image_token_count + caption_length != memory_length:
        raise ValueError("image and caption token lengths do not reconstruct attention memory")
    if result.pred_proposition.shape[0] != sample_count:
        raise ValueError("prediction sample count does not match attention")
    is_proposition_level = query_count == kb.num_propositions
    if not is_proposition_level and query_count != kb.num_diseases:
        raise ValueError("attention query count matches neither propositions nor diseases")

    top_indices = result.attention_top_indices.numpy().astype(np.int32, copy=False)
    top_weights = result.attention_top_weights.numpy().astype(np.float32, copy=False)
    entropy = result.attention_entropy.numpy().astype(np.float32, copy=False)

    arrays: dict[str, Any] = {
        "gt": result.gt.numpy().astype(np.float32, copy=False),
        "pred_proposition": result.pred_proposition.numpy().astype(np.float32, copy=False),
        "support": result.support.numpy().astype(np.float32, copy=False),
        "compatibility": result.compatibility.numpy().astype(np.float32, copy=False),
        "disease_scores": result.disease_scores.numpy().astype(np.float32, copy=False),
        "attention_entropy": entropy,
        "attention_top_indices": top_indices,
        "attention_top_weights": top_weights,
        "image_token_count": result.image_token_count.numpy().astype(np.int32, copy=False),
        "caption_input_ids": result.caption_input_ids.numpy().astype(np.int64, copy=False),
        "caption_attention_mask": result.caption_attention_mask.numpy().astype(np.int8, copy=False),
        "image_paths": np.asarray(image_paths, dtype=str),
        "full_attention_sample_indices": result.full_attention_sample_indices.numpy().astype(np.int64, copy=False),
        "full_attention": result.full_attention.numpy().astype(np.float16, copy=False),
    }
    if result.pred_label is not None:
        arrays["pred_label"] = result.pred_label.numpy().astype(np.float32, copy=False)
    if aggregation_weights is not None:
        arrays["aggregation_weights"] = aggregation_weights.detach().cpu().numpy().astype(np.float32, copy=False)

    artifact = output / (
        "test_proposition_evidence.npz" if is_proposition_level else "test_disease_level_evidence.npz"
    )
    _atomic_npz(artifact, **arrays)
    proposition_index_path = output / "proposition_index.json"
    if is_proposition_level:
        kb.write_proposition_index(proposition_index_path)
    disease_index = [
        {"disease_id": disease, "disease_index": index}
        for index, disease in enumerate(kb.disease_ids)
    ]
    _atomic_json(output / "disease_index.json", disease_index)
    schema = {
        "artifact": artifact.name,
        "sample_count": sample_count,
        "query_level": "proposition" if is_proposition_level else "disease",
        "query_count": query_count,
        "memory_length": memory_length,
        "image_token_count": image_token_count,
        "caption_token_count": caption_length,
        "top_evidence_per_query": top_indices.shape[-1],
        "full_attention_sample_count": int(result.full_attention_sample_indices.numel()),
        "attention_axes": ["sample", "query", "memory_token"],
        "memory_layout": {
            "image": [0, image_token_count],
            "caption": [image_token_count, memory_length],
        },
        "rendering": {
            "image_patch_grid": int(round(image_token_count ** 0.5)),
            "image_patch_grid_is_square": int(round(image_token_count ** 0.5)) ** 2 == image_token_count,
            "caption_tokenizer": "the tokenizer selected for the run",
        },
        "fields": {
            "gt": "one-hot ground-truth disease labels (N, C)",
            "pred_proposition": "calibrated disease probabilities (N, C)",
            "support": "raw query support s (N, Q)",
            "compatibility": "polarity-aware proposition compatibility or disease support (N, Q)",
            "disease_scores": "pre-calibration disease scores z (N, C)",
            "full_attention": "full cosine cross-attention for selected samples (K, J, L), float16",
            "full_attention_sample_indices": "dataset sample indices corresponding to full_attention (K)",
            "attention_entropy": "normalized entropy per proposition (N, J)",
            "attention_top_indices": "top memory positions per proposition (N, J, K)",
            "attention_top_weights": "weights for attention_top_indices (N, J, K)",
            "caption_input_ids": "caption token IDs for tokenizer decoding (N, T)",
            "caption_attention_mask": "caption valid-token mask (N, T)",
        },
    }
    schema_path = output / (
        "test_proposition_evidence.schema.json" if is_proposition_level else "test_disease_level_evidence.schema.json"
    )
    _atomic_json(schema_path, schema)
    paths = {
        "evidence": str(artifact),
        "schema": str(schema_path),
        "disease_index": str(output / "disease_index.json"),
    }
    if is_proposition_level:
        paths["proposition_index"] = str(proposition_index_path)
    return paths

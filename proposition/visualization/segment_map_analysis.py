"""Summarize proposition attention maps over the image patches only.

Works with proposition-level P1/P2 artifacts and the segment-query artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _proposition_diseases(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError("proposition_index.json must contain a list")
    return np.asarray([entry["disease_index"] for entry in entries], dtype=np.int64)


def analyze(evidence_path: str | Path) -> dict[str, Any]:
    """Return image-only map similarity metrics and write them beside evidence."""

    evidence_path = Path(evidence_path)
    checklist = evidence_path.parent
    proposition_index = checklist / "proposition_index.json"
    if not proposition_index.is_file():
        raise FileNotFoundError(f"proposition index not found: {proposition_index}")

    with np.load(evidence_path, allow_pickle=False) as evidence:
        full_attention = evidence["full_attention"].astype(np.float64)
        selected_indices = evidence["full_attention_sample_indices"].astype(np.int64)
        image_counts = evidence["image_token_count"].astype(np.int64)
        predictions = evidence["pred_proposition"].astype(np.float64)
    if full_attention.ndim != 3:
        raise ValueError("full_attention must have shape (K, Q, L)")
    if selected_indices.shape != (full_attention.shape[0],):
        raise ValueError("full_attention_sample_indices must align with full_attention")
    if full_attention.shape[0] == 0:
        raise ValueError("the evidence artifact contains no full-attention samples")
    if not np.all(image_counts == image_counts[0]):
        raise ValueError("all samples must have the same image token count")
    image_count = int(image_counts[0])
    if int(round(image_count ** 0.5)) ** 2 != image_count:
        raise ValueError("image token count must form a square patch grid")
    if np.any(selected_indices < 0) or np.any(selected_indices >= predictions.shape[0]):
        raise ValueError("full-attention sample index is outside prediction rows")

    proposition_disease = _proposition_diseases(proposition_index)
    if proposition_disease.shape[0] != full_attention.shape[1]:
        raise ValueError(
            "full-attention query count must equal propositions; disease-level artifacts are unsupported"
        )
    image_maps = full_attention[:, :, :image_count]
    normalized_maps = image_maps / np.clip(image_maps.sum(axis=-1, keepdims=True), 1e-12, None)

    within_case: list[float] = []
    used_props: set[int] = set()
    for local_index, sample_index in enumerate(selected_indices):
        top_disease = int(np.argmax(predictions[sample_index]))
        propositions = np.flatnonzero(proposition_disease == top_disease)
        used_props.update(int(proposition) for proposition in propositions)
        for left, right in combinations(propositions, 2):
            correlation = _pearson(normalized_maps[local_index, left], normalized_maps[local_index, right])
            if correlation is not None:
                within_case.append(correlation)

    across_cases: list[float] = []
    for proposition in range(full_attention.shape[1]):
        for left, right in combinations(range(full_attention.shape[0]), 2):
            correlation = _pearson(normalized_maps[left, proposition], normalized_maps[right, proposition])
            if correlation is not None:
                across_cases.append(correlation)
                used_props.add(proposition)

    max_over_mean = normalized_maps.max(axis=-1) / normalized_maps.mean(axis=-1)
    metrics: dict[str, Any] = {
        "evidence": str(evidence_path),
        "within_case_same_disease_corr": float(np.mean(within_case)) if within_case else None,
        "same_prop_across_cases_corr": float(np.mean(across_cases)) if across_cases else None,
        "image_max_over_mean": float(np.median(max_over_mean)),
        "image_mass_share": float(image_maps.sum(axis=-1).mean()),
        "num_cases": int(full_attention.shape[0]),
        "num_props_used": len(used_props),
    }
    _atomic_json(checklist / "segment_map_analysis.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

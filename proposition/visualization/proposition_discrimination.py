"""Measure whether proposition supports discriminate individual disease findings.

This utility only reads saved proposition evidence and its adjacent proposition
index, so it can compare P1 and P2 runs without loading a checkpoint or model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


    def _load_proposition_index(path: Path, query_count: int) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    with path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, list) or len(index) != query_count:
        raise ValueError(
            "proposition_index.json must be a list aligned with the support axis "
            f"(expected {query_count}, found {len(index) if isinstance(index, list) else 'non-list'})"
        )
    try:
            diseases = np.asarray([int(entry["disease_index"]) for entry in index], dtype=np.int64)
            polarities = np.asarray([int(entry["polarity"]) for entry in index], dtype=np.int8)
            if not np.all((polarities == 1) | (polarities == -1)):
                raise ValueError("each proposition index polarity must be 1 or -1")
            disease_ids = {int(entry["disease_index"]): str(entry["disease_id"]) for entry in index}
            return diseases, polarities, disease_ids
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("each proposition index entry needs an integer disease_index") from exc


def _share_above(values: np.ndarray, threshold: float) -> float | None:
    return float(np.mean(values > threshold)) if values.size else None


def compute_proposition_discrimination_from_evidence(
    evidence: Mapping[str, np.ndarray],
    proposition_diseases: Sequence[int] | np.ndarray,
    proposition_polarities: Sequence[int] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute saved-evidence metrics, treating polarity as ✓ (positive) or ✕."""

    required = {"support", "pred_proposition", "compatibility"}
    missing = required.difference(evidence)
    if missing:
        raise ValueError(f"evidence is missing required arrays: {sorted(missing)}")
    support = np.asarray(evidence["support"])
    pred = np.asarray(evidence["pred_proposition"])
    diseases = np.asarray(proposition_diseases, dtype=np.int64)
        polarities = np.asarray(proposition_polarities, dtype=np.int8)
    compatibility = np.asarray(evidence["compatibility"])
    if support.ndim != 2 or compatibility.shape != support.shape:
        raise ValueError("support and compatibility must have matching shape (N, J)")
    if pred.ndim != 2 or pred.shape[0] != support.shape[0]:
        raise ValueError("pred_proposition must have shape (N, C) matching support")
    if diseases.shape != (support.shape[1],):
        raise ValueError("proposition disease indices must align with the support axis")
        if polarities.shape != (support.shape[1],) or not np.all((polarities == 1) | (polarities == -1)):
            raise ValueError("proposition polarities must align with support and be 1 or -1")
    if support.shape[0] == 0:
        raise ValueError("evidence contains no test cases")
    if diseases.min() < 0 or diseases.max() >= pred.shape[1]:
        raise ValueError("proposition disease indices are outside pred_proposition class range")

    top1 = np.argmax(pred, axis=1)
    ranking = np.argsort(-pred, axis=1, kind="stable")
    top2 = ranking[:, 1] if pred.shape[1] >= 2 else top1
    top1_positive: list[np.ndarray] = []
    top1_negative: list[np.ndarray] = []
    other: list[np.ndarray] = []
    within_stds: list[float] = []
    for case_index, disease_index in enumerate(top1):
        own = diseases == disease_index
        own_support = support[case_index, own]
        if own_support.size:
            within_stds.append(float(np.std(own_support)))
                positive = polarities[own] == 1
            top1_positive.append(own_support[positive])
            top1_negative.append(own_support[~positive])
        other.append(support[case_index, ~own])

    top2_values, top2_counts = np.unique(top2, return_counts=True)
    most_common_position = int(np.argmax(top2_counts))
    most_common_disease = int(top2_values[most_common_position])
    return {
        "top1_pos_share_s_gt_0.5": _share_above(
            np.concatenate(top1_positive) if top1_positive else np.empty(0), threshold
        ),
        "top1_neg_share_s_gt_0.5": _share_above(
            np.concatenate(top1_negative) if top1_negative else np.empty(0), threshold
        ),
        "top1_within_disease_std_s": float(np.mean(within_stds)) if within_stds else None,
        "other_share_s_gt_0.5": _share_above(
            np.concatenate(other) if other else np.empty(0), threshold
        ),
        "top2_most_common_disease": {
            "disease_index": most_common_disease,
            "disease_id": None,
            "num_cases": int(top2_counts[most_common_position]),
        },
        "num_cases": int(support.shape[0]),
    }


def run(evidence_path: str | Path) -> Path:
    evidence_path = Path(evidence_path)
    index_path = evidence_path.with_name("proposition_index.json")
    if not index_path.is_file():
        raise FileNotFoundError(f"missing proposition index next to evidence: {index_path}")
    with np.load(evidence_path, allow_pickle=False) as loaded:
        evidence = {name: loaded[name] for name in loaded.files}
        diseases, polarities, disease_ids = _load_proposition_index(
            index_path, np.asarray(evidence["support"]).shape[1]
        )
        result = compute_proposition_discrimination_from_evidence(evidence, diseases, polarities)
    result["top2_most_common_disease"]["disease_id"] = disease_ids[
        result["top2_most_common_disease"]["disease_index"]
    ]
    output_path = evidence_path.with_name("proposition_discrimination.json")
    _atomic_json(output_path, result)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, help="Path to test_proposition_evidence.npz")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    print(run(parsed.evidence))

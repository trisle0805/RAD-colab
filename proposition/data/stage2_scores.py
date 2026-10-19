"""Validated, encoder-free inputs for Stage-2 diagnosis calibration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

Stage2RoutingMode = Literal[
    "image_morphology",
    "full_fused",
    "full_image_morphology_text_other",
]

_ROUTING_MODES = frozenset((
    "image_morphology", "full_fused", "full_image_morphology_text_other",
))


@dataclass(frozen=True)
class Stage2ScoreArtifacts:
    case_ids: np.ndarray
    proposition_ids: np.ndarray
    score_image: np.ndarray
    score_text: np.ndarray
    score_fused: np.ndarray | None
    image_trainable: np.ndarray
    text_trainable: np.ndarray
    active_scoring: np.ndarray


@dataclass(frozen=True)
class Stage2Proposition:
    proposition_id: str
    disease_id: str
    category: str
    polarity: int
    image_trainable: bool
    text_trainable: bool
    active_scoring: bool


@dataclass(frozen=True)
class DiseaseTargets:
    case_ids: np.ndarray
    disease_ids: tuple[str, ...]
    targets: np.ndarray


@dataclass(frozen=True)
class Stage2Dataset:
    case_ids: np.ndarray
    support_scores: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class RoutedStage2Scores:
    support_scores: np.ndarray
    proposition_mask: np.ndarray
    score_source: np.ndarray


def load_stage2_score_artifact(path: str | Path) -> Stage2ScoreArtifacts:
    """Load a frozen Stage-1 score export without reordering any identifiers."""
    source = Path(path)
    required = {
        "case_id", "proposition_id", "score_image", "score_text",
        "image_trainable", "text_trainable", "active_scoring",
    }
    try:
        with np.load(source, allow_pickle=False) as payload:
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"{source}: missing fields {sorted(missing)!r}")
            values = {name: np.asarray(payload[name]) for name in required}
            fused = np.asarray(payload["score_fused"]) if "score_fused" in payload.files else None
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to read Stage-2 score artifact {source}: {error}") from error

    case_ids = _string_vector(values["case_id"], "case_id", source)
    proposition_ids = _string_vector(values["proposition_id"], "proposition_id", source)
    _unique(case_ids, "case_id", source)
    _unique(proposition_ids, "proposition_id", source)
    shape = (len(case_ids), len(proposition_ids))
    image = _score_matrix(values["score_image"], "score_image", shape, source)
    text = _score_matrix(values["score_text"], "score_text", shape, source)
    if fused is not None:
        fused = _score_matrix(fused, "score_fused", shape, source)
    image_trainable = _binary_vector(values["image_trainable"], "image_trainable", len(proposition_ids), source)
    text_trainable = _binary_vector(values["text_trainable"], "text_trainable", len(proposition_ids), source)
    active = _binary_vector(values["active_scoring"], "active_scoring", len(proposition_ids), source)
    return Stage2ScoreArtifacts(case_ids, proposition_ids, image, text, fused, image_trainable, text_trainable, active)


def validate_stage2_splits(train: Stage2ScoreArtifacts, val: Stage2ScoreArtifacts, test: Stage2ScoreArtifacts) -> None:
    """Require identical proposition order/masks and disjoint case populations."""
    for label, artifact in (("validation", val), ("test", test)):
        if not np.array_equal(train.proposition_ids, artifact.proposition_ids):
            raise ValueError(f"{label} proposition IDs/order differ from train; reordering is forbidden")
        for field in ("image_trainable", "text_trainable", "active_scoring"):
            if not np.array_equal(getattr(train, field), getattr(artifact, field)):
                raise ValueError(f"{label} {field} differs from train")
    groups = (("train", train.case_ids), ("val", val.case_ids), ("test", test.case_ids))
    for index, (left_name, left_ids) in enumerate(groups):
        for right_name, right_ids in groups[index + 1:]:
            overlap = set(left_ids.tolist()).intersection(right_ids.tolist())
            if overlap:
                raise ValueError(f"case IDs overlap between {left_name} and {right_name}: {sorted(overlap)[:5]!r}")


def load_stage2_propositions(active_csv: str | Path, expected_proposition_ids: Sequence[str]) -> tuple[Stage2Proposition, ...]:
    """Load proposition metadata and explicitly align it to frozen NPZ order."""
    source = Path(active_csv)
    required = {"proposition_id", "disease_id", "category", "polarity", "text_trainable", "image_trainable", "active_scoring"}
    rows: dict[str, Stage2Proposition] = {}
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or required.difference(reader.fieldnames):
                raise ValueError(f"{source}: missing required columns {sorted(required.difference(reader.fieldnames or []))!r}")
            for number, row in enumerate(reader, start=2):
                proposition_id = (row.get("proposition_id") or "").strip()
                disease_id = (row.get("disease_id") or "").strip()
                category = (row.get("category") or "").strip()
                if not proposition_id or not disease_id or not category:
                    raise ValueError(f"{source}:{number}: proposition_id, disease_id, and category must be non-empty")
                if proposition_id in rows:
                    raise ValueError(f"{source}: duplicate proposition_id {proposition_id!r}")
                try:
                    polarity = int(row["polarity"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{source}:{number}: polarity must be -1 or 1") from error
                if polarity not in (-1, 1):
                    raise ValueError(f"{source}:{number}: polarity must be -1 or 1")
                rows[proposition_id] = Stage2Proposition(
                    proposition_id, disease_id, category, polarity,
                    _csv_bool(row["image_trainable"], source, number),
                    _csv_bool(row["text_trainable"], source, number),
                    _csv_bool(row["active_scoring"], source, number),
                )
    except FileNotFoundError:
        raise
    expected = tuple(str(value) for value in expected_proposition_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("expected proposition IDs contain duplicates")
    missing = set(expected).difference(rows)
    active_extra = sorted(
        identifier
        for identifier in set(rows).difference(expected)
        if rows[identifier].active_scoring
    )
    if missing or active_extra:
        raise ValueError(
            f"{source}: proposition IDs differ; missing={sorted(missing)[:5]!r}, "
            f"active_extra={active_extra[:5]!r}"
        )
    return tuple(rows[identifier] for identifier in expected)


def load_stage2_disease_targets(csv_path: str | Path, expected_case_ids: Sequence[str]) -> DiseaseTargets:
    """Load binary multi-label disease targets and align them by case ID."""
    source = Path(csv_path)
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{source}: CSV must include a header")
            fields = tuple(reader.fieldnames)
            label_columns = tuple(sorted(column for column in fields if column.startswith("label_")))
            if not label_columns:
                raise ValueError(f"{source}: no label_* disease columns")
            image_column = "image" if "image" in fields else fields[0]
            rows: dict[str, np.ndarray] = {}
            for number, row in enumerate(reader, start=2):
                case_id = (row.get("case_id") or "").strip() if "case_id" in fields else Path((row.get(image_column) or "").strip()).name
                if not case_id:
                    raise ValueError(f"{source}:{number}: case_id must not be empty")
                if case_id in rows:
                    raise ValueError(f"{source}: duplicate case_id {case_id!r}")
                try:
                    labels = np.asarray([int(row[column]) for column in label_columns], dtype=np.int8)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{source}:{number}: label values must be integers") from error
                if not np.all(np.isin(labels, (0, 1))):
                    raise ValueError(f"{source}:{number}: label values must be binary")
                rows[case_id] = labels
    except FileNotFoundError:
        raise
    requested = np.asarray([str(case_id) for case_id in expected_case_ids], dtype=str)
    _unique(requested, "expected case_id", source)
    missing = [case_id for case_id in requested.tolist() if case_id not in rows]
    if missing:
        raise ValueError(f"{source}: missing requested case IDs {missing[:5]!r}")
    disease_ids = tuple(column.removeprefix("label_") for column in label_columns)
    if len(set(disease_ids)) != len(disease_ids) or any(not identifier for identifier in disease_ids):
        raise ValueError(f"{source}: invalid disease label column names")
    return DiseaseTargets(requested, disease_ids, np.stack([rows[case_id] for case_id in requested], axis=0))


def build_stage2_support(artifacts: Stage2ScoreArtifacts, propositions: Sequence[Stage2Proposition], mode: Stage2RoutingMode) -> RoutedStage2Scores:
    """Route frozen support scores according to one predefined Stage-2 ablation."""
    if mode not in _ROUTING_MODES:
        raise ValueError(f"unsupported Stage-2 routing mode {mode!r}")
    if len(propositions) != len(artifacts.proposition_ids):
        raise ValueError("proposition metadata length differs from score artifact")
    metadata_ids = np.asarray([item.proposition_id for item in propositions], dtype=str)
    if not np.array_equal(metadata_ids, artifacts.proposition_ids):
        raise ValueError("proposition metadata order differs from score artifact")
    for name in ("image_trainable", "text_trainable", "active_scoring"):
        expected = np.asarray([getattr(item, name) for item in propositions], dtype=bool)
        if not np.array_equal(expected, getattr(artifacts, name)):
            raise ValueError(f"metadata {name} disagrees with score artifact")
    active = artifacts.active_scoring
    image = artifacts.image_trainable
    text = artifacts.text_trainable
    morphology = np.asarray([item.category == "morphology" for item in propositions], dtype=bool)
    sources = np.full(len(propositions), "excluded", dtype="U8")
    support = np.full(artifacts.score_image.shape, np.nan, dtype=np.float32)
    if mode == "image_morphology":
        mask = active & morphology & image
        support[:, mask] = artifacts.score_image[:, mask]
        sources[mask] = "image"
    elif mode == "full_fused":
        both = active & image & text
        text_only = active & ~image & text
        image_only = active & image & ~text
        mask = both | text_only | image_only
        support[:, both] = (artifacts.score_image[:, both] + artifacts.score_text[:, both]) / 2.0
        support[:, text_only] = artifacts.score_text[:, text_only]
        support[:, image_only] = artifacts.score_image[:, image_only]
        sources[both], sources[text_only], sources[image_only] = "fused", "text", "image"
        if artifacts.score_fused is not None and not np.allclose(support[:, mask], artifacts.score_fused[:, mask], rtol=1e-5, atol=1e-6):
            raise ValueError("stored score_fused does not match reconstructed Stage-1 fusion")
    else:
        morph_image = active & morphology & image
        text_preferred = active & ~morph_image & text
        image_fallback = active & ~morph_image & ~text & image
        mask = morph_image | text_preferred | image_fallback
        support[:, morph_image] = artifacts.score_image[:, morph_image]
        support[:, text_preferred] = artifacts.score_text[:, text_preferred]
        support[:, image_fallback] = artifacts.score_image[:, image_fallback]
        sources[morph_image], sources[text_preferred], sources[image_fallback] = "image", "text", "image"
    return RoutedStage2Scores(support, mask, sources)


def _string_vector(value: np.ndarray, field: str, source: Path) -> np.ndarray:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{source}: {field} must be a one-dimensional string array")
    result = np.asarray(value, dtype=str)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{source}: {field} contains an empty identifier")
    return result


def _unique(value: np.ndarray, field: str, source: Path) -> None:
    unique, counts = np.unique(value, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"{source}: duplicate {field} values {unique[counts > 1][:5].tolist()!r}")


def _score_matrix(value: np.ndarray, field: str, shape: tuple[int, int], source: Path) -> np.ndarray:
    if value.ndim != 2 or value.shape != shape or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{source}: {field} must be numeric with shape {shape}, got {value.shape}")
    result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)) or np.any(result < -1e-6) or np.any(result > 1.0 + 1e-6):
        raise ValueError(f"{source}: {field} must contain finite values in [0, 1]")
    return np.clip(result, 0.0, 1.0)


def _binary_vector(value: np.ndarray, field: str, expected: int, source: Path) -> np.ndarray:
    if value.ndim != 1 or len(value) != expected or not np.all(np.isin(value, (0, 1, False, True))):
        raise ValueError(f"{source}: {field} must be a binary vector of length {expected}")
    return np.asarray(value, dtype=bool)


def _csv_bool(value: str | None, source: Path, row_number: int) -> bool:
    if value is None or value.strip() not in {"0", "1"}:
        raise ValueError(f"{source}:{row_number}: expected 0 or 1")
    return value.strip() == "1"
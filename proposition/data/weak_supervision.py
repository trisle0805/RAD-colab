"""Validated loaders for immutable Stage-1 weak-supervision v3 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

UNKNOWN_LABEL = -1
_VALID_LABELS = frozenset((0, 1, UNKNOWN_LABEL))
_REQUIRED_LABEL_FIELDS = (
    "case_id",
    "split",
    "proposition_id",
    "label_text",
    "label_img",
    "p_img",
)
_REQUIRED_MASK_FIELDS = (
    "proposition_id",
    "text_trainable",
    "image_trainable",
    "active_scoring",
    "min_labels",
)
_REQUIRED_CSV_FIELDS = (
    "proposition_id",
    "disease_id",
    "category",
    "canonicalDescription",
    "polarity",
    "text_trainable",
    "image_trainable",
    "active_scoring",
)


@dataclass(frozen=True)
class ActiveProposition:
    """Audit metadata for one proposition in the v3 active-set manifest."""

    proposition_id: str
    disease_id: str
    category: str
    canonical_description: str
    polarity: int
    text_trainable: bool
    image_trainable: bool
    active_scoring: bool


@dataclass(frozen=True)
class WeakSupervisionArtifacts:
    """Aligned v3 weak labels and proposition-level trainability masks.

    Label arrays retain their artifact layout: ``[case, proposition]``. Their
    columns are explicitly identified by ``proposition_ids`` and must never be
    interpreted through an assumed KB or CSV position.
    """

    case_ids: NDArray[np.str_]
    splits: NDArray[np.str_]
    proposition_ids: NDArray[np.str_]
    label_text: NDArray[np.int8]
    label_img: NDArray[np.int8]
    p_img: NDArray[np.floating[Any]]
    text_trainable: NDArray[np.bool_]
    image_trainable: NDArray[np.bool_]
    active_scoring: NDArray[np.bool_]
    min_labels: int
    active_propositions: tuple[ActiveProposition, ...]
    metadata: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]

    @property
    def active_indices(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.active_scoring)

    @property
    def active_proposition_ids(self) -> NDArray[np.str_]:
        return self.proposition_ids[self.active_scoring]

    def case_indices(self, case_ids: Iterable[str]) -> NDArray[np.intp]:
        index = {case_id: position for position, case_id in enumerate(self.case_ids.tolist())}
        requested = list(case_ids)
        missing = [case_id for case_id in requested if case_id not in index]
        if missing:
            raise KeyError(f"weak-label artifact has no case_id values: {missing[:5]!r}")
        return np.asarray([index[case_id] for case_id in requested], dtype=np.intp)


def _load_npz(path: str | Path, required_fields: tuple[str, ...]) -> dict[str, NDArray[Any]]:
    artifact_path = Path(path)
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            missing = set(required_fields).difference(payload.files)
            if missing:
                raise ValueError(
                    f"{artifact_path}: missing required fields {sorted(missing)!r}; "
                    f"found {sorted(payload.files)!r}"
                )
            return {name: np.asarray(payload[name]) for name in required_fields}
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Weak-supervision artifact not found: {artifact_path}") from error
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(str(artifact_path)):
            raise
        raise ValueError(f"Unable to read NPZ artifact {artifact_path}: {error}") from error


def _as_string_vector(value: NDArray[Any], field: str, source: Path) -> NDArray[np.str_]:
    if value.ndim != 1:
        raise ValueError(f"{source}: {field} must be one-dimensional, got shape {value.shape}")
    if value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{source}: {field} must have a string dtype, got {value.dtype}")
    normalized = np.asarray(value, dtype=str)
    if np.any(np.char.str_len(normalized) == 0):
        raise ValueError(f"{source}: {field} contains an empty identifier")
    return normalized


def _require_unique(values: NDArray[np.str_], field: str, source: Path) -> None:
    unique, counts = np.unique(values, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        raise ValueError(f"{source}: duplicate {field} values: {duplicates[:5].tolist()!r}")


def _as_binary_vector(value: NDArray[Any], field: str, source: Path) -> NDArray[np.bool_]:
    if value.ndim != 1:
        raise ValueError(f"{source}: {field} must be one-dimensional, got shape {value.shape}")
    if not np.all(np.isin(value, (0, 1, False, True))):
        raise ValueError(f"{source}: {field} must contain only 0/1 values")
    return np.asarray(value, dtype=bool)


def _as_label_matrix(
    value: NDArray[Any], field: str, case_count: int, proposition_count: int, source: Path
) -> NDArray[np.int8]:
    if value.shape != (case_count, proposition_count):
        raise ValueError(
            f"{source}: {field} must have shape ({case_count}, {proposition_count}), "
            f"got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.integer) or not np.all(np.isin(value, tuple(_VALID_LABELS))):
        raise ValueError(f"{source}: {field} must contain only PRESENT=1, ABSENT=0, UNKNOWN=-1")
    return np.asarray(value, dtype=np.int8)


def _read_active_csv(path: str | Path) -> tuple[ActiveProposition, ...]:
    csv_path = Path(path)
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{csv_path}: CSV must include a header")
            missing = set(_REQUIRED_CSV_FIELDS).difference(reader.fieldnames)
            if missing:
                raise ValueError(f"{csv_path}: missing required columns {sorted(missing)!r}")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    polarity = int(row["polarity"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{csv_path}:{row_number}: polarity must be -1 or 1") from error
                if polarity not in (-1, 1):
                    raise ValueError(f"{csv_path}:{row_number}: polarity must be -1 or 1")
                fields = ("proposition_id", "disease_id", "category", "canonicalDescription")
                if any(not row[field] or not row[field].strip() for field in fields):
                    raise ValueError(f"{csv_path}:{row_number}: required metadata is empty")
                rows.append(
                    ActiveProposition(
                        proposition_id=row["proposition_id"].strip(),
                        disease_id=row["disease_id"].strip(),
                        category=row["category"].strip(),
                        canonical_description=row["canonicalDescription"].strip(),
                        polarity=polarity,
                        text_trainable=_parse_bool(row["text_trainable"], csv_path, row_number),
                        image_trainable=_parse_bool(row["image_trainable"], csv_path, row_number),
                        active_scoring=_parse_bool(row["active_scoring"], csv_path, row_number),
                    )
                )
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Active-proposition CSV not found: {csv_path}") from error
    _require_unique(np.asarray([row.proposition_id for row in rows], dtype=str), "proposition_id", csv_path)
    return tuple(rows)


def _parse_bool(value: str | None, source: Path, row_number: int) -> bool:
    if value is None or value.strip() not in {"0", "1"}:
        raise ValueError(f"{source}:{row_number}: expected 0 or 1")
    return value.strip() == "1"


def _load_metadata(path: str | Path) -> Mapping[str, Any]:
    metadata_path = Path(path)
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Weak-supervision metadata not found: {metadata_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in weak-supervision metadata {metadata_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{metadata_path}: metadata root must be an object")
    return payload


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of an artifact without loading its full content."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weak_supervision_v3(
    *,
    labels_npz: str | Path,
    active_masks_npz: str | Path,
    active_csv: str | Path,
    metadata_json: str | Path,
    verify_hashes: bool = False,
) -> WeakSupervisionArtifacts:
    """Load v3 artifacts and validate all ID-based data contracts.

    ``verify_hashes`` validates only unambiguous SHA256 entries exposed by the
    metadata manifest. A manifest with no recognized digest fields is accepted;
    the returned hashes always record the files actually loaded for audit.
    """

    labels_path = Path(labels_npz)
    masks_path = Path(active_masks_npz)
    csv_path = Path(active_csv)
    metadata_path = Path(metadata_json)
    labels = _load_npz(labels_path, _REQUIRED_LABEL_FIELDS)
    masks = _load_npz(masks_path, _REQUIRED_MASK_FIELDS)
    metadata = _load_metadata(metadata_path)

    case_ids = _as_string_vector(labels["case_id"], "case_id", labels_path)
    splits = _as_string_vector(labels["split"], "split", labels_path)
    proposition_ids = _as_string_vector(labels["proposition_id"], "proposition_id", labels_path)
    _require_unique(case_ids, "case_id", labels_path)
    _require_unique(proposition_ids, "proposition_id", labels_path)
    if splits.shape != case_ids.shape:
        raise ValueError(f"{labels_path}: split must align one-to-one with case_id")

    text_trainable = _as_binary_vector(masks["text_trainable"], "text_trainable", masks_path)
    image_trainable = _as_binary_vector(masks["image_trainable"], "image_trainable", masks_path)
    active_scoring = _as_binary_vector(masks["active_scoring"], "active_scoring", masks_path)
    mask_proposition_ids = _as_string_vector(masks["proposition_id"], "proposition_id", masks_path)
    _require_unique(mask_proposition_ids, "proposition_id", masks_path)
    if set(proposition_ids.tolist()) != set(mask_proposition_ids.tolist()):
        raise ValueError(
            "labels and active-mask proposition_id sets differ: "
            f"missing from masks={sorted(set(proposition_ids) - set(mask_proposition_ids))[:5]!r}; "
            f"missing from labels={sorted(set(mask_proposition_ids) - set(proposition_ids))[:5]!r}"
        )
    if not (text_trainable.shape == image_trainable.shape == active_scoring.shape == mask_proposition_ids.shape):
        raise ValueError(f"{masks_path}: proposition mask lengths must match proposition_id")
    mask_order = {prop_id: index for index, prop_id in enumerate(mask_proposition_ids.tolist())}
    reorder = np.asarray([mask_order[prop_id] for prop_id in proposition_ids], dtype=np.intp)
    text_trainable = text_trainable[reorder]
    image_trainable = image_trainable[reorder]
    active_scoring = active_scoring[reorder]

    min_labels_values = np.asarray(masks["min_labels"])
    if min_labels_values.size != 1 or not np.issubdtype(min_labels_values.dtype, np.integer):
        raise ValueError(f"{masks_path}: min_labels must be one integer value")
    min_labels = int(min_labels_values.reshape(-1)[0])
    if min_labels < 1:
        raise ValueError(f"{masks_path}: min_labels must be positive")

    label_text = _as_label_matrix(labels["label_text"], "label_text", len(case_ids), len(proposition_ids), labels_path)
    label_img = _as_label_matrix(labels["label_img"], "label_img", len(case_ids), len(proposition_ids), labels_path)
    p_img = np.asarray(labels["p_img"])
    if p_img.shape != label_img.shape or not np.issubdtype(p_img.dtype, np.number):
        raise ValueError(f"{labels_path}: p_img must be numeric and align with label_img")

    active_rows = _read_active_csv(csv_path)
    rows_by_id = {row.proposition_id: row for row in active_rows}
    artifact_ids = set(proposition_ids.tolist())
    csv_ids = set(rows_by_id)
    if artifact_ids != csv_ids:
        raise ValueError(
            "active_propositions CSV and NPZ proposition IDs differ: "
            f"missing from CSV={sorted(artifact_ids - csv_ids)[:5]!r}; "
            f"missing from NPZ={sorted(csv_ids - artifact_ids)[:5]!r}"
        )
    for column, expected in (
        ("text_trainable", text_trainable),
        ("image_trainable", image_trainable),
        ("active_scoring", active_scoring),
    ):
        observed = np.asarray([getattr(rows_by_id[prop_id], column) for prop_id in proposition_ids], dtype=bool)
        if not np.array_equal(observed, expected):
            mismatch = proposition_ids[observed != expected][:5].tolist()
            raise ValueError(f"{column} differs between active CSV and masks for {mismatch!r}")

    hashes = {
        "labels_raw_v3.npz": sha256_file(labels_path),
        "active_masks_v3.npz": sha256_file(masks_path),
        "active_propositions_v3.csv": sha256_file(csv_path),
        "metadata_final_v3.json": sha256_file(metadata_path),
    }
    if verify_hashes:
        _verify_manifest_hashes(metadata, hashes)

    return WeakSupervisionArtifacts(
        case_ids=case_ids,
        splits=splits,
        proposition_ids=proposition_ids,
        label_text=label_text,
        label_img=label_img,
        p_img=p_img,
        text_trainable=text_trainable,
        image_trainable=image_trainable,
        active_scoring=active_scoring,
        min_labels=min_labels,
        active_propositions=tuple(rows_by_id[prop_id] for prop_id in proposition_ids),
        metadata=metadata,
        artifact_hashes=hashes,
    )


def _verify_manifest_hashes(metadata: Mapping[str, Any], actual_hashes: Mapping[str, str]) -> None:
    candidate_maps = [metadata.get("sha256"), metadata.get("artifact_hashes"), metadata.get("hashes")]
    expected = next((mapping for mapping in candidate_maps if isinstance(mapping, Mapping)), None)
    if expected is None:
        return
    for filename, actual in actual_hashes.items():
        expected_digest = expected.get(filename)
        if expected_digest is not None and expected_digest != actual:
            raise ValueError(f"metadata SHA256 mismatch for {filename}: expected {expected_digest}, got {actual}")

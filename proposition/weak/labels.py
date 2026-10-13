"""Data contracts and deterministic helpers for WS_03 weak labels.

This module deliberately has no model imports.  It can therefore be used by
Stage 1, the survey notebook, and tests without downloading NLI/CLIP models.
All WS_03 files are written under ``weak_labels_dir``; WS_02 inputs are only
read by the runner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

UNKNOWN = np.int8(-1)
ABSENT = np.int8(0)
PRESENT = np.int8(1)

LLM_NOT_SENT = np.int8(-1)
LLM_NOT_MENTIONED = np.int8(0)
LLM_PRESENT = np.int8(1)
LLM_ABSENT = np.int8(2)

LLM_GROUP_CODES = {"": 0, "A": 1, "B": 2, "C": 3, "D": 4}
LLM_GROUP_NAMES = {value: key for key, value in LLM_GROUP_CODES.items()}
REVIEW_COLUMNS = ("proposition_id", "disease_id", "category", "canonicalDescription")


@dataclass(frozen=True)
class Proposition:
    """An immutable WS_02 proposition, retaining audit-only review context."""

    proposition_id: str
    disease_id: str
    category: str
    image_applicable: bool
    sub_hypotheses: tuple[str, ...]
    hypotheses: tuple[str, ...]
    canonical_description: str


@dataclass
class WeakLabels:
    """In-memory representation of the public ``labels.npz`` contract."""

    arrays: dict[str, np.ndarray]

    @property
    def case_id(self) -> np.ndarray:
        return self.arrays["case_id"]

    @property
    def proposition_id(self) -> np.ndarray:
        return self.arrays["proposition_id"]

    @property
    def case_index(self) -> dict[str, int]:
        return {str(value): index for index, value in enumerate(self.case_id.tolist())}

    @property
    def proposition_index(self) -> dict[str, int]:
        return {str(value): index for index, value in enumerate(self.proposition_id.tolist())}

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **self.arrays)
        return output


def load_weak_labels(path: str | Path) -> WeakLabels:
    """Load and validate a WS_03 ``labels.npz`` file.

    The loader intentionally does not infer missing fields: an incomplete
    result is unsafe because unknown labels must never become absent labels.
    """

    required = {
        "case_id", "split", "proposition_id", "label_text_nli", "label_text",
        "llm_group", "llm_verdict", "llm_quote_invalid", "label_img",
        "label_final", "p_entail", "p_contra", "p_img", "best_sentence_idx",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"labels.npz is missing required arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in archive.files}
    rows, columns = arrays["label_final"].shape
    for name in required - {"case_id", "split", "proposition_id"}:
        if arrays[name].shape != (rows, columns):
            raise ValueError(f"{name} has shape {arrays[name].shape}, expected {(rows, columns)}")
    if arrays["case_id"].shape != (rows,) or arrays["split"].shape != (rows,):
        raise ValueError("case_id and split must have one entry per label row")
    if arrays["proposition_id"].shape != (columns,):
        raise ValueError("proposition_id must have one entry per label column")
    allowed = {int(UNKNOWN), int(ABSENT), int(PRESENT)}
    for name in ("label_text_nli", "label_text", "label_img", "label_final"):
        values = set(np.unique(arrays[name]).tolist())
        if not values <= allowed:
            raise ValueError(f"{name} contains invalid labels: {sorted(values - allowed)}")
    return WeakLabels(arrays)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_propositions(final_path: str | Path, review_path: str | Path) -> list[Proposition]:
    """Load immutable WS_02 final propositions in their established ordering."""

    with Path(final_path).open("r", encoding="utf-8") as handle:
        final = json.load(handle)
    if not isinstance(final, list) or not final:
        raise ValueError("propositions_final.json must be a non-empty JSON list")

    with Path(review_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(REVIEW_COLUMNS) <= set(reader.fieldnames or ()):
            raise ValueError("WS_02 propositions_review.csv lacks required audit columns")
        review = {row["proposition_id"]: row for row in reader}

    result: list[Proposition] = []
    identifiers: set[str] = set()
    for item in final:
        prop_id = str(item["proposition_id"])
        if prop_id in identifiers:
            raise ValueError(f"Duplicate proposition_id in final artifact: {prop_id}")
        identifiers.add(prop_id)
        context = review.get(prop_id)
        if context is None:
            raise ValueError(f"Final proposition is absent from review CSV: {prop_id}")
        hypotheses = tuple(str(value) for value in item["hypotheses"])
        sub_hypotheses = tuple(str(value) for value in item["sub_hypotheses"])
        if not hypotheses or len(hypotheses) != len(sub_hypotheses):
            raise ValueError(f"Invalid hypotheses for {prop_id}")
        result.append(Proposition(
            proposition_id=prop_id,
            disease_id=str(item["disease_id"]),
            category=str(item["category"]),
            image_applicable=bool(int(item["image_applicable"])),
            sub_hypotheses=sub_hypotheses,
            hypotheses=hypotheses,
            canonical_description=str(context["canonicalDescription"]),
        ))
    return result


def split_sentences(caption: str) -> list[str]:
    """Use the documented conservative sentence splitter, preserving content."""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(caption))]
    return [sentence for sentence in sentences if sentence]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE)


def mask_disease_aliases(caption: str, aliases: Iterable[str]) -> str:
    """Mask approved aliases with whole-word matching for NLI and LLM only."""

    masked = str(caption)
    for alias in sorted({value.strip() for value in aliases if value.strip()}, key=len, reverse=True):
        masked = _alias_pattern(alias).sub("this skin condition", masked)
    return masked


def suggested_alias_rows(disease_ids: Iterable[str]) -> list[dict[str, str]]:
    """Create conservative aliases for review; do not claim clinical synonyms."""

    rows: list[dict[str, str]] = []
    for disease_id in sorted(set(map(str, disease_ids))):
        variants = {disease_id, disease_id.replace("-", " "), disease_id.replace("_", " ")}
        for alias in sorted(variant for variant in variants if variant.strip()):
            rows.append({"disease_id": disease_id, "alias": alias, "source": "automatic_id_variant"})
    return rows


def write_alias_csv(path: str | Path, rows: Sequence[Mapping[str, str]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("disease_id", "alias", "source"))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def read_alias_csv(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"disease_id", "alias", "source"} <= set(reader.fieldnames or ()):
            raise ValueError("disease_alias_v1.csv must have disease_id, alias, source columns")
        return [str(row["alias"]).strip() for row in reader if str(row["alias"]).strip()]


def merge_labels(label_text: np.ndarray, label_img: np.ndarray, teacher: str = "both") -> np.ndarray:
    """Apply A4 exactly; a disagreement is unknown, never silently absent."""

    if teacher not in {"text", "image", "both"}:
        raise ValueError("teacher must be text, image, or both")
    if teacher == "text":
        return label_text.copy()
    if teacher == "image":
        return label_img.copy()
    final = np.full(label_text.shape, UNKNOWN, dtype=np.int8)
    same = label_text == label_img
    final[same] = label_text[same]
    text_unknown = label_text == UNKNOWN
    image_unknown = label_img == UNKNOWN
    final[text_unknown & ~image_unknown] = label_img[text_unknown & ~image_unknown]
    final[image_unknown & ~text_unknown] = label_text[image_unknown & ~text_unknown]
    return final


def normalise_whitespace(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def quote_is_supported(quote: str, caption: str) -> bool:
    clean_quote = normalise_whitespace(quote)
    return bool(clean_quote) and clean_quote in normalise_whitespace(caption)
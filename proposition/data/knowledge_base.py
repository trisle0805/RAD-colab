"""Knowledge-base loading and validation for proposition-based models.

This module is intentionally independent of tokenizers and devices.  It reads the
final proposition JSON once, validates its schema, preserves its ordering, and
builds the integer mappings consumed by the model and PECL loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


CATEGORY_ORDER = ("morphology", "location", "symptom", "history", "lab")
_CATEGORY_RANK = {category: rank for rank, category in enumerate(CATEGORY_ORDER)}


@dataclass(frozen=True)
class PropositionRecord:
    """One validated proposition, augmented with its owning disease metadata."""

    proposition_id: str
    disease_id: str
    disease_index: int
    category: str
    canonical_description: str
    polarity: int
    source_excerpt: tuple[str, ...]
    source_reference: str

    @property
    def alignment_text(self) -> str:
        """Return the polarity-aware text used as a PECL prototype."""

        if self.polarity == 1:
            return self.canonical_description
        return f"absence of {self.canonical_description}"

    def to_index_dict(self) -> dict[str, Any]:
        """Serialize with the field names required by the checklist artifact."""

        return {
            "proposition_id": self.proposition_id,
            "disease_id": self.disease_id,
            "disease_index": self.disease_index,
            "category": self.category,
            "canonicalDescription": self.canonical_description,
            "polarity": self.polarity,
            "sourceExcerpt": list(self.source_excerpt),
            "sourceReference": self.source_reference,
        }


@dataclass(frozen=True)
class PropositionKnowledgeBase:
    """Validated KB plus deterministic text and disease index mappings."""

    disease_ids: tuple[str, ...]
    propositions: tuple[PropositionRecord, ...]
    prop_disease: torch.LongTensor
    prop_polarity: torch.Tensor
    unique_texts: tuple[str, ...]
    query_text_index: torch.LongTensor
    align_text_index: torch.LongTensor
    unique_alignment_texts: tuple[str, ...]
    align_unique_index: torch.LongTensor
    disease_to_prop_indices: tuple[torch.LongTensor, ...]
    disease_to_align_unique_indices: tuple[torch.LongTensor, ...]
    disease_query_texts: tuple[str, ...]

    @property
    def num_diseases(self) -> int:
        return len(self.disease_ids)

    @property
    def num_propositions(self) -> int:
        return len(self.propositions)

    @property
    def disease_prop_counts(self) -> torch.LongTensor:
        return torch.tensor(
            [indices.numel() for indices in self.disease_to_prop_indices],
            dtype=torch.long,
        )

    @property
    def negative_alignment_texts(self) -> tuple[str, ...]:
        return tuple(
            record.alignment_text
            for record in self.propositions
            if record.polarity == -1
        )

    def write_proposition_index(self, path: str | Path) -> None:
        """Write checklist provenance in exactly the flattened proposition order."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_index_dict() for record in self.propositions]
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(output_path)


def _require_nonempty_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value.strip()


def _parse_source_excerpt(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, str):
        excerpts = (value.strip(),) if value.strip() else ()
    elif isinstance(value, list):
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"{context}: sourceExcerpt entries must be non-empty strings")
        excerpts = tuple(item.strip() for item in value)
    else:
        raise ValueError(f"{context}: sourceExcerpt must be a string or list of strings")

    if not excerpts:
        raise ValueError(f"{context}: sourceExcerpt must not be empty")
    return excerpts


def _deduplicate_with_indices(texts: Sequence[str]) -> tuple[tuple[str, ...], torch.LongTensor]:
    unique: list[str] = []
    text_to_index: dict[str, int] = {}
    indices: list[int] = []
    for text in texts:
        index = text_to_index.get(text)
        if index is None:
            index = len(unique)
            text_to_index[text] = index
            unique.append(text)
        indices.append(index)
    return tuple(unique), torch.tensor(indices, dtype=torch.long)


def _validate_label_order(disease_ids: Sequence[str], label_names: Sequence[str]) -> None:
    labels = tuple(label_names)
    diseases = tuple(disease_ids)
    if labels == diseases:
        return

    details: list[str] = []
    for index in range(max(len(labels), len(diseases))):
        csv_name = labels[index] if index < len(labels) else "<missing>"
        kb_name = diseases[index] if index < len(diseases) else "<missing>"
        if csv_name != kb_name:
            details.append(f"index {index}: CSV={csv_name!r}, KB={kb_name!r}")
    raise ValueError(
        "KB disease order must exactly match the CSV label header; " + "; ".join(details)
    )


def load_proposition_kb(
    kb_path: str | Path,
    label_names: Sequence[str] | None = None,
) -> PropositionKnowledgeBase:
    """Load and validate a proposition KB without modifying or normalizing its data.

    Args:
        kb_path: Path to ``kb_propositions.json``.
        label_names: Label names from CSV columns 2 onward.  When supplied, their
            values and order must exactly equal the KB disease order.
    """

    path = Path(kb_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Proposition KB not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in proposition KB {path}: {error}") from error

    if not isinstance(raw, list) or not raw:
        raise ValueError("Proposition KB root must be a non-empty list of diseases")

    disease_ids: list[str] = []
    records: list[PropositionRecord] = []
    seen_diseases: set[str] = set()
    seen_propositions: set[str] = set()
    disease_to_props: list[torch.LongTensor] = []

    for disease_index, disease in enumerate(raw):
        context = f"disease[{disease_index}]"
        if not isinstance(disease, dict):
            raise ValueError(f"{context}: disease entry must be an object")

        disease_id = _require_nonempty_string(disease.get("disease_id"), "disease_id", context)
        if disease_id in seen_diseases:
            raise ValueError(f"{context}: duplicate disease_id {disease_id!r}")
        seen_diseases.add(disease_id)
        disease_ids.append(disease_id)

        source_reference = _require_nonempty_string(
            disease.get("sourceReference"), "sourceReference", context
        )
        propositions = disease.get("propositions")
        if not isinstance(propositions, list) or not propositions:
            raise ValueError(f"{context}: propositions must be a non-empty list")

        disease_prop_indices: list[int] = []
        previous_category_rank = -1
        for local_index, proposition in enumerate(propositions):
            prop_context = f"{context}.propositions[{local_index}]"
            if not isinstance(proposition, dict):
                raise ValueError(f"{prop_context}: proposition must be an object")

            proposition_id = _require_nonempty_string(
                proposition.get("proposition_id"), "proposition_id", prop_context
            )
            if proposition_id in seen_propositions:
                raise ValueError(f"{prop_context}: duplicate proposition_id {proposition_id!r}")
            seen_propositions.add(proposition_id)

            category = _require_nonempty_string(
                proposition.get("category"), "category", prop_context
            )
            if category not in _CATEGORY_RANK:
                raise ValueError(
                    f"{prop_context}: unsupported category {category!r}; "
                    f"expected one of {CATEGORY_ORDER}"
                )
            category_rank = _CATEGORY_RANK[category]
            if category_rank < previous_category_rank:
                raise ValueError(f"{prop_context}: categories are not in canonical order")
            previous_category_rank = category_rank

            canonical_description = _require_nonempty_string(
                proposition.get("canonicalDescription"),
                "canonicalDescription",
                prop_context,
            )
            polarity = proposition.get("polarity")
            if not isinstance(polarity, int) or isinstance(polarity, bool) or polarity not in (-1, 1):
                raise ValueError(f"{prop_context}: polarity must be integer -1 or 1")
            source_excerpt = _parse_source_excerpt(
                proposition.get("sourceExcerpt"), prop_context
            )

            disease_prop_indices.append(len(records))
            records.append(
                PropositionRecord(
                    proposition_id=proposition_id,
                    disease_id=disease_id,
                    disease_index=disease_index,
                    category=category,
                    canonical_description=canonical_description,
                    polarity=polarity,
                    source_excerpt=source_excerpt,
                    source_reference=source_reference,
                )
            )
        disease_to_props.append(torch.tensor(disease_prop_indices, dtype=torch.long))

    if label_names is not None:
        _validate_label_order(disease_ids, label_names)

    query_texts = tuple(record.canonical_description for record in records)
    alignment_texts = tuple(record.alignment_text for record in records)

    unique_texts, combined_indices = _deduplicate_with_indices(query_texts + alignment_texts)
    proposition_count = len(records)
    query_text_index = combined_indices[:proposition_count]
    align_text_index = combined_indices[proposition_count:]

    unique_alignment_texts, align_unique_index = _deduplicate_with_indices(alignment_texts)
    disease_to_align: list[torch.LongTensor] = []
    disease_query_texts: list[str] = []
    for prop_indices in disease_to_props:
        unique_indices = sorted(
            {int(align_unique_index[index]) for index in prop_indices.tolist()}
        )
        disease_to_align.append(torch.tensor(unique_indices, dtype=torch.long))
        disease_query_texts.append(
            "; ".join(alignment_texts[index] for index in prop_indices.tolist())
        )

    return PropositionKnowledgeBase(
        disease_ids=tuple(disease_ids),
        propositions=tuple(records),
        prop_disease=torch.tensor(
            [record.disease_index for record in records], dtype=torch.long
        ),
        prop_polarity=torch.tensor(
            [record.polarity for record in records], dtype=torch.int8
        ),
        unique_texts=unique_texts,
        query_text_index=query_text_index,
        align_text_index=align_text_index,
        unique_alignment_texts=unique_alignment_texts,
        align_unique_index=align_unique_index,
        disease_to_prop_indices=tuple(disease_to_props),
        disease_to_align_unique_indices=tuple(disease_to_align),
        disease_query_texts=tuple(disease_query_texts),
    )

"""Prepare proposition review tables for weak supervision.

The original proposition knowledge base is read-only. This module creates a
human-reviewable CSV and, only after review, locks a deterministic final
proposition version for the weak-label pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HYPOTHESIS_TEMPLATE = "The patient has {sub_hypothesis}."
SUB_HYPOTHESIS_SEPARATOR = " ‖ "

INTERNAL_OR_SYSTEMIC_KEYWORDS = (
    "lung",
    "pulmonary",
    "kidney",
    "renal",
    "liver",
    "heart",
    "cardiac",
    "pericardium",
    "pleura",
    "brain",
    "central nervous system",
    "muscle",
    "weakness",
    "lymph node",
    "metastases",
    "gastrointestinal",
    "pancreas",
    "omentum",
    "testis",
    "cervix",
    "oropharynx",
    "hypertension",
    "internal organs",
    "multisystem",
    "larval migration",
    "parasitic infection",
)

BORDERLINE_KEYWORDS = (
    "eye",
    "eyes",
    "ocular",
    "conjunctiva",
    "conjunctival",
    "mucosa",
    "mucosal",
    "oral",
    "mouth",
    "genital",
    "vulva",
    "vulvar",
    "penile",
    "systemic",
    "generalized",
    "widespread",
)

TEXT_ONLY_CATEGORY_NAMES = {"symptom", "history"}
LAB_CATEGORY_NAMES = {"lab", "laboratory", "biopsy", "imaging", "diagnostic"}
LOCATION_PREPOSITIONS = r"(?:on|in|over|at|around|across|of)"

REVIEW_COLUMNS = (
    "proposition_id",
    "disease_id",
    "category",
    "polarity",
    "canonicalDescription",
    "auto_observable",
    "auto_reason",
    "image_applicable",
    "expression_type",
    "sub_hypotheses",
    "review_needed",
    "final_observable",
    "final_sub_hypotheses",
    "note",
)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)", text.casefold()) is not None


def _matching_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    return [keyword for keyword in keywords if _contains_keyword(text, keyword)]


def _split_list_items(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip(" ,;"))
    normalized = re.sub(r"\s*(?:,|;|\band/or\b|\bor\b|\band\b)\s*", "|", normalized)
    return [item.strip(" ,;") for item in normalized.split("|") if item.strip(" ,;")]


def _deduplicate(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = " ".join(item.split()).strip(" ,;")
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _site_list_sub_hypotheses(description: str) -> list[str] | None:
    match = re.match(
        rf"^(?P<prefix>.+?\s+{LOCATION_PREPOSITIONS}\s+)(?P<sites>[^.]+)$",
        description,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    sites = match.group("sites")
    if not re.search(r",|\band\b|\bor\b", sites, flags=re.IGNORECASE):
        return None

    items = _split_list_items(sites)
    if len(items) < 2 or any(len(item.split()) > 6 for item in items):
        return None
    first_article = re.match(r"^(the|a|an)\s+", items[0], flags=re.IGNORECASE)
    if first_article:
        article = first_article.group(1)
        items = [
            item if re.match(r"^(the|a|an)\s+", item, flags=re.IGNORECASE) else f"{article} {item}"
            for item in items
        ]
    return _deduplicate(f"{match.group('prefix').strip()} {site}" for site in items)


def _alternative_sub_hypotheses(description: str) -> list[str] | None:
    if not re.search(r"\bor\b", description, flags=re.IGNORECASE):
        return None
    if re.search(r"\bwith\b", description, flags=re.IGNORECASE):
        return None

    words = description.split()
    try:
        or_index = next(index for index, word in enumerate(words) if word.casefold() == "or")
    except StopIteration:
        return None

    before = " ".join(words[:or_index])
    after = " ".join(words[or_index + 1 :])
    if not before or not after:
        return None

    suffix_match = re.match(r"^(?P<alternatives>.+?)\s+(?P<suffix>[A-Za-z][A-Za-z -]+)$", before)
    if suffix_match and re.search(r",|\s", suffix_match.group("alternatives")):
        alternatives = _split_list_items(suffix_match.group("alternatives"))
        if len(alternatives) >= 2:
            suffix = suffix_match.group("suffix").strip()
            return _deduplicate([*(f"{item} {suffix}" for item in alternatives), after])

    left_items = _split_list_items(before)
    if len(left_items) == 1:
        return _deduplicate([before, after])
    return None


def classify_expression(description: str) -> tuple[str, list[str], bool]:
    """Classify and conservatively split a canonical description.

    Only ``site_list`` and obvious two-way ``alternatives`` are split. Every
    automatic split is marked for human review.
    """

    clean = " ".join(description.split()).strip()
    site_list = _site_list_sub_hypotheses(clean)
    if site_list:
        return "site_list", site_list, True

    alternatives = _alternative_sub_hypotheses(clean)
    if alternatives:
        return "alternatives", alternatives, True

    lower = clean.casefold()
    if " with " in lower and " and " in lower:
        return "pattern", [clean], False
    if " and " in lower:
        return "and_pair", [clean], False
    return "single", [clean], False


def classify_observability(category: str, description: str) -> tuple[int, str, int]:
    """Return ``auto_observable``, explanation, and review flag.

    ``observable`` means usable from an image *or* clinical caption. Symptoms
    and history are therefore text-only candidates, not automatic exclusions.
    """

    category_key = category.casefold().strip()
    internal_matches = _matching_keywords(description, INTERNAL_OR_SYSTEMIC_KEYWORDS)
    borderline_matches = _matching_keywords(description, BORDERLINE_KEYWORDS)

    if category_key in LAB_CATEGORY_NAMES:
        return 0, f"unsupported_record_category:{category_key}", 0
    if internal_matches:
        return 0, f"internal_or_systemic:{', '.join(internal_matches)}", 0
    if category_key in TEXT_ONLY_CATEGORY_NAMES:
        return 1, f"text_only_caption_category:{category_key}", 1
    if category_key in {"morphology", "location"}:
        if borderline_matches:
            return 1, f"borderline_surface_or_systemic:{', '.join(borderline_matches)}", 1
        return 1, "surface_morphology_or_location", 0
    return 0, f"unsupported_category:{category_key}", 0


def _load_kb(kb_path: str | Path) -> list[dict[str, Any]]:
    path = Path(kb_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("KB root must be a non-empty list")
    return payload


def build_review_rows(kb_path: str | Path) -> list[dict[str, Any]]:
    """Build one review row per original KB proposition without mutating the KB."""

    rows: list[dict[str, Any]] = []
    for disease in _load_kb(kb_path):
        disease_id = str(disease["disease_id"])
        for proposition in disease["propositions"]:
            description = str(proposition["canonicalDescription"]).strip()
            category = str(proposition["category"]).strip()
            observable, reason, needs_review = classify_observability(category, description)
            expression_type, sub_hypotheses, split_needs_review = classify_expression(description)
            final_sub_hypotheses = SUB_HYPOTHESIS_SEPARATOR.join(sub_hypotheses) if observable else ""
            rows.append(
                {
                    "proposition_id": str(proposition["proposition_id"]),
                    "disease_id": disease_id,
                    "category": category,
                    "polarity": int(proposition["polarity"]),
                    "canonicalDescription": description,
                    "auto_observable": observable,
                    "auto_reason": reason,
                    "image_applicable": int(observable == 1 and category.casefold() == "morphology"),
                    "expression_type": expression_type if observable else "not_applicable",
                    "sub_hypotheses": final_sub_hypotheses,
                    "review_needed": int(needs_review or split_needs_review),
                    "final_observable": observable,
                    "final_sub_hypotheses": final_sub_hypotheses,
                    "note": "",
                }
            )
    return sorted(rows, key=lambda row: (-int(row["review_needed"]), row["proposition_id"]))


def write_review_csv(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write the review table in a stable column order."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in REVIEW_COLUMNS} for row in rows)
    return path


def _parse_binary(value: Any, field: str, proposition_id: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{proposition_id}: {field} must be 0 or 1") from error
    if parsed not in (0, 1):
        raise ValueError(f"{proposition_id}: {field} must be 0 or 1")
    return parsed


def _parse_sub_hypotheses(value: Any, proposition_id: str) -> list[str]:
    hypotheses = _deduplicate(str(value).split(SUB_HYPOTHESIS_SEPARATOR))
    if not hypotheses:
        raise ValueError(f"{proposition_id}: final_sub_hypotheses must not be empty")
    return hypotheses


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_review_csv(review_csv_path: str | Path) -> list[dict[str, str]]:
    with Path(review_csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise ValueError("Review CSV columns do not match the required Stage 02 schema")
        return list(reader)


def export_final_version(
    kb_path: str | Path,
    review_csv_path: str | Path,
    output_dir: str | Path,
    *,
    reviewed_on: str | None = None,
) -> dict[str, Path]:
    """Validate a reviewed CSV and lock the deterministic final Stage 02 version."""

    kb_path = Path(kb_path)
    kb = _load_kb(kb_path)
    review_rows = _load_review_csv(review_csv_path)
    rows_by_id = {row["proposition_id"]: row for row in review_rows}
    if len(rows_by_id) != len(review_rows):
        raise ValueError("Review CSV contains duplicate proposition_id values")

    ordered_records: list[tuple[dict[str, Any], dict[str, str]]] = []
    disease_ids: list[str] = []
    all_prop_ids: set[str] = set()
    for disease in kb:
        disease_id = str(disease["disease_id"])
        disease_ids.append(disease_id)
        for proposition in disease["propositions"]:
            proposition_id = str(proposition["proposition_id"])
            all_prop_ids.add(proposition_id)
            row = rows_by_id.get(proposition_id)
            if row is None:
                raise ValueError(f"Review CSV is missing {proposition_id}")
            ordered_records.append((proposition, row))
    extra_ids = set(rows_by_id) - all_prop_ids
    if extra_ids:
        raise ValueError(f"Review CSV contains IDs absent from KB: {sorted(extra_ids)[:5]}")

    final: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for proposition, row in ordered_records:
        proposition_id = str(proposition["proposition_id"])
        observable = _parse_binary(row["final_observable"], "final_observable", proposition_id)
        if not observable:
            continue
        sub_hypotheses = _parse_sub_hypotheses(row["final_sub_hypotheses"], proposition_id)
        category = str(proposition["category"])
        image_applicable = _parse_binary(row["image_applicable"], "image_applicable", proposition_id)
        expected_image_applicable = int(category.casefold() == "morphology")
        if image_applicable != expected_image_applicable:
            raise ValueError(
                f"{proposition_id}: image_applicable must be {expected_image_applicable} "
                f"for an observable {category!r} proposition"
            )
        disease_id = row["disease_id"]
        if disease_id not in disease_ids:
            raise ValueError(f"{proposition_id}: disease_id is absent from KB")
        final.append(
            {
                "proposition_id": proposition_id,
                "disease_id": disease_id,
                "category": category,
                "polarity": int(proposition["polarity"]),
                "image_applicable": image_applicable,
                "sub_hypotheses": sub_hypotheses,
                "hypotheses": [HYPOTHESIS_TEMPLATE.format(sub_hypothesis=item) for item in sub_hypotheses],
            }
        )
        counts[disease_id]["observable_total"] += 1
        counts[disease_id][f"category_{category}"] += 1
        counts[disease_id]["image_applicable"] += image_applicable

    missing_diseases = [disease_id for disease_id in disease_ids if not counts[disease_id]["observable_total"]]
    if missing_diseases:
        raise ValueError(f"Each disease needs an observable proposition; missing: {missing_diseases}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reviewed_csv = destination / "propositions_review.csv"
    reviewed_csv.write_bytes(Path(review_csv_path).read_bytes())
    final_path = destination / "propositions_final.json"
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_path = destination / "summary_per_disease.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("disease_id", "observable_total", "image_applicable", "category_morphology", "category_location", "category_symptom", "category_history", "category_lab")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for disease_id in disease_ids:
            writer.writerow({field: counts[disease_id][field] if field != "disease_id" else disease_id for field in fieldnames})

    metadata = {
        "stage": "WS_02_propositions",
        "reviewed_on": reviewed_on or date.today().isoformat(),
        "kb_path": str(kb_path),
        "kb_sha256": _sha256_file(kb_path),
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
        "sub_hypothesis_separator": SUB_HYPOTHESIS_SEPARATOR,
        "internal_or_systemic_keywords": list(INTERNAL_OR_SYSTEMIC_KEYWORDS),
        "borderline_keywords": list(BORDERLINE_KEYWORDS),
        "total_kb_propositions": len(ordered_records),
        "final_observable_propositions": len(final),
    }
    metadata_path = destination / "meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "review_csv": reviewed_csv,
        "final_json": final_path,
        "summary_csv": summary_path,
        "metadata": metadata_path,
    }

"""Read-only WS_03 label-survey reports.

Reports quantify the generated labels and never alter ``labels.npz``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .labels import ABSENT, PRESENT, UNKNOWN, Proposition, WeakLabels


def _counts(labels: np.ndarray) -> tuple[int, int, int, float, float]:
    positive = int(np.count_nonzero(labels == PRESENT))
    absent = int(np.count_nonzero(labels == ABSENT))
    unknown = int(np.count_nonzero(labels == UNKNOWN))
    total = len(labels)
    coverage = (positive + absent) / total if total else 0.0
    positive_rate = positive / (positive + absent) if positive + absent else 0.0
    return positive, absent, unknown, positive_rate, coverage


def write_survey(
    weak_labels: WeakLabels, propositions: Sequence[Proposition], output_dir: str | Path,
    *, case_diseases: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write the mandatory Part-B survey tables without making interpretations."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = weak_labels.arrays
    splits = arrays["split"].astype(str)
    final = arrays["label_final"]
    text = arrays["label_text"]
    image = arrays["label_img"]
    llm_group = arrays["llm_group"]
    verdict = arrays["llm_verdict"]
    quote_invalid = arrays["llm_quote_invalid"]
    results: dict[str, Path] = {}

    per_proposition: list[dict[str, object]] = []
    for column, proposition in enumerate(propositions):
        row: dict[str, object] = {
            "proposition_id": proposition.proposition_id,
            "disease_id": proposition.disease_id,
            "category": proposition.category,
            "image_applicable": int(proposition.image_applicable),
            "canonicalDescription": proposition.canonical_description,
        }
        for split in ("all", "train", "val", "test"):
            mask = np.ones(len(splits), dtype=bool) if split == "all" else splits == split
            present, absent, unknown, rate, coverage = _counts(final[mask, column])
            prefix = f"{split}_"
            row.update({
                f"{prefix}n_present": present, f"{prefix}n_absent": absent,
                f"{prefix}n_unknown": unknown, f"{prefix}positive_rate": rate,
                f"{prefix}coverage": coverage,
                f"{prefix}n_text_determined": int(np.count_nonzero(text[mask, column] != UNKNOWN)),
                f"{prefix}n_image_determined": int(np.count_nonzero(image[mask, column] != UNKNOWN)),
            })
        row["positive_only"] = int(row["all_n_present"] > 0 and row["all_n_absent"] == 0)
        row["no_label"] = int(row["all_n_present"] + row["all_n_absent"] == 0)
        if case_diseases is not None:
            own = np.asarray(case_diseases, dtype=str) == proposition.disease_id
            row["n_present_own_disease"] = int(np.count_nonzero(final[own, column] == PRESENT))
            row["n_present_other_disease"] = int(np.count_nonzero(final[~own, column] == PRESENT))
        per_proposition.append(row)
    per_prop_path = destination / "stats_per_proposition.csv"
    pd.DataFrame(per_proposition).to_csv(per_prop_path, index=False)
    results["per_proposition"] = per_prop_path

    group_rows: list[dict[str, object]] = []
    categories = ["all", *sorted({prop.category for prop in propositions})]
    for category in categories:
        columns = list(range(len(propositions))) if category == "all" else [i for i, prop in enumerate(propositions) if prop.category == category]
        values = final[:, columns].reshape(-1)
        positive, absent, unknown, rate, coverage = _counts(values)
        group_rows.append({"group": category, "n_present": positive, "n_absent": absent, "n_unknown": unknown, "positive_to_absent_ratio": positive / absent if absent else np.nan, "positive_rate": rate, "coverage": coverage})
    groups_path = destination / "stats_per_group.csv"
    pd.DataFrame(group_rows).to_csv(groups_path, index=False)
    results["per_group"] = groups_path

    conflicts = (text != UNKNOWN) & (image != UNKNOWN) & (text != image)
    both = (text != UNKNOWN) & (image != UNKNOWN)
    agreement = pd.DataFrame({
        "text_label": ["present", "present", "absent", "absent"],
        "image_label": ["present", "absent", "present", "absent"],
        "n": [
            int(np.count_nonzero((text == PRESENT) & (image == PRESENT))),
            int(np.count_nonzero((text == PRESENT) & (image == ABSENT))),
            int(np.count_nonzero((text == ABSENT) & (image == PRESENT))),
            int(np.count_nonzero((text == ABSENT) & (image == ABSENT))),
        ],
    })
    agreement["n_both_determined"] = int(np.count_nonzero(both))
    agreement["n_conflict"] = int(np.count_nonzero(conflicts))
    agreement_path = destination / "text_image_agreement.csv"
    agreement.to_csv(agreement_path, index=False)
    results["agreement"] = agreement_path

    llm_rows: list[dict[str, object]] = []
    for code, name in ((1, "A"), (2, "B"), (3, "C"), (4, "D")):
        selected = llm_group == code
        n_selected = int(np.count_nonzero(selected))
        n_same = int(np.count_nonzero(selected & (arrays["label_text_nli"] == text)))
        before = arrays["label_text_nli"]
        row = {
            "group": name, "n_sent": n_selected, "n_unchanged": n_same,
            "n_one_to_unknown": int(np.count_nonzero(selected & (before == PRESENT) & (text == UNKNOWN))),
            "n_one_to_zero": int(np.count_nonzero(selected & (before == PRESENT) & (text == ABSENT))),
            "n_added_zero": int(np.count_nonzero(selected & (before == UNKNOWN) & (text == ABSENT))),
            "n_quote_invalid": int(np.count_nonzero(selected & (quote_invalid == 1))),
            "n_present_verdict": int(np.count_nonzero(selected & (verdict == 1))),
            "n_absent_verdict": int(np.count_nonzero(selected & (verdict == 2))),
        }
        llm_rows.append(row)
    llm_path = destination / "llm_impact.csv"
    pd.DataFrame(llm_rows).to_csv(llm_path, index=False)
    results["llm_impact"] = llm_path

    metrics = {
        "n_cases": int(final.shape[0]), "n_propositions": int(final.shape[1]),
        "n_conflicts": int(np.count_nonzero(conflicts)),
        "n_both_text_image_determined": int(np.count_nonzero(both)),
        "no_label_propositions": [str(row["proposition_id"]) for row in per_proposition if row["no_label"]],
        "positive_only_propositions": [str(row["proposition_id"]) for row in per_proposition if row["positive_only"]],
    }
    summary_json = destination / "survey_summary.json"
    summary_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md = destination / "survey_summary.md"
    summary_md.write_text(
        "\n".join([
            f"n_cases: {metrics['n_cases']}", f"n_propositions: {metrics['n_propositions']}",
            f"n_conflicts: {metrics['n_conflicts']}",
            "no_label: " + ", ".join(metrics["no_label_propositions"]),
            "positive_only: " + ", ".join(metrics["positive_only_propositions"]), "",
        ]), encoding="utf-8")
    results["summary_json"] = summary_json
    results["summary_md"] = summary_md
    return results
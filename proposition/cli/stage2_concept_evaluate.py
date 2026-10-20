"""Evaluate an immutable averaged Round-A concept head on held-out test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from proposition.data.stage2_concept_features import build_stage2_concept_features
from proposition.data.stage2_scores import load_stage2_disease_targets, load_stage2_propositions, load_stage2_score_artifact
from proposition.data.weak_supervision import sha256_file
from proposition.engine.stage2_concept_audit import audit_linear_contributions, concept_weight_rows, disease_sparsity_rows, same_disease_contribution_fraction
from proposition.engine.stage2_concept_ensemble import FrozenConceptLinearHead
from proposition.engine.stage2_concept_metrics import bootstrap_topk_confidence_intervals, single_label_diagnosis_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Round-A scalar concept head on held-out test data.")
    parser.add_argument("--frozen_head", required=True)
    parser.add_argument("--test_scores", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--active_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--nonzero_threshold", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: Path(getattr(args, name)) for name in ("frozen_head", "test_scores", "test_csv", "active_csv")}
    before = {name: sha256_file(path) for name, path in paths.items()}
    head = FrozenConceptLinearHead.load(paths["frozen_head"])
    artifact = load_stage2_score_artifact(paths["test_scores"])
    propositions = load_stage2_propositions(paths["active_csv"], artifact.proposition_ids)
    routed = build_stage2_concept_features(artifact, propositions, head.routing_mode)
    _validate_features(head, routed)
    targets = load_stage2_disease_targets(paths["test_csv"], artifact.case_ids)
    if targets.disease_ids != head.disease_ids:
        raise ValueError("test disease label columns/order differ from frozen head")
    outputs = head.contribution_outputs(routed.support_scores)
    metrics = single_label_diagnosis_metrics(outputs["logits"], targets.targets, head.disease_ids)
    output = Path(args.output_dir)
    for name in ("metrics", "predictions", "contributions", "diagnostics"):
        (output / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions" / "test_frozen.npz", case_id=artifact.case_ids, disease_id=np.asarray(head.disease_ids), targets=targets.targets.argmax(axis=1),
                        logits=outputs["logits"], probabilities=outputs["probabilities"], routing_mode=np.asarray(head.routing_mode))
    _save_contributions(output, artifact.case_ids, head, outputs)
    _write_json(output / "metrics" / "test_summary.json", {"metrics": _summary(metrics)})
    _write_rows(output / "metrics" / "test_per_disease.csv", metrics["per_disease"])
    np.save(output / "metrics" / "confusion_matrix_test.npy", metrics["confusion_matrix"])
    _write_json(output / "metrics" / "test_confidence_intervals.json", bootstrap_topk_confidence_intervals(outputs["logits"], targets.targets, seed=args.seed, n_bootstrap=args.bootstrap_samples))
    _write_audits(output, head, outputs, args.nonzero_threshold)
    after = {name: sha256_file(path) for name, path in paths.items()}
    if after != before:
        raise RuntimeError("frozen-head or test input hashes changed during evaluation")
    _write_json(output / "evaluation_manifest.json", {"input_hashes": before, "routing_mode": head.routing_mode, "head_mode": head.head_mode, "feature_ids": list(head.feature_ids), "seeds": list(head.seeds)})
    print(json.dumps(_summary(metrics), sort_keys=True))


def _validate_features(head: FrozenConceptLinearHead, routed: Any) -> None:
    if routed.feature_ids != head.feature_ids:
        raise ValueError("test routing feature IDs/order differ from frozen head")
    for feature, base, disease, category, polarity, source in zip(routed.features, head.base_proposition_ids, head.feature_disease_ids, head.feature_categories, head.feature_polarities, head.feature_sources):
        if (feature.base_proposition_id, feature.disease_id, feature.category, feature.polarity, feature.source) != (base, disease, category, int(polarity), source):
            raise ValueError("test routing feature metadata differs from frozen head")


def _save_contributions(output_dir: Path, case_ids: np.ndarray, head: FrozenConceptLinearHead, outputs: dict[str, np.ndarray]) -> None:
    logits = outputs["logits"]
    ranking = np.argsort(-logits, axis=1, kind="mergesort")
    rows = np.arange(len(case_ids))
    top1, top2 = ranking[:, 0], ranking[:, 1]
    terms = outputs["contributions"]
    proposition_terms, proposition_ids = head.aggregate_contributions_by_proposition(terms)
    np.savez_compressed(output_dir / "contributions" / "test_contributions.npz", case_id=case_ids, disease_id=np.asarray(head.disease_ids),
                        feature_id=np.asarray(head.feature_ids), base_proposition_id=np.asarray(head.base_proposition_ids), source=np.asarray(head.feature_sources),
                        feature_disease_id=np.asarray(head.feature_disease_ids), feature_category=np.asarray(head.feature_categories), feature_polarity=head.feature_polarities,
                        logits=logits, probabilities=outputs["probabilities"], top1_index=top1, top2_index=top2, margin=logits[rows, top1] - logits[rows, top2],
                        contribution_top1=terms[rows, top1], contribution_top2=terms[rows, top2], delta_contribution=terms[rows, top1] - terms[rows, top2],
                        proposition_id=np.asarray(proposition_ids), proposition_contributions=proposition_terms, bias=outputs["bias"], weights=outputs["weights"],
                        input_scores=outputs["input_scores"], support_scores=outputs["support_scores"])


def _write_audits(output_dir: Path, head: FrozenConceptLinearHead, outputs: dict[str, np.ndarray], threshold: float) -> None:
    _write_rows(output_dir / "diagnostics" / "concept_weights.csv", concept_weight_rows(outputs["weights"], head.disease_ids, head.feature_ids, head.feature_disease_ids, nonzero_threshold=threshold))
    _write_rows(output_dir / "diagnostics" / "disease_sparsity.csv", disease_sparsity_rows(outputs["weights"], head.disease_ids, head.feature_disease_ids, nonzero_threshold=threshold))
    fractions, summary = same_disease_contribution_fraction(outputs["input_scores"], outputs["weights"], head.disease_ids, head.feature_disease_ids)
    _write_rows(output_dir / "diagnostics" / "same_disease_contribution_fraction_test.csv", fractions)
    _write_json(output_dir / "diagnostics" / "same_disease_contribution_fraction_summary_test.json", summary)
    _write_json(output_dir / "diagnostics" / "contribution_reconstruction_check_test.json", audit_linear_contributions(outputs["logits"], outputs["contributions"], outputs["bias"]))


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"per_disease", "probabilities", "confusion_matrix"}}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda item: item.item() if isinstance(item, np.generic) else item), encoding="utf-8")


if __name__ == "__main__":
    main()

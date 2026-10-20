"""Run the isolated sklearn G-B-L2 Stage-2 replication gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from proposition.data.stage2_scores import build_stage2_support, load_stage2_disease_targets, load_stage2_propositions, load_stage2_score_artifact, validate_stage2_splits
from proposition.engine.stage2_concept_reference import enforce_reference_gate, run_reference_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate the historical standardized G-B-L2 scalar diagnostic probe.")
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}_scores", required=True)
        parser.add_argument(f"--{split}_csv", required=True)
    parser.add_argument("--active_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selected_C", type=float, default=0.1)
    parser.add_argument("--expected_val_top1", type=float)
    parser.add_argument("--expected_test_top1", type=float)
    parser.add_argument("--tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = {split: load_stage2_score_artifact(getattr(args, f"{split}_scores")) for split in ("train", "val", "test")}
    validate_stage2_splits(artifacts["train"], artifacts["val"], artifacts["test"])
    propositions = load_stage2_propositions(args.active_csv, artifacts["train"].proposition_ids)
    routed = {split: build_stage2_support(artifacts[split], propositions, "full_fused") for split in ("train", "val", "test")}
    targets = {split: load_stage2_disease_targets(getattr(args, f"{split}_csv"), artifacts[split].case_ids) for split in ("train", "val", "test")}
    disease_ids = targets["train"].disease_ids
    if any(targets[split].disease_ids != disease_ids for split in ("val", "test")):
        raise ValueError("disease label columns/order must match across train, val, and test CSVs")
    result = run_reference_probe(routed["train"].support_scores[:, routed["train"].proposition_mask], targets["train"].targets,
                                 routed["val"].support_scores[:, routed["val"].proposition_mask], targets["val"].targets,
                                 routed["test"].support_scores[:, routed["test"].proposition_mask], targets["test"].targets,
                                 disease_ids, seed=args.seed, selected_c=args.selected_C)
    if (args.expected_val_top1 is None) != (args.expected_test_top1 is None):
        raise ValueError("expected_val_top1 and expected_test_top1 must be supplied together")
    if args.expected_val_top1 is not None:
        enforce_reference_gate(result, expected_val_top1=args.expected_val_top1, expected_test_top1=args.expected_test_top1, tolerance=args.tolerance)
        result["gate_passed"] = True
    else:
        result["gate_passed"] = None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_metrics")}, sort_keys=True, default=_json_default))


def _json_default(value: Any) -> Any:
    try:
        return value.tolist()
    except AttributeError:
        return value.item()

if __name__ == "__main__":
    main()
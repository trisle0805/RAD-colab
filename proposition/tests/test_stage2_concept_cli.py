from __future__ import annotations

import csv
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from proposition.cli import stage2_concept


def _write_scores(path, case_ids: tuple[str, ...]) -> None:
    count = len(case_ids)
    image = np.tile(np.asarray([[0.9, 0.2]], dtype=np.float32), (count, 1))
    text = np.tile(np.asarray([[0.8, 0.3]], dtype=np.float32), (count, 1))
    np.savez(
        path,
        case_id=np.asarray(case_ids),
        proposition_id=np.asarray(("p1", "p2")),
        score_image=image,
        score_text=text,
        score_fused=np.column_stack(((image[:, 0] + text[:, 0]) / 2, text[:, 1])),
        image_trainable=np.asarray((True, False)),
        text_trainable=np.asarray((True, True)),
        active_scoring=np.asarray((True, True)),
    )

def _write_distinct_scores(path, case_ids: tuple[str, ...]) -> None:
    count = len(case_ids)
    image = np.tile(np.asarray([[0.2, 0.9]], dtype=np.float32), (count, 1))
    text = np.tile(np.asarray([[0.1, 0.8]], dtype=np.float32), (count, 1))
    np.savez(
        path,
        case_id=np.asarray(case_ids),
        proposition_id=np.asarray(("p1", "p2")),
        score_image=image,
        score_text=text,
        score_fused=np.column_stack(((image[:, 0] + text[:, 0]) / 2, text[:, 1])),
        image_trainable=np.asarray((True, False)),
        text_trainable=np.asarray((True, True)),
        active_scoring=np.asarray((True, True)),
    )


def _write_labels(path, case_ids: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "label_d1", "label_d2"))
        writer.writeheader()
        for index, case_id in enumerate(case_ids):
            writer.writerow({"case_id": case_id, "label_d1": int(index % 2 == 0), "label_d2": int(index % 2 == 1)})


def _write_active(path) -> None:
    path.write_text(
        "proposition_id,disease_id,category,polarity,text_trainable,image_trainable,active_scoring\n"
        "p1,d1,morphology,1,1,1,1\n"
        "p2,d2,other,1,1,0,1\n",
        encoding="utf-8",
    )


def _arguments(tmp_path, output_dir, *, evaluate_test: bool) -> list[str]:
    args = [
        "stage2_concept", "--train_scores", str(tmp_path / "train.npz"), "--train_csv", str(tmp_path / "train.csv"),
        "--val_scores", str(tmp_path / "val.npz"), "--val_csv", str(tmp_path / "val.csv"),
        "--active_csv", str(tmp_path / "active.csv"), "--output_dir", str(output_dir), "--head_mode", "global",
        "--routing_mode", "full_fused", "--epochs", "1", "--patience", "1", "--learning_rate", "0.01", "--batch_size", "2",
    ]
    if evaluate_test:
        args.extend(("--evaluate_test", "--test_scores", str(tmp_path / "test.npz"), "--test_csv", str(tmp_path / "test.csv")))
    else:
        args.extend(("--test_scores", str(tmp_path / "missing-test.npz"), "--test_csv", str(tmp_path / "missing-test.csv")))
    return args


def test_default_cli_does_not_access_or_export_test_artifacts(tmp_path, monkeypatch):
    _write_scores(tmp_path / "train.npz", ("train-a", "train-b"))
    _write_scores(tmp_path / "val.npz", ("val-a", "val-b"))
    _write_labels(tmp_path / "train.csv", ("train-a", "train-b"))
    _write_labels(tmp_path / "val.csv", ("val-a", "val-b"))
    _write_active(tmp_path / "active.csv")
    output_dir = tmp_path / "validation-only"

    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, output_dir, evaluate_test=False))
    stage2_concept.main()

    assert (output_dir / "metrics" / "val_summary.json").is_file()
    assert not (output_dir / "metrics" / "test_summary.json").exists()
    assert not (output_dir / "metrics" / "test_confidence_intervals.json").exists()
    assert not (output_dir / "predictions" / "test_best.npz").exists()
    assert not (output_dir / "contributions" / "test_contributions.npz").exists()


def test_cli_evaluates_test_only_when_explicitly_enabled(tmp_path, monkeypatch):
    for split in ("train", "val"):
        case_ids = (f"{split}-a", f"{split}-b")
        _write_scores(tmp_path / f"{split}.npz", case_ids)
        _write_labels(tmp_path / f"{split}.csv", case_ids)
    test_case_ids = ("test-a", "test-b")
    _write_distinct_scores(tmp_path / "test.npz", test_case_ids)
    _write_labels(tmp_path / "test.csv", test_case_ids)
    _write_active(tmp_path / "active.csv")
    output_dir = tmp_path / "with-test"

    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, output_dir, evaluate_test=True))
    stage2_concept.main()

    assert (output_dir / "metrics" / "test_summary.json").is_file()
    assert (output_dir / "metrics" / "test_confidence_intervals.json").is_file()
    assert (output_dir / "predictions" / "test_best.npz").is_file()
    assert (output_dir / "contributions" / "test_contributions.npz").is_file()
    assert (output_dir / "diagnostics" / "same_disease_contribution_fraction_val.csv").is_file()
    assert (output_dir / "diagnostics" / "same_disease_contribution_fraction_test.csv").is_file()
    assert (output_dir / "diagnostics" / "same_disease_contribution_fraction_summary_test.json").is_file()
    assert (output_dir / "diagnostics" / "contribution_reconstruction_check_test.json").is_file()
    test_artifact = np.load(output_dir / "contributions" / "test_contributions.npz")
    expected_fractions = stage2_concept.same_disease_contribution_fraction(
        test_artifact["input_scores"], test_artifact["weights"], ("d1", "d2"), ("d1", "d2"),
    )[0]
    with (output_dir / "diagnostics" / "same_disease_contribution_fraction_test.csv").open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert [float(row["same_disease_contribution_fraction"]) for row in audit_rows] == pytest.approx(
        [row["same_disease_contribution_fraction"] for row in expected_fractions]
    )
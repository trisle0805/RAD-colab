"""Metrics, prediction artifacts, and resumable checkpoint utilities."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from engine.train_proposition import EpochResult


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def multiclass_metrics(
    gt: NDArray[np.floating],
    pred: NDArray[np.floating],
    topk: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    """Compute deterministic single-label rank metrics used by SkinCAP."""

    if gt.ndim != 2 or pred.shape != gt.shape or gt.shape[0] == 0:
        raise ValueError("gt and pred must be matching non-empty (N, C) arrays")
    topk = tuple(sorted(set(int(k) for k in topk)))
    if not topk or topk[0] < 1 or topk[-1] > pred.shape[1]:
        raise ValueError("topk must be positive and not exceed the class count")

    targets = np.argmax(gt, axis=1)
    true_scores = pred[np.arange(pred.shape[0]), targets]
    ranks = 1 + np.sum(pred > true_scores[:, None], axis=1)
    predicted = np.argmax(pred, axis=1)
    result = {
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
    }
    for k in topk:
        result[f"top{k}_accuracy"] = float(np.mean(ranks <= k))
        if k >= 2:
            reciprocal = np.where(ranks <= k, 1.0 / ranks, 0.0)
            result[f"mrr_at_{k}"] = float(np.mean(reciprocal))
            result[f"macro_mrr_at_{k}"] = float(
                np.mean([np.mean(reciprocal[targets == c]) for c in np.unique(targets)])
            )
    return result


def final_metrics(
    gt: NDArray[np.floating],
    pred: NDArray[np.floating],
    topk: Sequence[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Compute rank metrics and RAD-compatible aggregate multilabel metrics."""

    result: dict[str, Any] = multiclass_metrics(gt, pred, topk)
    predicted_binary = np.zeros_like(gt, dtype=np.int64)
    predicted_binary[np.arange(gt.shape[0]), np.argmax(pred, axis=1)] = 1
    per_class_auc = []
    per_class_ap = []
    for class_index in range(gt.shape[1]):
        target = gt[:, class_index]
        per_class_ap.append(float(average_precision_score(target, pred[:, class_index])))
        per_class_auc.append(
            float(roc_auc_score(target, pred[:, class_index]))
            if np.unique(target).size == 2
            else float("nan")
        )
    result.update(
        {
            "macro_auc": float(np.nanmean(per_class_auc)),
            "mAP": float(np.mean(per_class_ap)),
            "rad_legacy_metrics": {
                "mean_auc": float(np.nanmean(per_class_auc)),
                "mean_ap": float(np.mean(per_class_ap)),
                "mean_accuracy": float(
                    np.mean([
                        accuracy_score(gt[:, c], predicted_binary[:, c])
                        for c in range(gt.shape[1])
                    ])
                ),
                "mean_f1": float(f1_score(gt, predicted_binary, average="macro", zero_division=0)),
                "mean_precision": float(
                    precision_score(gt, predicted_binary, average="macro", zero_division=0)
                ),
                "mean_recall": float(
                    recall_score(gt, predicted_binary, average="macro", zero_division=0)
                ),
                "subset_accuracy": float(accuracy_score(gt, predicted_binary)),
            },
        }
    )
    return result


def save_validation_artifacts(
    result: EpochResult,
    output_dir: str | Path,
    epoch: int,
    *,
    topk: Sequence[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Persist validation predictions and return prefixed epoch metrics."""

    if epoch <= 0:
        raise ValueError("epoch must be one-based and positive")
    output_dir = Path(output_dir)
    gt = result.gt.numpy()
    proposition = result.pred_proposition.numpy()
    arrays: dict[str, Any] = {"gt": gt, "pred_proposition": proposition}
    if result.pred_label is not None:
        arrays["pred_label"] = result.pred_label.numpy()
    relative_prediction = Path("predictions") / f"val_epoch_{epoch:03d}.npz"
    _atomic_npz(output_dir / relative_prediction, **arrays)

    proposition_metrics = final_metrics(gt, proposition, topk)
    summary: dict[str, Any] = {
        "epoch": epoch,
        "prediction_file": str(relative_prediction).replace("\\", "/"),
        **{f"loss_{name}": value for name, value in result.losses.items()},
        **{f"proposition_{name}": value for name, value in proposition_metrics.items() if name != "rad_legacy_metrics"},
    }
    for name, value in proposition_metrics["rad_legacy_metrics"].items():
        summary[f"proposition_{name}"] = value
    if result.pred_label is not None:
        label_metrics = final_metrics(gt, result.pred_label.numpy(), topk)
        summary.update(
            {f"label_{name}": value for name, value in label_metrics.items() if name != "rad_legacy_metrics"}
        )
        for name, value in label_metrics["rad_legacy_metrics"].items():
            summary[f"label_{name}"] = value
    summary["validation_score"] = proposition_metrics["macro_f1"]
    return summary


def append_metrics_history(output_dir: str | Path, record: Mapping[str, Any]) -> None:
    """Append one flat validation record and atomically update latest JSON."""

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history = metrics_dir / "metrics_history.csv"
    write_header = not history.exists() or history.stat().st_size == 0
    with history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_json(metrics_dir / "latest_metrics.json", dict(record))


def save_final_metrics(
    result: EpochResult,
    output_dir: str | Path,
    *,
    best_epoch: int,
    best_validation_score: float,
    topk: Sequence[int] = (1, 2, 3),
) -> dict[str, Any]:
    """Persist test predictions and final metrics without branch fusion."""

    gt = result.gt.numpy()
    proposition = result.pred_proposition.numpy()
    arrays: dict[str, Any] = {"gt": gt, "pred_proposition": proposition}
    payload: dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_validation_score),
        "proposition": final_metrics(gt, proposition, topk),
    }
    if result.pred_label is not None:
        label = result.pred_label.numpy()
        arrays["pred_label"] = label
        payload["label_aux"] = final_metrics(gt, label, topk)
    output_dir = Path(output_dir)
    _atomic_npz(output_dir / "predictions" / "test_best.npz", **arrays)
    _atomic_json(output_dir / "final_metrics.json", payload)
    return payload


def capture_rng_state(pecl_generator: torch.Generator) -> dict[str, Any]:
    """Capture Python, NumPy, torch CPU/CUDA, and PECL generator states."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "pecl_generator": pecl_generator.get_state(),
    }


def restore_rng_state(state: Mapping[str, Any], pecl_generator: torch.Generator) -> None:
    """Restore all RNG streams captured by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda", "pecl_generator"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    pecl_generator.set_state(state["pecl_generator"])


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> None:
    """Atomically save a checkpoint mapping."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)

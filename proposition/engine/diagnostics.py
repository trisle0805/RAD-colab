"""Disease-label-free diagnostics for Stage-1 proposition evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import _array


def within_patient_differentiation(scores: Any) -> dict[str, Any]:
    """Report per-patient proposition score spread and its aggregate summary."""
    values = _array(scores).astype(float)
    if values.ndim != 2:
        raise ValueError("scores must have shape [case, proposition]")
    row_std = np.nanstd(values, axis=1)
    return {"per_patient_std": row_std.tolist(), "mean_within_patient_std": float(np.nanmean(row_std)) if np.isfinite(row_std).any() else None}


def proposition_correlation(scores: Any, *, near_identical: float = 0.99) -> dict[str, Any]:
    """Return pairwise proposition correlations and near-identical pair fraction."""
    values = _array(scores).astype(float)
    if values.ndim != 2 or not 0 <= near_identical <= 1:
        raise ValueError("scores must be [case, proposition] and near_identical must be in [0, 1]")
    count = values.shape[1]
    correlation = np.full((count, count), np.nan)
    pairs: list[dict[str, Any]] = []
    for left in range(count):
        correlation[left, left] = 1.0
        for right in range(left + 1, count):
            valid = np.isfinite(values[:, left]) & np.isfinite(values[:, right])
            if valid.sum() > 1 and np.std(values[valid, left]) > 0 and np.std(values[valid, right]) > 0:
                value = float(np.corrcoef(values[valid, left], values[valid, right])[0, 1])
            else:
                value = None
            correlation[left, right] = correlation[right, left] = np.nan if value is None else value
            pairs.append({"left": left, "right": right, "correlation": value, "near_identical": bool(value is not None and abs(value) >= near_identical)})
    usable = [pair for pair in pairs if pair["correlation"] is not None]
    return {"correlation": correlation.tolist(), "pairs": pairs, "near_identical_fraction": (sum(pair["near_identical"] for pair in usable) / len(usable)) if usable else None}


def _invoke(model: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return model(*args, **kwargs)


def query_swap(model: Callable[..., Any], batch: Any, queries: Any, *, swap: Sequence[int] | None = None, call_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare model outputs under original and deterministically swapped queries."""
    query_array = _array(queries)
    order = np.asarray(swap if swap is not None else np.arange(len(query_array))[::-1], dtype=int)
    if order.shape != (len(query_array),) or np.any(order < 0) or np.any(order >= len(query_array)):
        raise ValueError("swap must be a permutation-sized sequence of valid query indices")
    kwargs = dict(call_kwargs or {})
    original = _invoke(model, batch, queries, **kwargs)
    swapped = _invoke(model, batch, query_array[order], **kwargs)
    return {"swap_indices": order.tolist(), "original": original, "swapped": swapped}


def query_ablation(model: Callable[..., Any], batch: Any, queries: Any, *, mode: str, seed: int = 0, call_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate zero, mean, or seeded-random query ablations through a callable model."""
    query_array = _array(queries)
    if mode == "zero":
        ablated = np.zeros_like(query_array)
    elif mode == "mean":
        ablated = np.broadcast_to(query_array.mean(axis=0, keepdims=True), query_array.shape).copy()
    elif mode == "random":
        ablated = np.random.default_rng(seed).standard_normal(query_array.shape).astype(query_array.dtype, copy=False)
    else:
        raise ValueError("mode must be 'zero', 'mean', or 'random'")
    kwargs = dict(call_kwargs or {})
    return {"mode": mode, "seed": seed if mode == "random" else None, "baseline": _invoke(model, batch, queries, **kwargs), "ablated": _invoke(model, batch, ablated, **kwargs)}


def save_attention_artifact(attention: Any, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Persist attention deterministically as a compressed NPZ without pickle data."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"attention": _array(attention)}
    if metadata:
        for key in sorted(metadata):
            payload[f"metadata_{key}"] = np.asarray(metadata[key])
    np.savez_compressed(destination, **payload)
    return destination

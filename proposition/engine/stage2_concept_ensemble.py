"""Frozen effective-weight artifacts for Round-A scalar concept-head averaging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from scipy.stats import spearmanr

from proposition.data.stage2_concept_features import Stage2ConceptFeature, aggregate_feature_contributions_by_proposition


@dataclass(frozen=True)
class FrozenConceptLinearHead:
    """Immutable effective-weight linear head for evaluation and explanations."""

    feature_ids: tuple[str, ...]
    base_proposition_ids: tuple[str, ...]
    feature_disease_ids: tuple[str, ...]
    feature_categories: tuple[str, ...]
    feature_polarities: np.ndarray
    feature_sources: tuple[str, ...]
    disease_ids: tuple[str, ...]
    effective_weights: np.ndarray
    bias: np.ndarray
    head_mode: Literal["local", "global"]
    routing_mode: str
    regularization: str
    lambda_reg: float
    refit_epochs: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        count = len(self.feature_ids)
        if not count or len(set(self.feature_ids)) != count:
            raise ValueError("frozen head needs unique non-empty feature IDs")
        if any(len(values) != count for values in (self.base_proposition_ids, self.feature_disease_ids, self.feature_categories, self.feature_polarities, self.feature_sources)):
            raise ValueError("frozen feature metadata must align")
        if len(set(self.disease_ids)) != len(self.disease_ids) or len(self.disease_ids) < 2:
            raise ValueError("frozen head needs unique disease IDs")
        weights = np.asarray(self.effective_weights, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if weights.shape != (len(self.disease_ids), count) or bias.shape != (len(self.disease_ids),):
            raise ValueError("frozen weight/bias shape mismatches metadata")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("frozen weights and bias must be finite")
        if self.head_mode not in {"local", "global"}:
            raise ValueError("frozen head_mode must be local or global")
        if self.head_mode == "local" and np.any(weights < -1e-10):
            raise ValueError("local frozen effective weights must remain nonnegative")
        if self.head_mode == "local":
            feature_owner = np.asarray([self.disease_ids.index(disease) for disease in self.feature_disease_ids])
            local_mask = np.equal(np.arange(len(self.disease_ids))[:, None], feature_owner[None, :])
            if np.any(np.abs(weights[~local_mask]) > 1e-10):
                raise ValueError("local frozen effective weights must be zero outside feature-owning disease")
        object.__setattr__(self, "effective_weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "feature_polarities", np.asarray(self.feature_polarities, dtype=np.int8))

    def contribution_outputs(self, support_scores: Any) -> dict[str, np.ndarray]:
        """Run immutable scalar-only inference and exact feature contributions."""
        raw = np.asarray(support_scores, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.feature_ids) or not np.all(np.isfinite(raw)):
            raise ValueError("support_scores must be finite [case, feature] values")
        if np.any(raw < -1e-6) or np.any(raw > 1.0 + 1e-6):
            raise ValueError("support_scores must lie in [0, 1]")
        input_scores = np.clip(raw, 0.0, 1.0)
        if self.head_mode == "local":
            input_scores = np.where(self.feature_polarities[None, :] > 0, input_scores, 1.0 - input_scores)
        contributions = input_scores[:, None, :] * self.effective_weights[None, :, :]
        logits = self.bias[None, :] + contributions.sum(axis=2, dtype=np.float64)
        return {
            "support_scores": raw,
            "input_scores": input_scores,
            "weights": self.effective_weights,
            "bias": self.bias,
            "contributions": contributions,
            "reconstructed_logits": logits,
            "logits": logits,
            "probabilities": _softmax(logits),
        }

    def aggregate_contributions_by_proposition(self, contributions: Any) -> tuple[np.ndarray, tuple[str, ...]]:
        features = tuple(
            Stage2ConceptFeature(identifier, base, disease, category, int(polarity), source)
            for identifier, base, disease, category, polarity, source in zip(self.feature_ids, self.base_proposition_ids, self.feature_disease_ids, self.feature_categories, self.feature_polarities, self.feature_sources)
        )
        return aggregate_feature_contributions_by_proposition(np.asarray(contributions), features)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target,
            feature_id=np.asarray(self.feature_ids), base_proposition_id=np.asarray(self.base_proposition_ids), feature_disease_id=np.asarray(self.feature_disease_ids), feature_category=np.asarray(self.feature_categories),
            feature_polarity=self.feature_polarities, source=np.asarray(self.feature_sources), disease_id=np.asarray(self.disease_ids),
            weights=self.effective_weights, bias=self.bias, head_mode=np.asarray(self.head_mode), routing_mode=np.asarray(self.routing_mode),
            regularization=np.asarray(self.regularization), lambda_reg=np.asarray(self.lambda_reg), refit_epochs=np.asarray(self.refit_epochs), seeds=np.asarray(self.seeds, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrozenConceptLinearHead":
        source = Path(path)
        required = {"feature_id", "base_proposition_id", "feature_disease_id", "feature_category", "feature_polarity", "source", "disease_id", "weights", "bias", "head_mode", "routing_mode", "regularization", "lambda_reg", "refit_epochs", "seeds"}
        with np.load(source, allow_pickle=False) as payload:
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"{source}: missing frozen-head fields {sorted(missing)!r}")
            values = {name: np.asarray(payload[name]) for name in required}
        return cls(
            feature_ids=tuple(values["feature_id"].astype(str).tolist()), base_proposition_ids=tuple(values["base_proposition_id"].astype(str).tolist()),
            feature_disease_ids=tuple(values["feature_disease_id"].astype(str).tolist()), feature_categories=tuple(values["feature_category"].astype(str).tolist()), feature_polarities=values["feature_polarity"],
            feature_sources=tuple(values["source"].astype(str).tolist()), disease_ids=tuple(values["disease_id"].astype(str).tolist()),
            effective_weights=values["weights"], bias=values["bias"], head_mode=str(values["head_mode"].item()), routing_mode=str(values["routing_mode"].item()),
            regularization=str(values["regularization"].item()), lambda_reg=float(values["lambda_reg"].item()), refit_epochs=int(values["refit_epochs"].item()),
            seeds=tuple(int(value) for value in values["seeds"].tolist()),
        )


def derive_refit_epochs(best_epochs: Sequence[int]) -> int:
    """Return the documented half-up median epoch for an odd number of folds."""
    values = np.sort(np.asarray(best_epochs, dtype=np.int64))
    if values.ndim != 1 or not len(values) or np.any(values < 1):
        raise ValueError("best epochs must be a non-empty sequence of positive integers")
    # CV defaults to five folds; for even folds this rounds .5 upward rather than
    # relying on Python's banker rounding.
    median = float(np.median(values))
    return int(np.floor(median + 0.5))


def average_frozen_heads(heads: Sequence[FrozenConceptLinearHead]) -> FrozenConceptLinearHead:
    """Average effective weights/biases while preserving one linear frozen head."""
    if not heads:
        raise ValueError("at least one frozen head is required")
    first = heads[0]
    for head in heads[1:]:
        for name in ("feature_ids", "base_proposition_ids", "feature_disease_ids", "feature_categories", "feature_sources", "disease_ids", "head_mode", "routing_mode", "regularization", "lambda_reg", "refit_epochs"):
            if getattr(head, name) != getattr(first, name):
                raise ValueError(f"cannot average heads with different {name}")
        if not np.array_equal(head.feature_polarities, first.feature_polarities):
            raise ValueError("cannot average heads with different feature polarities")
    seeds = tuple(seed for head in heads for seed in head.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("averaged heads must have distinct seed identities")
    return FrozenConceptLinearHead(
        feature_ids=first.feature_ids, base_proposition_ids=first.base_proposition_ids, feature_disease_ids=first.feature_disease_ids, feature_categories=first.feature_categories,
        feature_polarities=first.feature_polarities, feature_sources=first.feature_sources, disease_ids=first.disease_ids,
        effective_weights=np.mean([head.effective_weights for head in heads], axis=0), bias=np.mean([head.bias for head in heads], axis=0),
        head_mode=first.head_mode, routing_mode=first.routing_mode, regularization=first.regularization,
        lambda_reg=first.lambda_reg, refit_epochs=first.refit_epochs, seeds=seeds,
    )


def stability_rows(
    heads: Sequence[FrozenConceptLinearHead], averaged: FrozenConceptLinearHead, support_scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Report pairwise effective-weight and bias stability without test data."""
    rows: list[dict[str, Any]] = []
    labels = ["seed_" + "_".join(map(str, head.seeds)) for head in heads]
    all_heads = list(heads) + [averaged]
    all_labels = labels + ["averaged"]
    for left_index, left in enumerate(all_heads):
        for right_index in range(left_index + 1, len(all_heads)):
            right = all_heads[right_index]
            a, b = left.effective_weights.ravel(), right.effective_weights.ravel()
            correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
            spearman = float(spearmanr(a, b).statistic) if np.std(a) and np.std(b) else None
            top1_agreement = None
            if support_scores is not None:
                left_top1 = left.contribution_outputs(support_scores)["logits"].argmax(axis=1)
                right_top1 = right.contribution_outputs(support_scores)["logits"].argmax(axis=1)
                top1_agreement = float(np.mean(left_top1 == right_top1))
            rows.append({"left": all_labels[left_index], "right": all_labels[right_index], "pearson_effective_weight": correlation,
                         "spearman_effective_weight": spearman, "top1_agreement_train_val": top1_agreement,
                         "mean_abs_weight_difference": float(np.abs(a - b).mean()), "max_abs_weight_difference": float(np.abs(a - b).max()),
                         "mean_abs_bias_difference": float(np.abs(left.bias - right.bias).mean()), "max_abs_bias_difference": float(np.abs(left.bias - right.bias).max())})
    return rows


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)

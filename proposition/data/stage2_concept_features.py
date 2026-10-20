"""Expanded, scalar-only concept-feature routing for Round-A Stage-2 heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from proposition.data.stage2_scores import Stage2Proposition, Stage2ScoreArtifacts

ConceptFeatureRoutingMode = Literal[
    "image_morphology",
    "full_fused",
    "full_image_morphology_text_other",
    "full_dual_morphology",
]

_CONCEPT_FEATURE_ROUTING_MODES = frozenset((
    "image_morphology",
    "full_fused",
    "full_image_morphology_text_other",
    "full_dual_morphology",
))


@dataclass(frozen=True)
class Stage2ConceptFeature:
    """One auditable scalar coordinate consumed by a Stage-2 concept head."""

    feature_id: str
    base_proposition_id: str
    disease_id: str
    category: str
    polarity: int
    source: Literal["image", "text", "fused"]


@dataclass(frozen=True)
class RoutedConceptFeatures:
    """Expanded scalar feature matrix and its immutable column metadata."""

    case_ids: np.ndarray
    support_scores: np.ndarray
    features: tuple[Stage2ConceptFeature, ...]
    routing_mode: ConceptFeatureRoutingMode

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return tuple(feature.feature_id for feature in self.features)


def build_stage2_concept_features(
    artifacts: Stage2ScoreArtifacts,
    propositions: Sequence[Stage2Proposition],
    routing_mode: ConceptFeatureRoutingMode,
) -> RoutedConceptFeatures:
    """Build scalar-only features without assuming one feature per proposition.

    Columns preserve frozen proposition order; where a proposition expands, channels
    are emitted in stable ``image``, then ``text`` order.  No values besides the
    frozen image/text/fused scalar score matrices enter this function.
    """
    if routing_mode not in _CONCEPT_FEATURE_ROUTING_MODES:
        raise ValueError(f"unsupported Stage-2 concept-feature routing mode {routing_mode!r}")
    _validate_metadata(artifacts, propositions)

    feature_columns: list[np.ndarray] = []
    features: list[Stage2ConceptFeature] = []
    for index, proposition in enumerate(propositions):
        if not proposition.active_scoring:
            continue
        image, text = proposition.image_trainable, proposition.text_trainable
        morphology = proposition.category == "morphology"
        selected: list[tuple[Literal["image", "text", "fused"], np.ndarray]] = []
        if routing_mode == "image_morphology":
            if morphology and image:
                selected.append(("image", artifacts.score_image[:, index]))
        elif routing_mode == "full_fused":
            selected.extend(_full_fused_columns(artifacts, index, image=image, text=text))
        elif routing_mode == "full_image_morphology_text_other":
            if morphology and image:
                selected.append(("image", artifacts.score_image[:, index]))
            elif text:
                selected.append(("text", artifacts.score_text[:, index]))
            elif image:
                selected.append(("image", artifacts.score_image[:, index]))
        else:  # full_dual_morphology
            if morphology and image and text:
                selected.extend((("image", artifacts.score_image[:, index]), ("text", artifacts.score_text[:, index])))
            else:
                selected.extend(_full_fused_columns(artifacts, index, image=image, text=text))
        for source, values in selected:
            feature_columns.append(np.asarray(values, dtype=np.float32))
            features.append(Stage2ConceptFeature(
                feature_id=f"{proposition.proposition_id}::{source}",
                base_proposition_id=proposition.proposition_id,
                disease_id=proposition.disease_id,
                category=proposition.category,
                polarity=proposition.polarity,
                source=source,
            ))

    if not features:
        raise ValueError("selected routing produced no active scalar concept features")
    feature_ids = [feature.feature_id for feature in features]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("routing produced duplicate concept feature IDs")
    support = np.stack(feature_columns, axis=1)
    if not np.all(np.isfinite(support)) or np.any(support < -1e-6) or np.any(support > 1.0 + 1e-6):
        raise ValueError("routed concept features must be finite scores in [0, 1]")
    return RoutedConceptFeatures(
        case_ids=np.asarray(artifacts.case_ids, dtype=str),
        support_scores=np.clip(support, 0.0, 1.0),
        features=tuple(features),
        routing_mode=routing_mode,
    )


def aggregate_feature_contributions_by_proposition(
    contributions: np.ndarray,
    features: Sequence[Stage2ConceptFeature],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Sum trailing feature-axis contributions into stable base-proposition terms."""
    values = np.asarray(contributions)
    if values.ndim < 1 or values.shape[-1] != len(features):
        raise ValueError("contributions must end with the routed concept-feature axis")
    proposition_ids = tuple(dict.fromkeys(feature.base_proposition_id for feature in features))
    lookup = {identifier: index for index, identifier in enumerate(proposition_ids)}
    result = np.zeros((*values.shape[:-1], len(proposition_ids)), dtype=values.dtype)
    for feature_index, feature in enumerate(features):
        result[..., lookup[feature.base_proposition_id]] += values[..., feature_index]
    return result, proposition_ids


def _full_fused_columns(
    artifacts: Stage2ScoreArtifacts,
    index: int,
    *,
    image: bool,
    text: bool,
) -> list[tuple[Literal["image", "text", "fused"], np.ndarray]]:
    if image and text:
        values = (artifacts.score_image[:, index] + artifacts.score_text[:, index]) / 2.0
        if artifacts.score_fused is not None and not np.allclose(values, artifacts.score_fused[:, index], rtol=1e-5, atol=1e-6):
            raise ValueError("stored score_fused does not match reconstructed Stage-1 fusion")
        return [("fused", values)]
    if text:
        return [("text", artifacts.score_text[:, index])]
    if image:
        return [("image", artifacts.score_image[:, index])]
    return []


def _validate_metadata(artifacts: Stage2ScoreArtifacts, propositions: Sequence[Stage2Proposition]) -> None:
    if len(propositions) != len(artifacts.proposition_ids):
        raise ValueError("proposition metadata length differs from score artifact")
    identifiers = np.asarray([item.proposition_id for item in propositions], dtype=str)
    if not np.array_equal(identifiers, artifacts.proposition_ids):
        raise ValueError("proposition metadata order differs from score artifact")
    for name in ("image_trainable", "text_trainable", "active_scoring"):
        expected = np.asarray([getattr(item, name) for item in propositions], dtype=bool)
        if not np.array_equal(expected, getattr(artifacts, name)):
            raise ValueError(f"metadata {name} disagrees with score artifact")

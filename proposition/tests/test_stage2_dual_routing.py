from __future__ import annotations

import numpy as np
import pytest

from proposition.data.stage2_concept_features import (
    aggregate_feature_contributions_by_proposition,
    build_stage2_concept_features,
)
from proposition.data.stage2_scores import Stage2Proposition, Stage2ScoreArtifacts
from proposition.engine.stage2_concept_audit import audit_linear_contributions


def _artifact() -> Stage2ScoreArtifacts:
    return Stage2ScoreArtifacts(
        case_ids=np.asarray(("case-1", "case-2")),
        proposition_ids=np.asarray(("morph-both", "text-only", "image-only", "other-both")),
        score_image=np.asarray(((0.1, 0.0, 0.3, 0.8), (0.2, 0.0, 0.4, 0.6)), dtype=np.float32),
        score_text=np.asarray(((0.9, 0.5, 0.0, 0.2), (0.7, 0.4, 0.0, 0.4)), dtype=np.float32),
        score_fused=np.asarray(((0.5, 0.5, 0.3, 0.5), (0.45, 0.4, 0.4, 0.5)), dtype=np.float32),
        image_trainable=np.asarray((True, False, True, True)),
        text_trainable=np.asarray((True, True, False, True)),
        active_scoring=np.asarray((True, True, True, True)),
    )


def _propositions() -> tuple[Stage2Proposition, ...]:
    return (
        Stage2Proposition("morph-both", "d1", "morphology", 1, True, True, True),
        Stage2Proposition("text-only", "d1", "other", -1, False, True, True),
        Stage2Proposition("image-only", "d2", "other", 1, True, False, True),
        Stage2Proposition("other-both", "d2", "other", -1, True, True, True),
    )


def test_dual_morphology_routing_expands_only_morphology_dual_channels():
    routed = build_stage2_concept_features(_artifact(), _propositions(), "full_dual_morphology")

    assert routed.feature_ids == (
        "morph-both::image", "morph-both::text", "text-only::text",
        "image-only::image", "other-both::fused",
    )
    assert [item.base_proposition_id for item in routed.features] == [
        "morph-both", "morph-both", "text-only", "image-only", "other-both",
    ]
    assert [item.polarity for item in routed.features] == [1, 1, -1, 1, -1]
    np.testing.assert_allclose(routed.support_scores, ((0.1, 0.9, 0.5, 0.3, 0.5), (0.2, 0.7, 0.4, 0.4, 0.5)))


def test_channel_contributions_aggregate_by_base_proposition_and_reconstruct():
    routed = build_stage2_concept_features(_artifact(), _propositions(), "full_dual_morphology")
    terms = np.asarray([[[0.4, 0.27, 0.1, -0.2, 0.3], [0.2, 0.1, 0.0, 0.4, -0.1]]], dtype=np.float32)

    aggregated, proposition_ids = aggregate_feature_contributions_by_proposition(terms, routed.features)

    assert proposition_ids == ("morph-both", "text-only", "image-only", "other-both")
    np.testing.assert_allclose(aggregated, [[[0.67, 0.1, -0.2, 0.3], [0.3, 0.0, 0.4, -0.1]]])
    np.testing.assert_allclose(aggregated.sum(axis=-1), terms.sum(axis=-1))


def test_float64_contribution_audit_uses_documented_tolerance():
    contributions = np.asarray([[[0.1, 0.2], [0.3, -0.1]]], dtype=np.float32)
    bias = np.asarray((0.5, -0.2), dtype=np.float32)
    logits = bias[None, :] + contributions.sum(axis=2)

    audit = audit_linear_contributions(logits, contributions, bias)

    assert audit["verification_tolerance"] == pytest.approx(1e-4)
    assert audit["passes"] is True

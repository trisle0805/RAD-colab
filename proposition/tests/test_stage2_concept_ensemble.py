from __future__ import annotations

import numpy as np
import pytest

from proposition.engine.stage2_concept_ensemble import FrozenConceptLinearHead, average_frozen_heads, derive_refit_epochs


def _head(*, seed: int, weights: np.ndarray, bias: np.ndarray, mode: str = "global") -> FrozenConceptLinearHead:
    return FrozenConceptLinearHead(
        feature_ids=("p1::image", "p1::text"), base_proposition_ids=("p1", "p1"), feature_disease_ids=("d1", "d1"),
        feature_categories=("morphology", "morphology"), feature_polarities=np.asarray((1, -1)), feature_sources=("image", "text"), disease_ids=("d1", "d2"),
        effective_weights=weights, bias=bias, head_mode=mode, routing_mode="full_dual_morphology", regularization="l2",
        lambda_reg=.1, refit_epochs=7, seeds=(seed,),
    )


def test_averaged_frozen_head_equals_arithmetic_mean_of_seed_logits_and_contributions(tmp_path):
    first = _head(seed=42, weights=np.asarray(((1.0, -2.0), (.5, .25))), bias=np.asarray((.1, -.3)))
    second = _head(seed=43, weights=np.asarray(((3.0, 2.0), (1.5, -.75))), bias=np.asarray((-.1, .5)))
    averaged = average_frozen_heads((first, second))
    scores = np.asarray(((.2, .8), (.6, .1)))

    seed_logits = np.mean((first.contribution_outputs(scores)["logits"], second.contribution_outputs(scores)["logits"]), axis=0)
    outputs = averaged.contribution_outputs(scores)

    np.testing.assert_allclose(outputs["logits"], seed_logits)
    np.testing.assert_allclose(outputs["reconstructed_logits"], outputs["logits"])
    np.testing.assert_allclose(averaged.effective_weights, (first.effective_weights + second.effective_weights) / 2)
    aggregated, identifiers = averaged.aggregate_contributions_by_proposition(outputs["contributions"])
    assert identifiers == ("p1",)
    np.testing.assert_allclose(aggregated[..., 0], outputs["contributions"].sum(axis=-1))
    path = tmp_path / "averaged.npz"
    averaged.save(path)
    loaded = FrozenConceptLinearHead.load(path)
    np.testing.assert_allclose(loaded.contribution_outputs(scores)["logits"], outputs["logits"])


def test_local_average_remains_nonnegative_and_refit_epoch_rounding_is_explicit():
    first = _head(seed=42, weights=np.asarray(((1.0, 2.0), (0.0, 0.0))), bias=np.asarray((.1, -.3)), mode="local")
    second = _head(seed=43, weights=np.asarray(((3.0, .2), (0.0, 0.0))), bias=np.asarray((-.1, .5)), mode="local")
    assert np.all(average_frozen_heads((first, second)).effective_weights >= 0)
    assert derive_refit_epochs((1, 2, 4, 5)) == 3
    with pytest.raises(ValueError, match="positive"):
        derive_refit_epochs((0,))
    with pytest.raises(ValueError, match="outside feature-owning disease"):
        _head(seed=44, weights=np.asarray(((1.0, 2.0), (.1, 0.0))), bias=np.asarray((0.0, 0.0)), mode="local")

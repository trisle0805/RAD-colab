"""Train fixed-epoch scalar concept heads and freeze their effective weights."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from proposition.data.stage2_concept_features import RoutedConceptFeatures
from proposition.data.stage2_scores import Stage2Dataset
from proposition.engine.stage2_concept_cv import Stage2ConceptCandidate
from proposition.engine.stage2_concept_ensemble import FrozenConceptLinearHead
from proposition.engine.stage2_concept_trainer import Stage2ConceptTrainer, Stage2ConceptTrainerConfig
from proposition.models.stage2_concept_head import Stage2ConceptHead


def refit_fixed_epochs(
    dataset: Stage2Dataset,
    features: RoutedConceptFeatures,
    disease_ids: Sequence[str],
    candidate: Stage2ConceptCandidate,
    *,
    epochs: int,
    seed: int,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> tuple[FrozenConceptLinearHead, list[dict[str, float | int]]]:
    """Refit one seed on all pooled cases without validation or test access."""
    if not np.array_equal(dataset.case_ids, features.case_ids) or dataset.support_scores.shape != features.support_scores.shape:
        raise ValueError("pooled dataset and routed feature matrix must align")
    if candidate.routing_mode != features.routing_mode:
        raise ValueError("candidate routing differs from routed feature matrix")
    _seed(seed)
    disease_lookup = {identifier: index for index, identifier in enumerate(disease_ids)}
    model = Stage2ConceptHead(
        feature_disease_index=torch.tensor([disease_lookup[feature.disease_id] for feature in features.features]),
        feature_polarity=torch.tensor([feature.polarity for feature in features.features]),
        feature_mask=torch.ones(len(features.features), dtype=torch.bool),
        num_diseases=len(disease_ids), head_mode=candidate.head_mode,
    )
    trainer = Stage2ConceptTrainer(
        model=model, disease_ids=disease_ids, proposition_ids=features.feature_ids, routing_mode=candidate.routing_mode,
        input_hashes={}, score_source=[feature.source for feature in features.features],
        config=Stage2ConceptTrainerConfig(device=device, learning_rate=candidate.learning_rate, regularization=candidate.regularization,
            lambda_reg=candidate.lambda_reg, max_epochs=epochs, early_stopping_patience=1, batch_size=candidate.batch_size, base_seed=seed),
        output_dir=output_dir,
    )
    history = trainer.fit_fixed_epochs(dataset, epochs=epochs)
    frozen = FrozenConceptLinearHead(
        feature_ids=features.feature_ids, base_proposition_ids=tuple(feature.base_proposition_id for feature in features.features),
        feature_disease_ids=tuple(feature.disease_id for feature in features.features), feature_categories=tuple(feature.category for feature in features.features),
        feature_polarities=np.asarray([feature.polarity for feature in features.features]),
        feature_sources=tuple(feature.source for feature in features.features), disease_ids=tuple(disease_ids),
        effective_weights=model.effective_weights.detach().cpu().numpy(), bias=model.bias.detach().cpu().numpy(),
        head_mode=candidate.head_mode, routing_mode=candidate.routing_mode, regularization=candidate.regularization,
        lambda_reg=candidate.lambda_reg, refit_epochs=epochs, seeds=(seed,),
    )
    return frozen, history


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

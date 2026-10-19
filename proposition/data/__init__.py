"""Data contracts and datasets for Stage-1 evidence and Stage-2 calibration."""

from .stage2_scores import (
    DiseaseTargets,
    RoutedStage2Scores,
    Stage2Dataset,
    Stage2Proposition,
    Stage2ScoreArtifacts,
    build_stage2_support,
    load_stage2_disease_targets,
    load_stage2_propositions,
    load_stage2_score_artifact,
    validate_stage2_splits,
)

from .caption_masking import DEFAULT_MASK_TOKEN, DiseaseAliasMasker, disease_aliases, mask_disease_aliases
from .stage1_dataset import Stage1SkinCAPDataset
from .weak_supervision import (
    UNKNOWN_LABEL,
    ActiveProposition,
    WeakSupervisionArtifacts,
    load_weak_supervision_v3,
)

__all__ = [
    "DEFAULT_MASK_TOKEN",
    "DiseaseAliasMasker",
    "disease_aliases",
    "mask_disease_aliases",
    "Stage1SkinCAPDataset",
    "UNKNOWN_LABEL",
    "ActiveProposition",
    "WeakSupervisionArtifacts",
    "load_weak_supervision_v3",
    "DiseaseTargets",
    "RoutedStage2Scores",
    "Stage2Dataset",
    "Stage2Proposition",
    "Stage2ScoreArtifacts",
    "build_stage2_support",
    "load_stage2_disease_targets",
    "load_stage2_propositions",
    "load_stage2_score_artifact",
    "validate_stage2_splits",
]

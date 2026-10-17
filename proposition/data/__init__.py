"""Data contracts and datasets for Stage-1 proposition evidence learning."""

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
]

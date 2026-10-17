"""Data contracts and datasets for Stage-1 proposition evidence learning."""

from .weak_supervision import (
    UNKNOWN_LABEL,
    ActiveProposition,
    WeakSupervisionArtifacts,
    load_weak_supervision_v3,
)

__all__ = [
    "UNKNOWN_LABEL",
    "ActiveProposition",
    "WeakSupervisionArtifacts",
    "load_weak_supervision_v3",
]

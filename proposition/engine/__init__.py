"""Stage-1 training, metrics, artifacts, and diagnostics."""

from .diagnostics import (
    proposition_correlation,
    query_ablation,
    query_swap,
    save_attention_artifact,
    within_patient_differentiation,
)
from .metrics import evaluate_modalities, masked_binary_cross_entropy, modality_metrics

try:  # Keep NumPy-only metrics and diagnostics importable without PyTorch.
    from .trainer import EpochResult, Stage1Trainer, TrainerConfig
except ImportError as error:
    if error.name != "torch":
        raise

__all__ = (
    "evaluate_modalities",
    "masked_binary_cross_entropy",
    "modality_metrics",
    "proposition_correlation",
    "query_ablation",
    "query_swap",
    "save_attention_artifact",
    "within_patient_differentiation",
)
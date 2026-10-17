import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from proposition.data.weak_supervision import WeakSupervisionArtifacts
from proposition.losses.masked_bce import (
    MaskedBCELoss,
    build_pos_weight_artifacts,
    write_label_distribution,
)


def _artifacts() -> WeakSupervisionArtifacts:
    return WeakSupervisionArtifacts(
        case_ids=np.asarray(["train-1", "train-2", "train-3", "valid-1"]),
        splits=np.asarray(["train", "train", "train", "valid"]),
        proposition_ids=np.asarray(["positive-only", "negative-only", "mixed", "inactive"]),
        label_text=np.asarray([[1, 0, 1, 1], [1, 0, 0, 0], [-1, -1, 1, 1], [0, 1, 0, 0]], dtype=np.int8),
        label_img=np.asarray([[1, 0, 0, 1], [1, 0, 1, 0], [-1, -1, 0, 1], [0, 1, 1, 0]], dtype=np.int8),
        p_img=np.zeros((4, 4), dtype=np.float32),
        text_trainable=np.asarray([True, True, True, True]),
        image_trainable=np.asarray([True, True, True, False]),
        active_scoring=np.asarray([True, True, True, False]),
        min_labels=1,
        active_propositions=(),
        metadata={},
        artifact_hashes={},
    )


def test_pos_weights_use_only_train_valid_active_trainable_labels() -> None:
    weights = build_pos_weight_artifacts(_artifacts(), pos_weight_cap=1.5)

    assert weights.text_pos_weight.tolist() == pytest.approx([1.0, 1.0, 0.5])
    assert weights.image_pos_weight.tolist() == pytest.approx([1.0, 1.0, 1.5])
    assert [row.proposition_id for row in weights.rows] == [
        "positive-only", "negative-only", "mixed", "positive-only", "negative-only", "mixed",
    ]
    rows = {(row.proposition_id, row.modality): row for row in weights.rows}
    assert rows[("positive-only", "text")].n_present == 2
    assert rows[("positive-only", "text")].n_absent == 0
    assert rows[("positive-only", "text")].n_unknown == 1
    assert rows[("positive-only", "text")].single_class is True
    assert rows[("mixed", "image")].pos_weight == pytest.approx(1.5)
    assert ("inactive", "text") not in rows
    assert ("inactive", "image") not in rows


def test_positive_only_target_has_nonzero_loss() -> None:
    loss = MaskedBCELoss(torch.tensor([1.0]), torch.tensor([1.0]))
    result = loss(
        torch.tensor([[0.0]]),
        torch.zeros((1, 1)),
        torch.tensor([[1]]),
        torch.tensor([[-1]]),
        torch.tensor([True]),
        torch.tensor([False]),
    )

    assert result.item() == pytest.approx(float(F.softplus(torch.tensor(0.0))))


def test_unknown_and_masked_labels_do_not_contribute() -> None:
    loss = MaskedBCELoss(torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0]))
    logits = torch.tensor([[0.0, 100.0]])
    result = loss(
        logits,
        torch.zeros_like(logits),
        torch.tensor([[1, -1]]),
        torch.tensor([[-1, -1]]),
        torch.tensor([True, True]),
        torch.tensor([True, True]),
    )

    assert result.item() == pytest.approx(float(2.0 * F.softplus(torch.tensor(0.0))))


def test_empty_branches_return_differentiable_zero() -> None:
    loss = MaskedBCELoss(torch.tensor([1.0]), torch.tensor([1.0]))
    text_logits = torch.tensor([[0.5]], requires_grad=True)
    image_logits = torch.tensor([[-0.5]], requires_grad=True)
    result = loss(
        text_logits,
        image_logits,
        torch.tensor([[-1]]),
        torch.tensor([[-1]]),
        torch.tensor([True]),
        torch.tensor([True]),
    )
    result.backward()

    assert result.item() == 0.0
    assert text_logits.grad.item() == 0.0
    assert image_logits.grad.item() == 0.0


def test_loss_accepts_per_example_modality_masks() -> None:
    loss = MaskedBCELoss(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    result = loss(
        torch.tensor([[0.0, 20.0]]),
        torch.zeros((1, 2)),
        torch.tensor([[1, 0]]),
        torch.tensor([[-1, -1]]),
        torch.tensor([[True, False]]),
        torch.tensor([[False, False]]),
    )

    assert result.item() == pytest.approx(float(F.softplus(torch.tensor(0.0))))


def test_distribution_writer_uses_required_audit_columns(tmp_path) -> None:
    weights = build_pos_weight_artifacts(_artifacts())
    output = tmp_path / "audit" / "label_distribution.csv"
    write_label_distribution(weights.rows, output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["proposition_id", "modality", "N_present", "N_absent", "N_unknown", "pos_weight", "single_class"]
    assert len(rows) == 6

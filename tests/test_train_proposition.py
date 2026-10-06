import pytest
import torch
import torch.nn.functional as F

from engine.train_proposition import EncodedBatch, compute_objective, evaluate, train_one_epoch
from factory.pecl_loss import PECLLoss
from models.proposition_model import PropositionModel


def _components(*, label_branch: bool = True):
    torch.manual_seed(13)
    prop_disease = torch.tensor([0, 0, 1, 1])
    prop_polarity = torch.tensor([1, -1, 1, 1])
    model = PropositionModel(
        6, 2, prop_disease, prop_polarity, use_label_branch=label_branch
    )
    pecl = PECLLoss(
        (torch.tensor([0, 1]), torch.tensor([2, 3])), num_prototypes=4
    )
    return model, pecl


def _batch(batch_size: int = 2, *, label_query: bool = True) -> EncodedBatch:
    return EncodedBatch(
        memory=torch.randn(batch_size, 5, 6),
        query=torch.randn(4, 6),
        labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]])[:batch_size],
        text_pooled=torch.randn(batch_size, 6),
        image_pooled=torch.randn(batch_size, 6),
        alignment_prototypes=torch.randn(4, 6),
        memory_padding_mask=torch.tensor(
            [[False, False, False, True, True], [False, False, False, False, True]]
        )[:batch_size],
        label_query=torch.randn(2, 6) if label_query else None,
    )


def test_objective_matches_documented_sum() -> None:
    model, pecl = _components()
    batch = _batch()

    output = compute_objective(
        model,
        pecl,
        batch,
        lambda_label=0.7,
        beta_pecl=0.4,
        pecl_generator=torch.Generator().manual_seed(5),
    )

    assert output.label is not None
    expected = output.proposition + 0.4 * output.pecl + 0.7 * output.label
    assert torch.allclose(output.total, expected)
    direct_prop = F.binary_cross_entropy_with_logits(
        output.model_output.proposition.logits, batch.labels
    )
    assert torch.allclose(output.proposition, direct_prop)


def test_no_pecl_and_no_label_branch_schema() -> None:
    model, _ = _components(label_branch=False)
    batch = _batch(label_query=False)

    output = compute_objective(model, None, batch, lambda_label=0)

    assert output.label is None
    assert output.pecl.item() == 0
    assert torch.equal(output.total, output.proposition)
    with pytest.raises(ValueError, match="lambda_label must be zero"):
        compute_objective(model, None, batch, lambda_label=1)


def test_train_epoch_updates_parameters_and_returns_both_predictions() -> None:
    model, pecl = _components()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    batches = [_batch(), _batch()]
    before = model.pathway.scorer.layers[0].weight.detach().clone()

    result = train_one_epoch(
        model,
        pecl,
        batches,
        optimizer,
        accumulation_steps=2,
        pecl_generator=torch.Generator().manual_seed(7),
    )

    after = model.pathway.scorer.layers[0].weight.detach()
    assert not torch.equal(before, after)
    assert result.gt.shape == (4, 2)
    assert result.pred_proposition.shape == (4, 2)
    assert result.pred_label is not None and result.pred_label.shape == (4, 2)
    assert set(result.losses) == {
        "total", "proposition", "pecl", "pecl_text", "pecl_image", "label"
    }


def test_final_short_accumulation_window_still_updates() -> None:
    model, _ = _components(label_branch=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = model.pathway.scorer.layers[0].weight.detach().clone()

    result = train_one_epoch(
        model,
        None,
        [_batch(label_query=False)],
        optimizer,
        accumulation_steps=3,
        lambda_label=0,
    )

    assert not torch.equal(before, model.pathway.scorer.layers[0].weight.detach())
    assert result.pred_label is None
    assert "label" not in result.losses


def test_evaluate_does_not_update_parameters_or_emit_disabled_branch() -> None:
    model, pecl = _components(label_branch=False)
    batch = _batch(label_query=False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = evaluate(
        model,
        pecl,
        [batch],
        lambda_label=0,
        pecl_generator=torch.Generator().manual_seed(3),
    )

    assert result.pred_label is None
    assert model.training is False
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_empty_epoch_is_rejected() -> None:
    model, _ = _components(label_branch=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with pytest.raises(ValueError, match="no samples"):
        train_one_epoch(model, None, [], optimizer, lambda_label=0)

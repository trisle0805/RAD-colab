import torch
import torch.nn.functional as F

from proposition.losses.pecl import PECLLoss


def _loss(negative_ratio: int = 5) -> PECLLoss:
    # Prototype 1 is shared by diseases 0 and 1; the remaining prototypes are unique.
    mapping = (
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([3]),
    )
    return PECLLoss(mapping, 4, negative_ratio=negative_ratio)


def test_positive_and_negative_masks_deduplicate_and_exclude_overlap() -> None:
    module = _loss()
    labels = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)

    positive, negative = module._prototype_masks(labels)

    assert torch.equal(positive[0], torch.tensor([True, True, True, False]))
    assert torch.equal(negative[0], torch.tensor([False, False, False, True]))
    assert torch.equal(positive[1], torch.tensor([True, True, False, False]))
    # Shared prototype 1 belongs to a negative disease too, but must not be negative.
    assert torch.equal(negative[1], torch.tensor([False, False, True, True]))


def test_modality_loss_matches_manual_separate_positive_negative_means() -> None:
    module = _loss(negative_ratio=5)
    features = torch.tensor([[1.0, 0.0]])
    prototypes = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    temperature = 0.5

    actual = module.modality_loss(
        features, prototypes, labels, temperature=temperature,
        generator=torch.Generator().manual_seed(3)
    )

    logits = torch.tensor([2.0, 0.0, -2.0, 0.0])
    expected_positive = -F.logsigmoid(logits[:2]).mean()
    expected_negative = -F.logsigmoid(-logits[2:]).mean()
    assert torch.allclose(actual, expected_positive + expected_negative, atol=1e-6)


def test_full_batch_denominator_keeps_invalid_samples_as_zero_contribution() -> None:
    module = _loss(negative_ratio=0)
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    prototypes = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    actual = module.modality_loss(features, prototypes, labels, temperature=1.0)
    valid_only = module.modality_loss(
        features[:1], prototypes, labels[:1], temperature=1.0
    )

    assert torch.allclose(actual, valid_only / 2, atol=1e-6)


def test_negative_sampling_is_reproducible_with_dedicated_generator() -> None:
    module = _loss(negative_ratio=1)
    features = torch.tensor([[1.0, 0.2]])
    prototypes = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    labels = torch.tensor([[0.0, 0.0, 1.0]])

    first = module.modality_loss(
        features, prototypes, labels, temperature=0.5,
        generator=torch.Generator().manual_seed(19)
    )
    second = module.modality_loss(
        features, prototypes, labels, temperature=0.5,
        generator=torch.Generator().manual_seed(19)
    )

    assert torch.equal(first, second)


def test_forward_weights_modalities_and_detaches_only_image_prototypes() -> None:
    module = _loss()
    text = torch.randn(2, 5, requires_grad=True)
    image = torch.randn(2, 5, requires_grad=True)
    prototypes = torch.randn(4, 5, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    output = module(
        text, image, prototypes, labels,
        generator=torch.Generator().manual_seed(11)
    )
    assert torch.allclose(output.total, 0.1 * output.text + 0.001 * output.image)
    output.total.backward()

    assert text.grad is not None and torch.count_nonzero(text.grad) > 0
    assert image.grad is not None and torch.count_nonzero(image.grad) > 0
    assert prototypes.grad is not None and torch.count_nonzero(prototypes.grad) > 0

    prototypes_image_only = prototypes.detach().clone().requires_grad_()
    image_only = module.modality_loss(
        image.detach().clone().requires_grad_(),
        prototypes_image_only,
        labels,
        temperature=2.0,
        generator=torch.Generator().manual_seed(11),
        detach_prototypes=True,
    )
    image_only.backward()
    assert prototypes_image_only.grad is None


def test_zero_valid_samples_returns_differentiable_zero() -> None:
    module = _loss()
    features = torch.randn(2, 4, requires_grad=True)
    prototypes = torch.randn(4, 4, requires_grad=True)
    labels = torch.zeros(2, 3)

    result = module.modality_loss(features, prototypes, labels, temperature=0.5)

    assert result.item() == 0.0
    result.backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) == 0

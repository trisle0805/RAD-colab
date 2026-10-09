import pytest
import torch

from proposition.models.proposition_model import (
    CosineCrossAttention,
    DiseaseAggregator,
    PositiveCalibration,
    PropositionModel,
    PropositionPathway,
    SegmentQueryPathway,
)


def _fixture_tensors():
    torch.manual_seed(7)
    batch_size, query_count, memory_length, embed_dim = 2, 7, 11, 8
    query = torch.randn(query_count, embed_dim)
    memory = torch.randn(batch_size, memory_length, embed_dim)
    mask = torch.zeros(batch_size, memory_length, dtype=torch.bool)
    mask[0, -3:] = True
    mask[1, -1] = True
    return query, memory, mask


def test_cosine_attention_shapes_normalization_mask_and_logit_bounds() -> None:
    query, memory, mask = _fixture_tensors()
    module = CosineCrossAttention(embed_dim=8, tau_att=0.1)

    output = module(query, memory, mask)

    assert output.context.shape == (2, 7, 8)
    assert output.attention is not None
    assert output.attention.shape == (2, 7, 11)
    assert torch.allclose(output.attention.sum(dim=-1), torch.ones(2, 7), atol=1e-6)
    expanded_mask = mask.unsqueeze(1).expand_as(output.attention)
    assert torch.count_nonzero(output.attention[expanded_mask]) == 0

    projected_query = torch.nn.functional.normalize(module.q_proj(query), dim=-1)
    projected_memory = torch.nn.functional.normalize(module.k_proj(memory), dim=-1)
    logits = torch.einsum("qd,bld->bql", projected_query, projected_memory) / module.tau_att
    assert float(logits.min()) >= -10.0 - 1e-5
    assert float(logits.max()) <= 10.0 + 1e-5


def test_cosine_attention_chunking_matches_full_computation() -> None:
    query, memory, mask = _fixture_tensors()
    module = CosineCrossAttention(embed_dim=8)

    full = module(query, memory, mask, query_chunk_size=None)
    chunked = module(query, memory, mask, query_chunk_size=3)

    assert torch.allclose(chunked.context, full.context, atol=1e-6)
    assert chunked.attention is not None and full.attention is not None
    assert torch.allclose(chunked.attention, full.attention, atol=1e-6)


def test_cosine_attention_can_skip_attention_output() -> None:
    query, memory, mask = _fixture_tensors()
    output = CosineCrossAttention(embed_dim=8)(
        query, memory, mask, return_attention=False, query_chunk_size=2
    )
    assert output.context.shape == (2, 7, 8)
    assert output.attention is None


def test_cosine_attention_rejects_fully_masked_sample() -> None:
    query, memory, mask = _fixture_tensors()
    mask[0] = True
    with pytest.raises(ValueError, match="at least one unmasked"):
        CosineCrossAttention(embed_dim=8)(query, memory, mask)


def test_mean_aggregation_matches_python_reference() -> None:
    prop_disease = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    compatibility = torch.tensor(
        [[0.1, 0.9, 0.2, 0.4, 0.9, 0.3, 0.7], [0.8, 0.6, 0.1, 0.3, 0.5, 0.2, 1.0]]
    )
    aggregator = DiseaseAggregator(prop_disease, num_diseases=3, mode="mean")

    actual = aggregator(compatibility)
    expected = torch.stack(
        [
            torch.stack(
                [row[prop_disease == disease].mean() for disease in range(3)]
            )
            for row in compatibility
        ]
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_weighted_aggregation_initially_equals_mean_and_normalizes_per_disease() -> None:
    prop_disease = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    compatibility = torch.rand(3, 7)
    mean = DiseaseAggregator(prop_disease, 3, "mean")
    weighted = DiseaseAggregator(prop_disease, 3, "weighted")

    assert torch.allclose(weighted(compatibility), mean(compatibility), atol=1e-6)
    weights = weighted.weights()
    for disease in range(3):
        assert torch.allclose(weights[prop_disease == disease].sum(), torch.tensor(1.0))


def test_polarity_and_monotonic_calibration() -> None:
    prop_disease = torch.tensor([0, 0, 1])
    polarity = torch.tensor([1, -1, 1])
    pathway = PropositionPathway(
        embed_dim=4,
        num_diseases=2,
        prop_disease=prop_disease,
        prop_polarity=polarity,
    )

    support = torch.tensor([[0.2, 0.3, 0.8]])
    compatibility = torch.where(polarity.unsqueeze(0) == 1, support, 1.0 - support)
    assert torch.allclose(compatibility, torch.tensor([[0.2, 0.7, 0.8]]))

    calibration = PositiveCalibration(2)
    assert torch.all(calibration.slope > 0)
    low = calibration(torch.tensor([[0.2, 0.4]]))
    high = calibration(torch.tensor([[0.3, 0.4]]))
    assert high[0, 0] > low[0, 0]
    assert high[0, 1] == low[0, 1]
    assert torch.allclose(calibration.slope, torch.full((2,), 10.0), atol=1e-5)


def test_proposition_pathway_shapes_and_polarity() -> None:
    torch.manual_seed(5)
    prop_disease = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    polarity = torch.tensor([1, -1, 1, 1, -1, 1, 1])
    pathway = PropositionPathway(8, 3, prop_disease, polarity)
    query, memory, mask = _fixture_tensors()

    output = pathway(query, memory, mask, return_attention=True, query_chunk_size=3)

    assert output.logits.shape == (2, 3)
    assert output.probabilities.shape == (2, 3)
    assert output.support.shape == (2, 7)
    assert output.compatibility.shape == (2, 7)
    assert output.disease_scores.shape == (2, 3)
    assert output.attention is not None and output.attention.shape == (2, 7, 11)
    assert torch.allclose(output.compatibility[:, 0], output.support[:, 0])
    assert torch.allclose(output.compatibility[:, 1], 1.0 - output.support[:, 1])


def test_disease_level_path_skips_polarity_and_aggregation() -> None:
    torch.manual_seed(5)
    pathway = PropositionPathway(
        8,
        3,
        prop_disease=None,
        prop_polarity=None,
        query_level="disease",
    )
    query = torch.randn(3, 8)
    memory = torch.randn(2, 5, 8)

    output = pathway(query, memory)

    assert output.support.shape == (2, 3)
    assert torch.equal(output.compatibility, output.support)
    assert torch.equal(output.disease_scores, output.support)
    assert pathway.aggregation_weights() is None

def test_segment_query_pathway_shapes_context_and_chunking() -> None:
    torch.manual_seed(13)
    prop_disease = torch.tensor([0, 0, 1])
    prop_align_index = torch.tensor([0, 1, 2])
    pathway = SegmentQueryPathway(4, 2, prop_disease, prop_align_index, 3)
    query = torch.randn(3, 5, 4)
    content_mask = torch.tensor(
        [[False, True, True, False, False], [False, True, False, False, False], [False, True, True, True, False]]
    )
    memory = torch.randn(2, 6, 4)
    full = pathway(query, content_mask, memory, return_attention=True)
    chunked = pathway(query, content_mask, memory, return_attention=True, query_chunk_size=3)

    assert full.logits.shape == (2, 2)
    assert full.support.shape == (2, 2)
    assert full.attention is not None and full.attention.shape == (2, 3, 6)
    assert chunked.attention is not None
    assert torch.allclose(chunked.logits, full.logits, atol=1e-6)
    assert torch.allclose(chunked.attention, full.attention, atol=1e-6)

    disease_attention = torch.stack(
        [full.attention[:, prop_disease == disease].mean(dim=1) for disease in range(2)], dim=1
    )
    expected_context = disease_attention @ pathway.attention.v_proj(memory)
    content_queries = torch.stack(
        [query[text, content_mask[text]].mean(dim=0) for text in range(3)]
    )
    disease_query = torch.stack(
        [content_queries[prop_disease == disease].mean(dim=0) for disease in range(2)]
    )
    expected_support = torch.sigmoid(pathway.scorer(expected_context, disease_query))
    assert torch.allclose(full.support, expected_support, atol=1e-5)

def test_segment_query_ignores_special_and_padding_tokens() -> None:
    torch.manual_seed(17)
    pathway = SegmentQueryPathway(4, 1, torch.tensor([0, 0]), torch.tensor([0, 1]), 2)
    query = torch.randn(2, 5, 4)
    content_mask = torch.tensor([[False, True, True, False, False], [False, True, False, False, False]])
    memory = torch.randn(2, 6, 4)
    baseline = pathway(query, content_mask, memory).logits
    changed = query.clone()
    changed[~content_mask] = torch.randn_like(changed[~content_mask]) * 1000
    assert torch.allclose(pathway(changed, content_mask, memory).logits, baseline, atol=1e-6)


def test_optional_label_branch_schema_and_independent_parameters() -> None:
    prop_disease = torch.tensor([0, 0, 1])
    polarity = torch.tensor([1, -1, 1])
    query = torch.randn(3, 8)
    memory = torch.randn(2, 5, 8)
    label_query = torch.randn(2, 8)

    with_branch = PropositionModel(8, 2, prop_disease, polarity, use_label_branch=True)
    output = with_branch(query, memory, label_query=label_query)
    assert output.label is not None
    assert output.label.logits.shape == (2, 2)
    assert with_branch.label_branch is not None
    assert (
        with_branch.pathway.attention.q_proj.weight.data_ptr()
        != with_branch.label_branch.attention.q_proj.weight.data_ptr()
    )

    without_branch = PropositionModel(8, 2, prop_disease, polarity, use_label_branch=False)
    output_without = without_branch(query, memory)
    assert output_without.label is None
    with pytest.raises(ValueError, match="branch is disabled"):
        without_branch(query, memory, label_query=label_query)

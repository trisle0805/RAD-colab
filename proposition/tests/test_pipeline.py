import json
from pathlib import Path

import torch
from torch import nn

from proposition.data.knowledge_base import load_proposition_kb
from proposition.engine.pipeline import EncodedBatchStream, prepare_knowledge_tokens


class FakeTokenizer:
    def __call__(
        self, texts, *, add_special_tokens=True, padding=False, truncation=False,
        max_length=None, return_tensors=None,
    ):
        encoded = [[101] + list(range(10, 10 + len(text.split()))) + [102] for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        if padding:
            width = max_length if padding == "max_length" else max(map(len, encoded))
            encoded = [ids + [0] * (width - len(ids)) for ids in encoded]
        if return_tensors == "pt":
            input_ids = torch.tensor(encoded)
            return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).long()}
        return {"input_ids": encoded}


class FakeImageEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dim = dim

    def forward(self, images):
        tokens = self.scale * torch.ones(images.shape[0], 2, self.dim)
        return tokens, tokens.mean(1)


class FakeTextEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dim = dim

    def encode_text(self, tokens):
        hidden = self.scale * tokens["input_ids"].float().unsqueeze(-1).repeat(1, 1, self.dim)
        return hidden[:, 0], hidden


def _kb(tmp_path: Path):
    payload = [
        {
            "disease_id": "A",
            "sourceReference": "ref-a",
            "propositions": [
                {"proposition_id": "a1", "category": "morphology", "canonicalDescription": "red spot", "polarity": 1, "sourceExcerpt": "x"}
            ],
        },
        {
            "disease_id": "B",
            "sourceReference": "ref-b",
            "propositions": [
                {"proposition_id": "b1", "category": "symptom", "canonicalDescription": "no pain", "polarity": -1, "sourceExcerpt": "y"}
            ],
        },
    ]
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_proposition_kb(path, ["A", "B"])


def test_prepare_knowledge_tokens_measures_and_rounds(tmp_path) -> None:
    prepared = prepare_knowledge_tokens(FakeTokenizer(), _kb(tmp_path), multiple=8)

    assert prepared.measured_max_length == 6
    assert prepared.max_length == 8
    assert prepared.unique["input_ids"].shape[1] == 8
    assert prepared.alignment["input_ids"].shape[1] == 8
    assert prepared.alignment_content_mask.shape == prepared.alignment["input_ids"].shape
    assert not prepared.alignment_content_mask[:, 0].any()
    assert prepared.alignment_content_mask.any(dim=1).all()
    assert len(prepared.disease_lengths) == 2


def test_encoded_batch_stream_builds_memory_mask_and_keeps_gradients(tmp_path) -> None:
    kb = _kb(tmp_path)
    tokenizer = FakeTokenizer()
    prepared = prepare_knowledge_tokens(tokenizer, kb)
    image_encoder = FakeImageEncoder(4)
    text_encoder = FakeTextEncoder(4)
    raw = [{
        "image": torch.zeros(2, 3, 4, 4),
        "label": torch.tensor([[1, 0], [0, 1]]),
        "entity": ["short", "two words"],
    }]

    batch = next(iter(EncodedBatchStream(
        raw, image_encoder, text_encoder, tokenizer, kb, prepared,
        torch.device("cpu"), caption_max_length=5,
    )))

    assert batch.memory.shape == (2, 7, 4)
    assert batch.memory_padding_mask[:, :2].sum() == 0
    assert batch.memory_padding_mask[0, -2:].all()
    assert batch.query.shape == (2, 4)
    assert batch.alignment_prototypes.shape == (2, 4)
    assert batch.label_query is not None
    assert torch.equal(batch.image_token_count, torch.tensor([2, 2], dtype=torch.int32))
    assert batch.caption_input_ids.shape == (2, 5)
    assert batch.caption_attention_mask.shape == (2, 5)
    batch.query.sum().backward()
    assert text_encoder.scale.grad is not None


def test_none_frozen_queries_preserves_default_stream_outputs(tmp_path) -> None:
    kb = _kb(tmp_path)
    tokenizer = FakeTokenizer()
    prepared = prepare_knowledge_tokens(tokenizer, kb)
    image_encoder = FakeImageEncoder(4)
    text_encoder = FakeTextEncoder(4)
    raw = [{
        "image": torch.zeros(2, 3, 4, 4),
        "label": torch.tensor([[1, 0], [0, 1]]),
        "entity": ["short", "two words"],
    }]

    default_batch = next(iter(EncodedBatchStream(
        raw, image_encoder, text_encoder, tokenizer, kb, prepared,
        torch.device("cpu"), caption_max_length=5,
    )))
    explicit_none_batch = next(iter(EncodedBatchStream(
        raw, image_encoder, text_encoder, tokenizer, kb, prepared,
        torch.device("cpu"), caption_max_length=5, frozen_unique_pooled=None,
    )))

    assert torch.equal(default_batch.query, explicit_none_batch.query)
    assert torch.equal(default_batch.alignment_prototypes, explicit_none_batch.alignment_prototypes)
    assert torch.equal(default_batch.memory, explicit_none_batch.memory)


def test_frozen_unique_queries_remain_identical_after_caption_encoder_update(tmp_path) -> None:
    kb = _kb(tmp_path)
    tokenizer = FakeTokenizer()
    prepared = prepare_knowledge_tokens(tokenizer, kb)
    image_encoder = FakeImageEncoder(4)
    text_encoder = FakeTextEncoder(4)
    frozen = torch.arange(len(kb.unique_texts) * 4, dtype=torch.float32).reshape(-1, 4)
    raw = [
        {
            "image": torch.zeros(1, 3, 4, 4),
            "label": torch.tensor([[1, 0]]),
            "entity": ["first caption"],
        },
        {
            "image": torch.zeros(1, 3, 4, 4),
            "label": torch.tensor([[0, 1]]),
            "entity": ["second caption"],
        },
    ]
    stream = EncodedBatchStream(
        raw, image_encoder, text_encoder, tokenizer, kb, prepared, torch.device("cpu"),
        caption_max_length=5, frozen_unique_pooled=frozen,
    )
    iterator = iter(stream)
    first = next(iterator)
    optimizer = torch.optim.SGD(text_encoder.parameters(), lr=0.1)
    optimizer.zero_grad()
    first.text_pooled.sum().backward()
    assert text_encoder.scale.grad is not None and text_encoder.scale.grad.abs() > 0
    optimizer.step()
    second = next(iterator)

    assert torch.equal(first.query, frozen[kb.query_text_index])
    assert torch.equal(first.query, second.query)
    align_to_unique = torch.empty(len(kb.unique_alignment_texts), dtype=torch.long)
    align_to_unique[kb.align_unique_index] = kb.align_text_index
    assert torch.equal(first.alignment_prototypes, frozen[align_to_unique])
    assert not first.query.requires_grad

def test_segment_stream_uses_alignment_tokens_in_pecl_order_and_keeps_gradients(tmp_path) -> None:
    kb = _kb(tmp_path)
    tokenizer = FakeTokenizer()
    prepared = prepare_knowledge_tokens(tokenizer, kb)
    text_encoder = FakeTextEncoder(4)
    raw = [{
        "image": torch.zeros(1, 3, 4, 4),
        "label": torch.tensor([[1, 0]]),
        "entity": ["caption"],
    }]
    batch = next(iter(EncodedBatchStream(
        raw, FakeImageEncoder(4), text_encoder, tokenizer, kb, prepared, torch.device("cpu"),
        caption_max_length=5, query_level="proposition_segments",
    )))

    expected_pooled, expected_hidden = text_encoder.encode_text(prepared.alignment)
    assert torch.equal(batch.query, expected_hidden)
    assert torch.equal(batch.alignment_prototypes, expected_pooled)
    assert torch.equal(batch.query_token_mask, prepared.alignment_content_mask)
    batch.query[batch.query_token_mask].sum().backward()
    assert text_encoder.scale.grad is not None

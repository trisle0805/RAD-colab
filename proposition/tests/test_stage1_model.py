from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from proposition.data.weak_supervision import ActiveProposition, WeakSupervisionArtifacts
from proposition.models import Stage1EvidenceModel


class TinyImageEncoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, dimension)
        self.bn = nn.BatchNorm1d(dimension)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bn(self.projection(images).transpose(1, 2)).transpose(1, 2)
        return features, features.mean(dim=1)


class TinyTextEncoder(nn.Module):
    def __init__(self, vocabulary: int, dimension: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dimension)
        self.dropout = nn.Dropout(0.8)

    def encode_text(self, tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.dropout(self.embedding(tokens["input_ids"]))
        return features[:, 0], features


def artifacts() -> WeakSupervisionArtifacts:
    ids = np.asarray(["both", "text", "image", "inactive"])
    rows = tuple(
        ActiveProposition(
            proposition_id=identifier,
            disease_id="must-not-reach-query",
            category="must-not-reach-query",
            canonical_description=f"canonical {identifier}",
            polarity=1,
            text_trainable=identifier in {"both", "text"},
            image_trainable=identifier in {"both", "image"},
            active_scoring=identifier != "inactive",
        )
        for identifier in ids
    )
    return WeakSupervisionArtifacts(
        case_ids=np.asarray(["case-1"]), splits=np.asarray(["train"]), proposition_ids=ids,
        label_text=np.zeros((1, 4), dtype=np.int8), label_img=np.zeros((1, 4), dtype=np.int8),
        p_img=np.zeros((1, 4), dtype=np.float32), text_trainable=np.asarray([1, 1, 0, 0], dtype=bool),
        image_trainable=np.asarray([1, 0, 1, 0], dtype=bool), active_scoring=np.asarray([1, 1, 1, 0], dtype=bool),
        min_labels=1, active_propositions=rows, metadata={}, artifact_hashes={},
    )


def tokenizer(descriptions: tuple[str, ...]) -> dict[str, torch.Tensor]:
    assert descriptions == ("canonical both", "canonical text", "canonical image")
    ids = torch.arange(1, len(descriptions) + 1).unsqueeze(1).repeat(1, 3)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def build_model() -> Stage1EvidenceModel:
    return Stage1EvidenceModel(
        image_encoder=TinyImageEncoder(8), text_encoder=TinyTextEncoder(10, 8),
        weak_supervision=artifacts(), query_tokenizer=tokenizer, embed_dim=8, num_heads=2, query_chunk_size=2,
    )


def test_outputs_attention_chunks_and_fixed_fusion() -> None:
    model = build_model().train()
    outputs = model(
        torch.randn(2, 4, 3),
        {"input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]), "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])},
        return_attention=True,
    )
    assert outputs["logits_img"].shape == (2, 3)
    assert outputs["scores_txt"].shape == (2, 3)
    assert outputs["attention"]["image"].shape == (2, 3, 4)
    assert outputs["attention"]["text"].shape == (2, 3, 3)
    assert torch.all((outputs["scores_img"] >= 0) & (outputs["scores_img"] <= 1))
    support = outputs["support_scores"]
    assert torch.allclose(support[:, 0], (outputs["scores_img"][:, 0] + outputs["scores_txt"][:, 0]) / 2)
    assert torch.allclose(support[:, 1], outputs["scores_txt"][:, 1])
    assert torch.allclose(support[:, 2], outputs["scores_img"][:, 2])


def test_frozen_encoders_remain_eval_and_unchanged_after_step() -> None:
    model = build_model().train()
    assert not model.image_encoder.training and not model.text_encoder.training
    assert all(not parameter.requires_grad for parameter in model.image_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.text_encoder.parameters())
    before = [parameter.detach().clone() for parameter in (*model.image_encoder.parameters(), *model.text_encoder.parameters())]
    bn_mean = model.image_encoder.bn.running_mean.detach().clone()
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.1)
    outputs = model(torch.randn(2, 4, 3), {"input_ids": torch.ones((2, 3), dtype=torch.long), "attention_mask": torch.ones((2, 3), dtype=torch.long)})
    (outputs["logits_img"].sum() + outputs["logits_txt"].sum()).backward()
    optimizer.step()
    after = [parameter.detach() for parameter in (*model.image_encoder.parameters(), *model.text_encoder.parameters())]
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert torch.equal(bn_mean, model.image_encoder.bn.running_mean)
    assert not model.text_encoder.dropout.training


def test_attention_modules_are_distinct() -> None:
    model = build_model()
    assert model.image_cross_attention is not model.text_cross_attention
    assert model.image_cross_attention.attention.in_proj_weight is not model.text_cross_attention.attention.in_proj_weight


class NativeResWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.res_features = nn.Conv2d(3, 2048, kernel_size=1)
        self.res_l1 = nn.Linear(2048, 2048)
        self.res_l2 = nn.Linear(2048, 8)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("Stage 1 must use native res_features, not ModelRes.forward")


class NativeBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, 8)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class NativeTextWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert_model = NativeBert()
        self.mlp_embed = nn.Linear(8, 8)

    def encode_text(self, tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("Stage 1 must use native bert_model, not mlp_embed")


def test_legacy_wrappers_use_native_features_and_train_stage_projection() -> None:
    image_encoder = NativeResWrapper()
    text_encoder = NativeTextWrapper()
    model = Stage1EvidenceModel(
        image_encoder=image_encoder, text_encoder=text_encoder, weak_supervision=artifacts(),
        query_tokenizer=tokenizer, embed_dim=8, num_heads=2,
    ).train()
    assert all(not parameter.requires_grad for parameter in image_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in text_encoder.parameters())
    outputs = model(
        torch.randn(2, 3, 2, 2),
        {"input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]), "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])},
    )
    assert model.query_cache.shape == (3, 8)
    assert outputs["logits_img"].shape == (2, 3)
    (outputs["logits_img"].sum() + outputs["logits_txt"].sum()).backward()
    assert model.image_projection.weight.grad is not None
    assert image_encoder.res_features.weight.grad is None
    assert text_encoder.bert_model.embedding.weight.grad is None
    assert text_encoder.mlp_embed.weight.grad is None

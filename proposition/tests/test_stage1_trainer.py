from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader

from proposition.data.weak_supervision import ActiveProposition, WeakSupervisionArtifacts
from proposition.engine.trainer import Stage1Trainer, TrainerConfig, _modality_macro_aucs
from proposition.losses import MaskedBCELoss


class EvidenceModel(nn.Module):
    active_proposition_ids = ("p1", "p2")

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def _queries(self, device):
        return torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float32,
            device=device,
        )

    def forward(
        self,
        images,
        caption_tokens,
        *,
        query_embeddings=None,
        return_attention=False,
    ):
        logits = self.linear(images.float())

        # Support query-control diagnostics used by Stage1Trainer.
        if query_embeddings is not None:
            query_effect = query_embeddings.squeeze(-1).unsqueeze(0)
            logits = logits + 0.01 * query_effect

        scores = logits.sigmoid()

        output = {
            "logits_txt": logits,
            "logits_img": logits,
            "scores_txt": scores,
            "scores_img": scores,
            "support_scores": scores,
        }

        if return_attention:
            batch = images.shape[0]
            output["attention"] = {
                "image": torch.zeros(
                    batch, 2, 1,
                    device=images.device,
                ),
                "text": torch.zeros(
                    batch, 2, 1,
                    device=images.device,
                ),
            }

        return output


def artifacts(*, text_trainable=(True, True), image_trainable=(True, True)) -> WeakSupervisionArtifacts:
    ids = np.asarray(["p1", "p2"])
    props = tuple(ActiveProposition(value, "disease", "category", value, 1, text, image, True) for value, text, image in zip(ids, text_trainable, image_trainable))
    return WeakSupervisionArtifacts(np.asarray(["a", "b", "c", "d"]), np.asarray(["train", "train", "val", "val"]), ids,
        np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int8), np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int8),
        np.zeros((4, 2), dtype=np.float32), np.asarray(text_trainable, dtype=bool), np.asarray(image_trainable, dtype=bool), np.asarray([1, 1], dtype=bool), 1, props, {}, {"labels": "hash"})


def collate(rows):
    return {"case_id": [row["case_id"] for row in rows], "caption": [row["caption"] for row in rows], "image": torch.tensor([[row["value"]] for row in rows]), "disease_label": torch.tensor([99 for _ in rows])}


def loader(ids):
    return DataLoader([{"case_id": case_id, "caption": "caption", "value": index} for index, case_id in enumerate(ids)], batch_size=2, collate_fn=collate)


def tokenizer(captions):
    return {"input_ids": torch.ones((len(captions), 1), dtype=torch.long)}


def test_trainer_writes_stage1_outputs_and_resumes_rng(tmp_path: Path) -> None:
    trainer = Stage1Trainer(model=EvidenceModel(), tokenizer=tokenizer, weak_supervision=artifacts(), loss_fn=MaskedBCELoss(torch.ones(2), torch.ones(2)), config=TrainerConfig(early_stopping_patience=2, selection_min_present=1, selection_min_absent=1, config={"stage": 1}), output_dir=tmp_path)
    trainer.fit(loader(["a", "b"]), loader(["c", "d"]), max_epochs=1, test_dataloader=loader(["c", "d"]))
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "predictions" / "val_best.npz").exists()
    prediction = np.load(tmp_path / "predictions" / "val_best.npz")
    assert set(("case_id", "proposition_id", "score_text", "score_image", "score_fused", "label_text", "label_img", "text_valid_mask", "image_valid_mask", "text_trainable", "image_trainable", "active_scoring")).issubset(prediction.files)
    random.seed(77); np.random.seed(77); torch.manual_seed(77)
    trainer.save_checkpoint("rng.pt")
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    trainer.load_checkpoint(tmp_path / "checkpoints" / "rng.pt")
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected


def test_metrics_exclude_untrainable_modality_propositions() -> None:
    trainer = Stage1Trainer(
        model=EvidenceModel(), tokenizer=tokenizer,
        weak_supervision=artifacts(text_trainable=(True, False), image_trainable=(False, True)),
        loss_fn=MaskedBCELoss(torch.ones(2), torch.ones(2)),
        config=TrainerConfig(selection_min_present=1, selection_min_absent=1),
    )
    metrics = trainer._metrics({
        "score_text": np.asarray([[0.1, 0.9], [0.9, 0.1]]),
        "score_image": np.asarray([[0.1, 0.9], [0.9, 0.1]]),
        "label_text": np.asarray([[0, 1], [1, 0]], dtype=np.int8),
        "label_img": np.asarray([[0, 1], [1, 0]], dtype=np.int8),
    })
    assert [row["eligible"] for row in metrics["text"]["rows"]] == [True, False]
    assert [row["eligible"] for row in metrics["image"]["rows"]] == [False, True]
    assert metrics["selection_macro_auc"] == 1.0
    assert metrics["selection"] == {
        "source": "image", "metric": "macro_roc_auc", "min_present": 1, "min_absent": 1, "n_eligible": 1,
    }

def test_query_control_auc_uses_each_modalitys_weak_labels() -> None:
    scores = {
        "score_image": np.asarray([[0.1], [0.9]]),
        "score_text": np.asarray([[0.9], [0.1]]),
        "score_fused": np.asarray([[0.5], [0.5]]),
    }
    result = _modality_macro_aucs(
        scores,
        label_text=np.asarray([[0], [1]], dtype=np.int8),
        label_image=np.asarray([[0], [1]], dtype=np.int8),
        text_trainable=np.asarray([True]),
        image_trainable=np.asarray([True]),
        min_present=1,
        min_absent=1,
    )
    assert result == {"image_macro_auc": 1.0, "text_macro_auc": 0.0}


def test_default_strict_selection_excludes_insufficient_validation_labels() -> None:
    trainer = Stage1Trainer(
        model=EvidenceModel(), tokenizer=tokenizer, weak_supervision=artifacts(),
        loss_fn=MaskedBCELoss(torch.ones(2), torch.ones(2)),
    )
    metrics = trainer._metrics({
        "score_text": np.asarray([[0.1, 0.9], [0.9, 0.1]]),
        "score_image": np.asarray([[0.1, 0.9], [0.9, 0.1]]),
        "label_text": np.asarray([[0, 1], [1, 0]], dtype=np.int8),
        "label_img": np.asarray([[0, 1], [1, 0]], dtype=np.int8),
    })
    assert metrics["text"]["n_eligible"] == 0
    assert metrics["image"]["n_eligible"] == 0
    assert metrics["selection_macro_auc"] is None

def test_optimizer_rejects_frozen_parameters() -> None:
    model = EvidenceModel()
    model.linear.weight.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="requires_grad"):
        Stage1Trainer(model=model, tokenizer=tokenizer, weak_supervision=artifacts(), loss_fn=MaskedBCELoss(torch.ones(2), torch.ones(2)), optimizer=optimizer)

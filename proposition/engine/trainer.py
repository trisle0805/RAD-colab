"""Reusable Stage-1 epoch trainer and reproducible checkpointing.

This module accepts only proposition evidence targets. Dataset disease labels are
explicitly discarded before model, loss, metric, or checkpoint selection calls.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from proposition.data.weak_supervision import UNKNOWN_LABEL, WeakSupervisionArtifacts
from proposition.losses.masked_bce import MaskedBCELoss, TextTargetSemantics
from .diagnostics import proposition_correlation, within_patient_differentiation
from .metrics import evaluate_modalities, modality_metrics


@dataclass
class TrainerConfig:
    device: str = "cpu"
    early_stopping_patience: int = 10
    score_threshold: float = 0.5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    training_scope: str = "full_stage1"
    text_target_semantics: TextTargetSemantics = "ternary_masked"
    max_grad_norm: float | None = None

    # Strict weak-label eligibility for checkpoint selection/evaluation
    selection_min_present: int = 5
    selection_min_absent: int = 5

    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EpochResult:
    epoch: int
    split: str
    loss: float | None
    metrics: dict[str, Any]
    predictions: dict[str, np.ndarray]


class Stage1Trainer:
    """Train and evaluate proposition evidence without disease supervision.

    Dataloaders must yield mappings containing ``case_id``, ``image``, and
    ``caption``. Extra keys, including ``disease_label``, are never read.
    ``tokenizer`` turns caption strings into tensors accepted by the model.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Callable[[list[str]], Mapping[str, Tensor]],
        weak_supervision: WeakSupervisionArtifacts,
        loss_fn: MaskedBCELoss,
        optimizer: torch.optim.Optimizer | None = None,
        config: TrainerConfig | None = None,
        output_dir: str | Path | None = None,
        inference_only: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.artifacts = weak_supervision
        self.loss_fn = loss_fn
        self.config = config or TrainerConfig()
        self.inference_only = inference_only
        if self.config.training_scope not in {"full_stage1", "text_only", "text_adapt"}:
            raise ValueError("unsupported training_scope")
        if _is_text_evidence_only_scope(self.config.training_scope) and self.config.text_target_semantics != "caption_support_binary":
            raise ValueError("text-only evidence scopes require caption_support_binary text targets")
        if self.config.max_grad_norm is not None and self.config.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when supplied")
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.active_indices = weak_supervision.active_indices
        self.proposition_ids = tuple(weak_supervision.proposition_ids[self.active_indices].tolist())
        if tuple(getattr(model, "active_proposition_ids", self.proposition_ids)) != self.proposition_ids:
            raise ValueError("model active proposition IDs do not match weak-supervision artifacts")
        self.text_trainable = torch.as_tensor(weak_supervision.text_trainable[self.active_indices], dtype=torch.bool, device=self.device)
        self.image_trainable = torch.as_tensor(weak_supervision.image_trainable[self.active_indices], dtype=torch.bool, device=self.device)
        self.optimizer = optimizer
        if self.optimizer is None and not inference_only:
            self.optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.optimizer is None and not inference_only:
            raise RuntimeError("optimizer construction failed")
        if inference_only:
            self.output_dir = Path(output_dir) if output_dir is not None else None
            self.epoch = 0
            self.best_metric = float("-inf")
            self.best_epoch = None
            self.best_loss = None
            self.bad_epochs = 0
            self.history = []
            if self.output_dir is not None:
                self._prepare_output()
            return
        if any(not parameter.requires_grad for group in self.optimizer.param_groups for parameter in group["params"]):
            raise ValueError("optimizer must contain only requires_grad parameters")
        optimizer_ids = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
        expected_ids = {
            id(parameter)
            for name in ("text_cross_attention", "text_head")
            for parameter in getattr(model, name).parameters()
        } if self.config.training_scope == "text_only" else {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        if optimizer_ids != expected_ids:
            raise ValueError("optimizer parameters do not match the requested training scope")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.epoch = 0
        self.best_metric = float("-inf")
        self.best_epoch: int | None = None
        self.best_loss: float | None = None
        self.bad_epochs = 0
        self.history: list[dict[str, Any]] = []
        if self.output_dir is not None:
            self._prepare_output()

    def train_epoch(self, dataloader: Iterable[Mapping[str, Any]], *, epoch: int | None = None) -> EpochResult:
        """Perform one optimizer epoch using weak labels looked up by ``case_id``."""
        if self.inference_only:
            raise RuntimeError("inference-only trainer cannot train")
        self.model.train()
        return self._run_epoch(dataloader, split="train", training=True, epoch=self.epoch + 1 if epoch is None else epoch)

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable[Mapping[str, Any]], *, split: str = "val", epoch: int | None = None) -> EpochResult:
        """Evaluate evidence predictions against weak proposition labels only."""
        self.model.eval()
        return self._run_epoch(dataloader, split=split, training=False, epoch=self.epoch if epoch is None else epoch)

    def fit(
        self,
        train_dataloader: Iterable[Mapping[str, Any]],
        validation_dataloader: Iterable[Mapping[str, Any]],
        *,
        max_epochs: int,
        test_dataloader: Iterable[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run train/validation epochs, select by validation proposition macro AUC."""
        if self.inference_only:
            raise RuntimeError("inference-only trainer cannot fit")
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        for epoch in range(self.epoch + 1, max_epochs + 1):
            train = self.train_epoch(train_dataloader, epoch=epoch)
            validation = self.evaluate(validation_dataloader, split="val", epoch=epoch)
            selection = _selection_metric(validation.metrics)
            selection_key = (
                "selection_text_macro_auc"
                if _is_text_evidence_only_scope(self.config.training_scope)
                else "selection_macro_auc"
            )
            record = {
                "epoch": epoch,
                "train_loss": train.loss,
                "val_loss": validation.loss,
                selection_key: selection,
            }
            if _is_text_evidence_only_scope(self.config.training_scope):
                record["validation_text_bce"] = validation.loss
            self.history.append(record)
            self.epoch = epoch
            improved = selection is not None and (
                selection > self.best_metric
                or (
                    np.isclose(selection, self.best_metric)
                    and validation.loss is not None
                    and (self.best_loss is None or validation.loss < self.best_loss)
                )
            )
            if improved:
                self.best_metric, self.best_epoch, self.best_loss, self.bad_epochs = float(selection), epoch, validation.loss, 0
                self._save_best_outputs(validation)
                self.save_checkpoint("best.pt")
            else:
                self.bad_epochs += 1
            self.save_checkpoint("latest.pt")
            self._write_history()
            if self.bad_epochs >= self.config.early_stopping_patience:
                break
        if self.best_epoch is not None:
            self.load_checkpoint(self._checkpoint_path("best.pt"), load_optimizer=False)
            self._write_diagnostics(self.evaluate(validation_dataloader, split="val", epoch=self.best_epoch), validation_dataloader)
            if test_dataloader is not None:
                self._save_predictions(self.evaluate(test_dataloader, split="test", epoch=self.best_epoch), "test_best.npz")
        return self.history

    def _run_epoch(self, dataloader: Iterable[Mapping[str, Any]], *, split: str, training: bool, epoch: int) -> EpochResult:
        losses: list[float] = []
        collected: dict[str, list[np.ndarray]] = {key: [] for key in ("case_id", "score_text", "score_image", "score_fused", "label_text", "label_img")}
        for batch in dataloader:
            case_ids, images, captions = self._batch_inputs(batch)
            labels_text, labels_image = self._weak_labels(case_ids)
            tokens = {key: value.to(self.device) for key, value in self.tokenizer(captions).items()}
            outputs = self.model(images, tokens)
            loss = self.loss_fn(
                outputs["logits_txt"],
                outputs["logits_img"],
                labels_text,
                labels_image,
                self.text_trainable,
                self.image_trainable,
                include_image=not _is_text_evidence_only_scope(self.config.training_scope),
            )
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for group in self.optimizer.param_groups for parameter in group["params"]],
                        self.config.max_grad_norm,
                    )
                self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            collected["case_id"].append(np.asarray(case_ids, dtype=str))
            collected["score_text"].append(outputs["scores_txt"].detach().cpu().numpy())
            collected["score_image"].append(outputs["scores_img"].detach().cpu().numpy())
            collected["score_fused"].append(outputs["support_scores"].detach().cpu().numpy())
            collected["label_text"].append(labels_text.detach().cpu().numpy())
            collected["label_img"].append(labels_image.detach().cpu().numpy())
        predictions = {key: (np.concatenate(values) if values else np.empty((0,), dtype=str if key == "case_id" else np.float32)) for key, values in collected.items()}
        metrics = self._metrics(predictions)
        return EpochResult(epoch, split, float(np.mean(losses)) if losses else None, metrics, predictions)

    def _batch_inputs(self, batch: Mapping[str, Any]) -> tuple[list[str], Tensor, list[str]]:
        try:
            case_ids = [str(item) for item in batch["case_id"]]
            images = batch["image"].to(self.device)
            captions = [str(item) for item in batch["caption"]]
        except (KeyError, AttributeError, TypeError) as error:
            raise ValueError("batch must provide case_id, image, and caption") from error
        if len(case_ids) != len(captions) or images.shape[0] != len(case_ids):
            raise ValueError("batch case_id, image, and caption lengths must align")
        return case_ids, images, captions

    def _weak_labels(self, case_ids: list[str]) -> tuple[Tensor, Tensor]:
        indices = self.artifacts.case_indices(case_ids)
        text = torch.as_tensor(self.artifacts.label_text[indices][:, self.active_indices], device=self.device)
        image = torch.as_tensor(self.artifacts.label_img[indices][:, self.active_indices], device=self.device)
        return text, image

    def _metrics(
        self,
        prediction: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:

        outputs = {
            "text": prediction["score_text"],
            "image": prediction["score_image"],
        }

        labels = {
            "text": prediction["label_text"],
            "image": prediction["label_img"],
        }

        metric_kwargs = {
            "proposition_ids": self.proposition_ids,
            "threshold": self.config.score_threshold,
            "min_present": self.config.selection_min_present,
            "min_absent": self.config.selection_min_absent,
        }

        metrics = {
            "text": evaluate_modalities(
                {
                    "text": outputs["text"],
                },
                {
                    "text": labels["text"],
                },
                valid_propositions=(
                    self.text_trainable
                    .detach()
                    .cpu()
                    .numpy()
                ),
                text_target_semantics=self.config.text_target_semantics,
                **metric_kwargs,
            )["text"],

            "image": evaluate_modalities(
                {
                    "image": outputs["image"],
                },
                {
                    "image": labels["image"],
                },
                valid_propositions=(
                    self.image_trainable
                    .detach()
                    .cpu()
                    .numpy()
                ),
                **metric_kwargs,
            )["image"],
        }

        # --------------------------------------------------------
        # IMPORTANT:
        # Checkpoint selection uses IMAGE branch only.
        #
        # Text metrics are still reported, but text does not
        # influence best-checkpoint selection because its weak
        # labels currently contain too few reliable negatives.
        # --------------------------------------------------------

        if _is_text_evidence_only_scope(self.config.training_scope):
            metrics["selection_text_macro_auc"] = metrics["text"]["macro_roc_auc"]
            metrics["selection_macro_auc"] = metrics["selection_text_macro_auc"]
            selection_source = "text"
            selection_metrics = metrics["text"]
        else:
            metrics["selection_macro_auc"] = metrics["image"]["macro_roc_auc"]
            selection_source = "image"
            selection_metrics = metrics["image"]

        metrics["selection"] = {
            "source": selection_source,
            "metric": "macro_roc_auc",
            "min_present": (
                self.config.selection_min_present
            ),
            "min_absent": (
                self.config.selection_min_absent
            ),
            "n_eligible": (
                selection_metrics["n_eligible"]
            ),
        }

        return metrics

    def _prepare_output(self) -> None:
        for name in ("checkpoints", "metrics", "predictions", "diagnostics", "supervision"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        self.write_run_config()

    def write_run_config(self) -> None:
        """Persist the current immutable-by-convention experiment configuration."""
        if self.output_dir is not None:
            (self.output_dir / "run_config.json").write_text(
                json.dumps(dict(self.config.config), indent=2, sort_keys=True), encoding="utf-8"
            )

    def _save_best_outputs(self, result: EpochResult) -> None:
        if self.output_dir is None:
            return
        self._save_predictions(result, "val_best.npz")
        metrics_dir = self.output_dir / "metrics"
        (metrics_dir / "best_validation.json").write_text(json.dumps({"epoch": result.epoch, "loss": result.loss, "metrics": result.metrics}, indent=2), encoding="utf-8")
        self._write_metric_tables(result.metrics)

    @torch.no_grad()
    def _write_diagnostics(self, result: EpochResult, dataloader: Iterable[Mapping[str, Any]]) -> None:
        """Write disease-label-free score and query-control diagnostics for best validation model."""
        if self.output_dir is None:
            return
        diagnostics_dir = self.output_dir / "diagnostics"
        baseline = {
            "score_image": result.predictions["score_image"],
            "score_text": result.predictions["score_text"],
            "score_fused": result.predictions["score_fused"],
        }
        text_trainable = self.text_trainable.detach().cpu().numpy()
        image_trainable = self.image_trainable.detach().cpu().numpy()
        summary = {
            "baseline": _modality_macro_aucs(
                baseline,
                result.predictions["label_text"],
                result.predictions["label_img"],
                text_trainable,
                image_trainable,
                min_present=self.config.selection_min_present,
                min_absent=self.config.selection_min_absent,
                text_target_semantics=self.config.text_target_semantics,
            ),
            "within_patient_differentiation": within_patient_differentiation(baseline["score_fused"]),
            "proposition_correlation": proposition_correlation(baseline["score_fused"]),
        }
        queries = getattr(self.model, "_queries")(self.device).detach()
        controls: dict[str, np.ndarray] = {}
        for name, query_embeddings in _query_controls(queries):
            control = self._predict_with_queries(dataloader, query_embeddings)
            for score_name, scores in control.items():
                controls[f"query_{name}_{score_name}"] = scores
            summary[f"query_{name}"] = {
                **_modality_macro_aucs(
                    control,
                    result.predictions["label_text"],
                    result.predictions["label_img"],
                    text_trainable,
                    image_trainable,
                    min_present=self.config.selection_min_present,
                    min_absent=self.config.selection_min_absent,
                    text_target_semantics=self.config.text_target_semantics,
                ),
                "mean_absolute_score_delta": float(
                    np.nanmean(np.abs(control["score_fused"] - baseline["score_fused"]))
                ),
                "within_patient_differentiation": within_patient_differentiation(
                    control["score_fused"]
                ),
            }
        np.savez_compressed(
            diagnostics_dir / "query_controls.npz",
            **{f"baseline_{name}": scores for name, scores in baseline.items()},
            **controls,
        )
        (diagnostics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    @torch.no_grad()
    def save_attention_samples(self, dataloader: Iterable[Mapping[str, Any]], *, count: int) -> Path | None:
        """Persist a deterministic prefix of validation cross-attention matrices."""
        if self.output_dir is None or count < 1:
            return None
        remaining = count
        written: Path | None = None
        self.model.eval()
        for batch_index, batch in enumerate(dataloader):
            case_ids, images, captions = self._batch_inputs(batch)
            take = min(remaining, len(case_ids))
            tokens = {key: value.to(self.device) for key, value in self.tokenizer(captions).items()}
            outputs = self.model(images, tokens, return_attention=True)
            attention = outputs["attention"]
            assert isinstance(attention, Mapping)
            destination = self.output_dir / "diagnostics" / f"attention_samples_{batch_index:03d}.npz"
            image = attention["image"][:take].detach().cpu().numpy()
            text = attention["text"][:take].detach().cpu().numpy()
            np.savez_compressed(destination, image_attention=image, text_attention=text,
                                case_id=np.asarray(case_ids[:take]), proposition_id=np.asarray(self.proposition_ids))
            written = written or destination
            remaining -= take
            if not remaining:
                break
        return written

    @torch.no_grad()
    def _predict_with_queries(
        self, dataloader: Iterable[Mapping[str, Any]], query_embeddings: Tensor
    ) -> dict[str, np.ndarray]:
        scores = {"score_image": [], "score_text": [], "score_fused": []}
        self.model.eval()
        for batch in dataloader:
            _, images, captions = self._batch_inputs(batch)
            tokens = {key: value.to(self.device) for key, value in self.tokenizer(captions).items()}
            outputs = self.model(images, tokens, query_embeddings=query_embeddings)
            scores["score_image"].append(outputs["scores_img"].detach().cpu().numpy())
            scores["score_text"].append(outputs["scores_txt"].detach().cpu().numpy())
            scores["score_fused"].append(outputs["support_scores"].detach().cpu().numpy())
        empty = np.empty((0, len(self.proposition_ids)), dtype=np.float32)
        return {name: np.concatenate(values) if values else empty for name, values in scores.items()}

    def _save_predictions(self, result: EpochResult, filename: str) -> None:
        if self.output_dir is None:
            return
        predictions = result.predictions
        payload = dict(predictions)
        payload.update({
            "proposition_id": np.asarray(self.proposition_ids),
            "text_valid_mask": (
                (predictions["label_text"] != UNKNOWN_LABEL)
                if self.config.text_target_semantics == "ternary_masked"
                else np.ones_like(predictions["label_text"], dtype=bool)
            ) & self.text_trainable.detach().cpu().numpy(),
            "image_valid_mask": (predictions["label_img"] != UNKNOWN_LABEL) & self.image_trainable.detach().cpu().numpy(),
            "text_trainable": self.text_trainable.detach().cpu().numpy(),
            "image_trainable": self.image_trainable.detach().cpu().numpy(),
            "active_scoring": np.ones(len(self.proposition_ids), dtype=bool),
        })
        np.savez_compressed(self.output_dir / "predictions" / filename, **payload)

    def _write_metric_tables(self, metrics: Mapping[str, Any]) -> None:
        assert self.output_dir is not None
        rows = [dict(row, modality=modality) for modality in ("text", "image") for row in metrics[modality]["rows"]]
        with (self.output_dir / "metrics" / "proposition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)
        summary = {key: value for key, value in metrics.items() if key != "text" and key != "image"}
        summary["text"] = {key: value for key, value in metrics["text"].items() if key != "rows"}
        summary["image"] = {key: value for key, value in metrics["image"].items() if key != "rows"}
        (self.output_dir / "metrics" / "modality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _write_history(self) -> None:
        if self.output_dir is None or not self.history:
            return
        with (self.output_dir / "metrics" / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            selection_field = "selection_text_macro_auc" if _is_text_evidence_only_scope(self.config.training_scope) else "selection_macro_auc"
            fieldnames = ("epoch", "train_loss", "val_loss", selection_field)
            if _is_text_evidence_only_scope(self.config.training_scope):
                fieldnames += ("validation_text_bce",)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(self.history)

    def _checkpoint_path(self, name: str) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir is required for checkpointing")
        return self.output_dir / "checkpoints" / name

    def save_checkpoint(self, name: str = "latest.pt") -> Path:
        path = self._checkpoint_path(name)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "config": dict(self.config.config),
            "rng_state": _rng_state(),
            "proposition_ids": self.proposition_ids,
            "artifact_hashes": dict(self.artifacts.artifact_hashes),
            "git_commit": _git_commit(),
        }
        if self.config.training_scope == "text_adapt":
            anchor = getattr(self.model, "query_cache", None)
            if not isinstance(anchor, Tensor) or anchor.numel() == 0:
                raise RuntimeError("Round-B checkpoint requires a fixed query anchor")
            anchor_cpu = anchor.detach().cpu().contiguous()
            payload["query_anchor"] = anchor_cpu
            payload["query_anchor_sha256"] = _tensor_sha256(anchor_cpu)

        tmp_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        torch.save(
            payload,
            tmp_path,
        )

        # Verify file vừa ghi có thực sự đọc được
        torch.load(
            tmp_path,
            map_location="cpu",
            weights_only=False,
        )

        # Chỉ thay file chính sau khi verify thành công
        tmp_path.replace(path)
        return path

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if tuple(checkpoint["proposition_ids"]) != self.proposition_ids:
            raise ValueError("checkpoint proposition ID order does not match active artifacts")
        if dict(checkpoint.get("artifact_hashes", {})) != dict(self.artifacts.artifact_hashes):
            raise ValueError("checkpoint artifact hashes do not match active artifacts")
        self.model.load_state_dict(checkpoint["model_state"])
        if self.config.training_scope == "text_adapt":
            anchor = checkpoint.get("query_anchor")
            expected_hash = checkpoint.get("query_anchor_sha256")
            if not isinstance(anchor, Tensor) or not isinstance(expected_hash, str):
                raise ValueError("Round-B checkpoint is missing its query anchor")
            if _tensor_sha256(anchor) != expected_hash:
                raise ValueError("Round-B checkpoint query anchor hash does not match")
            restore_anchor = getattr(self.model, "restore_query_anchor", None)
            if not callable(restore_anchor):
                raise ValueError("Round-B model cannot restore a query anchor")
            restore_anchor(anchor)
        if load_optimizer:
            if self.inference_only:
                raise RuntimeError("inference-only trainer cannot load optimizer state")
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epoch, self.best_metric, self.best_epoch = checkpoint["epoch"], checkpoint["best_metric"], checkpoint["best_epoch"]
        self.best_loss = checkpoint.get("best_loss")
        self.bad_epochs, self.history = checkpoint.get("bad_epochs", 0), checkpoint.get("history", [])
        _restore_rng_state(checkpoint["rng_state"])

    def load_initial_weights(self, path: str | Path) -> None:
        """Load only model weights for a fresh retraining experiment."""
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if tuple(checkpoint["proposition_ids"]) != self.proposition_ids:
            raise ValueError("checkpoint proposition ID order does not match active artifacts")
        if dict(checkpoint.get("artifact_hashes", {})) != dict(self.artifacts.artifact_hashes):
            raise ValueError("checkpoint artifact hashes do not match active artifacts")
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch = 0
        self.best_metric = float("-inf")
        self.best_epoch = None
        self.best_loss = None
        self.bad_epochs = 0
        self.history = []

    @torch.no_grad()
    def export_best_predictions(
        self,
        loaders: Mapping[str, Iterable[Mapping[str, Any]]],
    ) -> None:
        """Export deterministic split predictions from the currently loaded best model."""
        for split, loader in loaders.items():
            result = self.evaluate(loader, split=split, epoch=self.best_epoch or 0)
            self._save_predictions(result, f"scores_{split}_best.npz")


def _modality_macro_aucs(
    scores: Mapping[str, np.ndarray],
    label_text: np.ndarray,
    label_image: np.ndarray,
    text_trainable: np.ndarray,
    image_trainable: np.ndarray,
    *,
    min_present: int = 1,
    min_absent: int = 1,
    text_target_semantics: str = "ternary_masked",
) -> dict[str, float | None]:
    """Evaluate each modality against its own weak-label target."""
    image = modality_metrics(
        scores["score_image"],
        label_image,
        valid_propositions=image_trainable,
        min_present=min_present,
        min_absent=min_absent,
    )
    text = modality_metrics(
        scores["score_text"],
        label_text,
        valid_propositions=text_trainable,
        min_present=min_present,
        min_absent=min_absent,
        target_semantics=text_target_semantics,
    )
    return {
        "image_macro_auc": image["macro_roc_auc"],
        "text_macro_auc": text["macro_roc_auc"],
    }


def _query_controls(queries: Tensor) -> list[tuple[str, Tensor]]:
    """Deterministic query controls for diagnostic-only evidence comparisons."""
    generator = torch.Generator(device=queries.device).manual_seed(0)
    return [
        ("swap", queries.flip(0)),
        ("zero", torch.zeros_like(queries)),
        ("mean", queries.mean(dim=0, keepdim=True).expand_as(queries)),
        ("random", torch.randn(queries.shape, generator=generator, device=queries.device, dtype=queries.dtype)),
    ]

def _selection_metric(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("selection_macro_auc")
    return None if value is None or not np.isfinite(value) else float(value)


def _is_text_evidence_only_scope(training_scope: str) -> bool:
    return training_scope in {"text_only", "text_adapt"}


def _tensor_sha256(value: Tensor) -> str:
    import hashlib

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # torch.set_rng_state requires a CPU ByteTensor.
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = (
            torch_state
            .detach()
            .cpu()
            .to(dtype=torch.uint8)
        )

    torch.set_rng_state(torch_state)

    if "cuda" in state and torch.cuda.is_available():
        cuda_states = []

        for cuda_state in state["cuda"]:
            if isinstance(cuda_state, torch.Tensor):
                cuda_state = (
                    cuda_state
                    .detach()
                    .cpu()
                    .to(dtype=torch.uint8)
                )

            cuda_states.append(cuda_state)

        torch.cuda.set_rng_state_all(cuda_states)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

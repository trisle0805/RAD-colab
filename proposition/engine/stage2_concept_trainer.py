"""Trainer and immutable-run exports for learned Stage-2 concept heads."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional

from proposition.data.stage2_scores import Stage2Dataset
from proposition.engine.stage2_concept_metrics import class_indices, single_label_diagnosis_metrics
from proposition.models.stage2_concept_head import Stage2ConceptHead


@dataclass
class Stage2ConceptTrainerConfig:
    device: str = "cpu"
    learning_rate: float = 1e-2
    regularization: str = "none"
    lambda_reg: float = 0.0
    max_epochs: int = 100
    early_stopping_patience: int = 10
    batch_size: int = 64
    selection_metric: str = "top_1_accuracy"


class Stage2ConceptTrainer:
    """Fit linear concept weights using multiclass CE over frozen scalar scores."""

    _SUPPORTED_SELECTION = frozenset(("top_1_accuracy", "cross_entropy"))

    def __init__(
        self,
        *,
        model: Stage2ConceptHead,
        disease_ids: Sequence[str],
        proposition_ids: Sequence[str],
        routing_mode: str,
        input_hashes: Mapping[str, str],
        score_source: Sequence[str],
        config: Stage2ConceptTrainerConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or Stage2ConceptTrainerConfig()
        if self.config.regularization not in {"none", "l1", "l2"} or self.config.lambda_reg < 0:
            raise ValueError("regularization must be none/l1/l2 and lambda_reg must be non-negative")
        if self.config.selection_metric not in self._SUPPORTED_SELECTION:
            raise ValueError("selection_metric must be top_1_accuracy or cross_entropy")
        if min(self.config.max_epochs, self.config.early_stopping_patience, self.config.batch_size) < 1:
            raise ValueError("max_epochs, early_stopping_patience, and batch_size must be positive")
        self.model, self.device = model.to(self.config.device), torch.device(self.config.device)
        self.disease_ids, self.proposition_ids = tuple(disease_ids), tuple(proposition_ids)
        if len(self.disease_ids) != model.num_diseases or len(set(self.disease_ids)) != len(self.disease_ids) or len(set(self.proposition_ids)) != len(self.proposition_ids):
            raise ValueError("model and identifier contracts disagree")
        self.routing_mode, self.input_hashes = routing_mode, dict(input_hashes)
        self.score_source = tuple(score_source)
        if len(self.score_source) != len(self.proposition_ids):
            raise ValueError("score_source must align with proposition_ids")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.epoch, self.best_epoch, self.bad_epochs = 0, None, 0
        self.best_metric = float("inf") if self.config.selection_metric == "cross_entropy" else float("-inf")
        self.history: list[dict[str, Any]] = []
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self._prepare_output()

    def fit(self, train: Stage2Dataset, validation: Stage2Dataset) -> list[dict[str, Any]]:
        self._validate_dataset(train)
        self._validate_dataset(validation)
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_epoch(train)
            validation_result = self.evaluate(validation)
            value = float(validation_result["metrics"][self.config.selection_metric])
            record = {"epoch": epoch, "train_loss": train_loss, "val_cross_entropy": validation_result["loss"], "selection_metric": value}
            self.history.append(record)
            self.epoch = epoch
            if self._is_improved(value):
                self.best_metric, self.best_epoch, self.bad_epochs = value, epoch, 0
                self.save_checkpoint("best.pt")
                self._write_json("metrics/best_validation.json", {"epoch": epoch, "loss": validation_result["loss"], "metrics": validation_result["metrics"]})
            else:
                self.bad_epochs += 1
            self.save_checkpoint("latest.pt")
            self._write_rows("metrics/history.csv", self.history)
            if self.bad_epochs >= self.config.early_stopping_patience:
                break
        if self.best_epoch is None:
            raise RuntimeError("validation never produced a selected checkpoint")
        completed = (self.epoch, self.bad_epochs, self.history)
        self.load_checkpoint(self._checkpoint_path("best.pt"), load_optimizer=False)
        self.epoch, self.bad_epochs, self.history = completed
        return self.history

    def evaluate(self, dataset: Stage2Dataset) -> dict[str, Any]:
        self._validate_dataset(dataset)
        self.model.eval()
        support, targets = self._tensors(dataset)
        with torch.no_grad():
            outputs = self.model.contribution_outputs(support)
            loss = functional.cross_entropy(outputs["logits"], targets)
        arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items()}
        metrics = single_label_diagnosis_metrics(arrays["logits"], dataset.targets, self.disease_ids)
        return {"loss": float(loss.cpu()), "metrics": metrics, "outputs": arrays}

    def export_best_split(self, dataset: Stage2Dataset, split: str) -> dict[str, Any]:
        if self.best_epoch is None:
            raise RuntimeError("fit must select a checkpoint before export")
        result = self.evaluate(dataset)
        if self.output_dir:
            self._save_predictions(dataset, result["outputs"], split)
            self._save_contributions(dataset, result["outputs"], split)
            metrics = result["metrics"]
            self._write_json(f"metrics/{split}_summary.json", {"epoch": self.best_epoch, "loss": result["loss"], "metrics": metrics})
            self._write_rows(f"metrics/{split}_per_disease.csv", metrics["per_disease"])
            if split == "test":
                np.save(self.output_dir / "metrics" / "confusion_matrix_test.npy", metrics["confusion_matrix"])
        return result

    def save_checkpoint(self, name: str) -> Path:
        path = self._checkpoint_path(name)
        payload = {"model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(), "epoch": self.epoch,
                   "best_epoch": self.best_epoch, "best_metric": self.best_metric, "bad_epochs": self.bad_epochs, "history": self.history,
                   "config": asdict(self.config), "disease_ids": self.disease_ids, "proposition_ids": self.proposition_ids,
                   "routing_mode": self.routing_mode, "input_hashes": self.input_hashes}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=False)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if tuple(checkpoint["disease_ids"]) != self.disease_ids or tuple(checkpoint["proposition_ids"]) != self.proposition_ids:
            raise ValueError("checkpoint identifiers do not match current run")
        if checkpoint["routing_mode"] != self.routing_mode or dict(checkpoint["input_hashes"]) != self.input_hashes:
            raise ValueError("checkpoint routing or input hashes do not match current run")
        self.model.load_state_dict(checkpoint["model_state"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epoch, self.best_epoch, self.best_metric, self.bad_epochs, self.history = (checkpoint[key] for key in ("epoch", "best_epoch", "best_metric", "bad_epochs", "history"))

    def _train_epoch(self, dataset: Stage2Dataset) -> float:
        self.model.train()
        support, targets = self._tensors(dataset)
        losses: list[float] = []
        for start in range(0, len(dataset.case_ids), self.config.batch_size):
            batch_support, batch_targets = support[start:start + self.config.batch_size], targets[start:start + self.config.batch_size]
            outputs = self.model(batch_support)
            loss = functional.cross_entropy(outputs["logits"], batch_targets)
            loss = loss + self.config.lambda_reg * self.model.regularization_penalty(self.config.regularization)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def _is_improved(self, value: float) -> bool:
        return value < self.best_metric if self.config.selection_metric == "cross_entropy" else value > self.best_metric

    def _tensors(self, dataset: Stage2Dataset) -> tuple[Tensor, Tensor]:
        return (torch.as_tensor(dataset.support_scores, dtype=torch.float32, device=self.device),
                torch.as_tensor(class_indices(dataset.targets, len(self.disease_ids)), dtype=torch.long, device=self.device))

    def _validate_dataset(self, dataset: Stage2Dataset) -> None:
        if dataset.support_scores.ndim != 2 or len(dataset.case_ids) != len(dataset.support_scores) or dataset.support_scores.shape[1] != len(self.proposition_ids):
            raise ValueError("dataset support scores differ from model contracts")
        class_indices(dataset.targets, len(self.disease_ids))
        mask = self.model.proposition_mask.detach().cpu().numpy()
        if not np.all(np.isfinite(dataset.support_scores[:, mask])):
            raise ValueError("included support scores must be finite")

    def _prepare_output(self) -> None:
        assert self.output_dir is not None
        for name in ("checkpoints", "metrics", "predictions", "contributions", "diagnostics"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        self._write_json("run_config.json", asdict(self.config))

    def _checkpoint_path(self, name: str) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir is required for checkpointing")
        return self.output_dir / "checkpoints" / name

    def _save_predictions(self, dataset: Stage2Dataset, output: Mapping[str, np.ndarray], split: str) -> None:
        assert self.output_dir is not None
        np.savez_compressed(self.output_dir / "predictions" / f"{split}_best.npz", case_id=dataset.case_ids, disease_id=np.asarray(self.disease_ids),
                            targets=class_indices(dataset.targets, len(self.disease_ids)), logits=output["logits"], probabilities=output["probabilities"],
                            bias=output["bias"], routing_mode=np.asarray(self.routing_mode))

    def _save_contributions(self, dataset: Stage2Dataset, output: Mapping[str, np.ndarray], split: str) -> None:
        assert self.output_dir is not None
        logits = output["logits"]
        ranking = np.argsort(-logits, axis=1, kind="mergesort")
        top1, top2 = ranking[:, 0], ranking[:, 1]
        contributions = output["contributions"]
        rows = np.arange(len(dataset.case_ids))
        top1_terms, top2_terms = contributions[rows, top1], contributions[rows, top2]
        np.savez_compressed(self.output_dir / "contributions" / f"{split}_contributions.npz", case_id=dataset.case_ids,
                            disease_id=np.asarray(self.disease_ids), proposition_id=np.asarray(self.proposition_ids), logits=logits,
                            probabilities=output["probabilities"], top1_index=top1, top2_index=top2, top1_logit=logits[rows, top1], top2_logit=logits[rows, top2],
                            margin=logits[rows, top1] - logits[rows, top2], contribution_top1=top1_terms, contribution_top2=top2_terms,
                            delta_contribution=top1_terms - top2_terms, bias=output["bias"], weights=output["weights"],
                            input_scores=output["input_scores"], support_scores=output["support_scores"], proposition_mask=self.model.proposition_mask.detach().cpu().numpy(), score_source=np.asarray(self.score_source))

    def _write_rows(self, relative: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not self.output_dir or not rows:
            return
        with (self.output_dir / relative).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, relative: str, value: Mapping[str, Any]) -> None:
        if self.output_dir:
            (self.output_dir / relative).write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
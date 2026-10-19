"""Training and immutable-run export for frozen Stage-2 diagnosis scores."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from proposition.data.stage2_scores import Stage2Dataset
from proposition.models.stage2_diagnosis import PropositionDiagnosisLayer
from .stage2_metrics import multilabel_diagnosis_metrics


@dataclass
class Stage2TrainerConfig:
    device: str = "cpu"
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    max_epochs: int = 100
    early_stopping_patience: int = 10
    selection_metric: str = "val_loss"
    threshold: float = 0.5
    batch_size: int = 64


class Stage2Trainer:
    """Fit only calibration parameters; Stage-1 evidence remains immutable input."""

    _SUPPORTED_SELECTION = frozenset(("val_loss", "macro_roc_auc", "macro_f1", "micro_f1", "balanced_accuracy"))

    def __init__(
        self,
        *,
        model: PropositionDiagnosisLayer,
        disease_ids: Sequence[str],
        proposition_ids: Sequence[str],
        routing_mode: str,
        input_hashes: Mapping[str, str],
        score_source: Sequence[str] | None = None,
        config: Stage2TrainerConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or Stage2TrainerConfig()
        if self.config.selection_metric not in self._SUPPORTED_SELECTION:
            raise ValueError(f"unsupported selection metric {self.config.selection_metric!r}")
        if self.config.max_epochs < 1 or self.config.early_stopping_patience < 1 or self.config.batch_size < 1:
            raise ValueError("max_epochs, early_stopping_patience, and batch_size must be positive")
        self.model, self.device = model.to(self.config.device), torch.device(self.config.device)
        self.disease_ids, self.proposition_ids = tuple(disease_ids), tuple(proposition_ids)
        if len(set(self.disease_ids)) != len(self.disease_ids) or len(set(self.proposition_ids)) != len(self.proposition_ids):
            raise ValueError("disease and proposition IDs must be unique")
        self.routing_mode, self.input_hashes = routing_mode, dict(input_hashes)
        self.score_source = tuple(score_source) if score_source is not None else tuple("unknown" for _ in self.proposition_ids)
        if len(self.score_source) != len(self.proposition_ids):
            raise ValueError("score_source must align with proposition IDs")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        expected = {id(parameter) for parameter in self.model.parameters() if parameter.requires_grad}
        actual = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
        if actual != expected or expected != {id(self.model.raw_gamma), id(self.model.bias)}:
            raise ValueError("optimizer must contain exactly raw_gamma and bias")
        self.epoch, self.best_epoch, self.best_metric, self.bad_epochs = 0, None, float("inf"), 0
        self.history: list[dict[str, Any]] = []
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self._prepare_output()

    def fit(self, train: Stage2Dataset, validation: Stage2Dataset) -> list[dict[str, Any]]:
        self._validate_dataset(train); self._validate_dataset(validation)
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_epoch(train)
            validation_result = self.evaluate(validation)
            value = self._selection_value(validation_result)
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": validation_result["loss"], "selection_metric": value}
            self.history.append(record); self.epoch = epoch
            improved = value is not None and np.isfinite(value) and self._is_improved(float(value))
            if improved:
                self.best_metric, self.best_epoch, self.bad_epochs = float(value), epoch, 0
                self.save_checkpoint("best.pt")
                self._write_json("metrics/best_validation.json", {"epoch": epoch, "loss": validation_result["loss"], "metrics": validation_result["metrics"]})
            else:
                self.bad_epochs += 1
            self.save_checkpoint("latest.pt"); self._write_history()
            if self.bad_epochs >= self.config.early_stopping_patience:
                break
        if self.best_epoch is None:
            raise RuntimeError("validation never produced a finite selection metric")
        completed_epoch, completed_bad_epochs, completed_history = self.epoch, self.bad_epochs, self.history
        self.load_checkpoint(self._checkpoint_path("best.pt"), load_optimizer=False)
        self.epoch, self.bad_epochs, self.history = completed_epoch, completed_bad_epochs, completed_history
        return self.history

    def _selection_value(self, validation_result: Mapping[str, Any]) -> float | None:
        if self.config.selection_metric == "val_loss":
            return float(validation_result["loss"])
        return validation_result["metrics"].get(self.config.selection_metric)

    def _is_improved(self, value: float) -> bool:
        if self.config.selection_metric == "val_loss":
            return value < self.best_metric
        return value > self.best_metric

    def evaluate(self, dataset: Stage2Dataset) -> dict[str, Any]:
        self._validate_dataset(dataset)
        self.model.eval()
        support, targets = self._tensors(dataset)
        with torch.no_grad():
            outputs = self.model.contribution_outputs(support)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], targets)
        probabilities = outputs["probabilities"].cpu().numpy()
        metrics = multilabel_diagnosis_metrics(probabilities, dataset.targets, self.disease_ids, threshold=self.config.threshold)
        return {"loss": float(loss.cpu()), "metrics": metrics, "outputs": {key: value.detach().cpu().numpy() for key, value in outputs.items()}}

    def export_best_split(self, dataset: Stage2Dataset, split: str) -> dict[str, Any]:
        if self.best_epoch is None:
            raise RuntimeError("fit must select a best checkpoint before export")
        result = self.evaluate(dataset)
        if self.output_dir:
            self._save_predictions(dataset, result["outputs"], split)
            self._save_contributions(dataset, result["outputs"], split)
            self._write_json(f"metrics/{split}_summary.json", {"epoch": self.best_epoch, "loss": result["loss"], "metrics": result["metrics"]})
            self._write_rows(f"metrics/{split}_per_disease.csv", result["metrics"]["per_disease"])
        return result

    def save_checkpoint(self, name: str) -> Path:
        path = self._checkpoint_path(name)
        payload = {
            "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.epoch, "best_epoch": self.best_epoch, "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs, "history": self.history, "config": asdict(self.config),
            "disease_ids": self.disease_ids, "proposition_ids": self.proposition_ids,
            "routing_mode": self.routing_mode, "input_hashes": self.input_hashes,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=False)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if tuple(checkpoint["disease_ids"]) != self.disease_ids:
            raise ValueError("checkpoint disease_ids do not match current run")
        if tuple(checkpoint["proposition_ids"]) != self.proposition_ids:
            raise ValueError("checkpoint proposition_ids do not match current run")
        if checkpoint["routing_mode"] != self.routing_mode:
            raise ValueError("checkpoint routing_mode does not match current run")
        if dict(checkpoint["input_hashes"]) != self.input_hashes:
            raise ValueError("checkpoint input hashes do not match current run")
        self.model.load_state_dict(checkpoint["model_state"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epoch, self.best_epoch, self.best_metric, self.bad_epochs = checkpoint["epoch"], checkpoint["best_epoch"], checkpoint["best_metric"], checkpoint["bad_epochs"]
        self.history = checkpoint["history"]

    def _train_epoch(self, dataset: Stage2Dataset) -> float:
        self.model.train(); support, targets = self._tensors(dataset); losses: list[float] = []
        for start in range(0, len(dataset.case_ids), self.config.batch_size):
            outputs = self.model(support[start:start + self.config.batch_size])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], targets[start:start + self.config.batch_size])
            self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step(); losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def _tensors(self, dataset: Stage2Dataset) -> tuple[Tensor, Tensor]:
        return torch.as_tensor(dataset.support_scores, dtype=torch.float32, device=self.device), torch.as_tensor(dataset.targets, dtype=torch.float32, device=self.device)

    def _validate_dataset(self, dataset: Stage2Dataset) -> None:
        if dataset.support_scores.ndim != 2 or dataset.targets.ndim != 2 or len(dataset.case_ids) != len(dataset.support_scores) or dataset.targets.shape[0] != len(dataset.case_ids):
            raise ValueError("dataset case IDs, support scores, and targets must align")
        if dataset.support_scores.shape[1] != len(self.proposition_ids) or dataset.targets.shape[1] != len(self.disease_ids):
            raise ValueError("dataset dimensions differ from model contracts")
        if not np.all(np.isfinite(dataset.support_scores[:, self.model.proposition_mask.detach().cpu().numpy()])):
            raise ValueError("included support scores must be finite")

    def _prepare_output(self) -> None:
        assert self.output_dir is not None
        for name in ("checkpoints", "metrics", "predictions", "diagnostics"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        self._write_json("run_config.json", asdict(self.config))

    def _checkpoint_path(self, name: str) -> Path:
        if self.output_dir is None: raise ValueError("output_dir is required for checkpointing")
        return self.output_dir / "checkpoints" / name

    def _save_predictions(self, dataset: Stage2Dataset, output: Mapping[str, np.ndarray], split: str) -> None:
        assert self.output_dir is not None
        np.savez_compressed(self.output_dir / "predictions" / f"{split}_best.npz", case_id=dataset.case_ids, disease_id=np.asarray(self.disease_ids), targets=dataset.targets, disease_evidence=output["disease_evidence"], logits=output["logits"], probabilities=output["probabilities"], gamma=output["gamma"], bias=output["bias"], routing_mode=np.asarray(self.routing_mode))

    def _save_contributions(self, dataset: Stage2Dataset, output: Mapping[str, np.ndarray], split: str) -> None:
        assert self.output_dir is not None
        index = self.model.proposition_disease_index.detach().cpu().numpy()
        np.savez_compressed(self.output_dir / "diagnostics" / f"contributions_{split}.npz", case_id=dataset.case_ids, proposition_id=np.asarray(self.proposition_ids), disease_id=np.asarray([self.disease_ids[i] for i in index]), support_scores=output["support_scores"], compatibility_scores=output["compatibility_scores"], logit_contributions=output["logit_contributions"], proposition_mask=self.model.proposition_mask.detach().cpu().numpy(), score_source=np.asarray(self.score_source))

    def _write_history(self) -> None:
        self._write_rows("metrics/history.csv", self.history)

    def _write_rows(self, relative: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not self.output_dir or not rows: return
        with (self.output_dir / relative).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)

    def _write_json(self, relative: str, value: Mapping[str, Any]) -> None:
        if self.output_dir: (self.output_dir / relative).write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
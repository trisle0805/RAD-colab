"""Train and evaluate the proposition extension on SkinCAP.

This entry point is intentionally separate from ``main_rad.py`` so the original
RAD implementation remains an unchanged baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer

from dataset.dataset_entity import Skin_Train_Dataset
from dataset.test_dataset import Skin_Test_Dataset
from models.clip_tqn import ModelRes, Text_Encoder_Bert
from proposition.data.knowledge_base import load_proposition_kb
from proposition.engine.checklist import collect_checklist, save_checklist_artifacts
from proposition.engine.artifacts import (
    append_metrics_history,
    capture_rng_state,
    restore_rng_state,
    save_checkpoint,
    save_final_metrics,
    save_validation_artifacts,
)
from proposition.engine.pipeline import EncodedBatchStream, prepare_knowledge_tokens
from proposition.engine.trainer import evaluate, train_one_epoch
from proposition.losses.pecl import PECLLoss
from proposition.models.proposition_model import PropositionModel
from scheduler import create_scheduler
from factory import utils


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _trainable(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _labels(csv_path: str) -> list[str]:
    labels = list(pd.read_csv(csv_path, nrows=0).columns[2:])
    if not labels:
        raise ValueError(f"no label columns found in {csv_path}")
    return labels

def _dataset_image_paths(dataset: Any) -> list[str]:
    """Return absolute test-image paths so downstream renderers are cwd-independent."""

    root = Path(dataset.image_root) if getattr(dataset, "image_root", None) else None
    paths: list[str] = []
    for raw_path in dataset.img_path_list:
        path = Path(str(raw_path))
        paths.append(str(path if path.is_absolute() or root is None else root / path))
    return paths


def _make_stream(
    loader, image_encoder, text_encoder, tokenizer, kb, tokens, device, args, *,
    training: bool = False, frozen_unique_pooled: torch.Tensor | None = None,
):
    return EncodedBatchStream(
        loader,
        image_encoder,
        text_encoder,
        tokenizer,
        kb,
        tokens,
        device,
        caption_max_length=args.max_length,
        query_level=args.query_level,
        use_label_branch=not args.no_label_branch,
        apply_fourier_augmentation=training,
        frozen_unique_pooled=frozen_unique_pooled,
    )


def _build_frozen_proposition_queries(
    bert_model_name: str,
    unique_tokens: Any,
    *,
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode static proposition texts once with the original ClinicalBERT CLS vectors."""

    frozen_encoder = AutoModel.from_pretrained(bert_model_name).to(device)
    frozen_encoder.eval()
    try:
        with torch.no_grad():
            encoded = frozen_encoder(**{
                name: value.to(device) for name, value in unique_tokens.items()
            })
            frozen = encoded.last_hidden_state[:, 0, :]
            frozen = frozen - frozen.mean(dim=0, keepdim=True)
        if frozen.shape[1] != embed_dim:
            raise ValueError(
                "frozen proposition query dimension "
                f"{frozen.shape[1]} does not match --embed_dim {embed_dim}"
            )
        return frozen.detach().cpu()
    finally:
        del frozen_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _validate_resume_freeze_config(
    checkpoint: dict[str, Any], freeze_proposition_queries: bool,
) -> None:
    """Reject resumes that would change the static proposition-query regime."""

    if "freeze_proposition_queries" not in checkpoint:
        saved = False
    else:
        saved = bool(checkpoint["freeze_proposition_queries"])
    if saved != freeze_proposition_queries:
        raise ValueError(
            "--freeze_proposition_queries differs from the checkpoint setting "
            f"(checkpoint={saved}, requested={freeze_proposition_queries})"
        )
    if saved and "frozen_proposition_queries" not in checkpoint:
        raise ValueError("checkpoint enables frozen proposition queries but has no saved tensor")

def _validate_query_configuration(args: argparse.Namespace) -> None:
    """Reject CLI combinations that do not define a segment-query regime."""

    if args.freeze_proposition_queries and args.query_level in ("disease", "proposition_segments"):
        raise NotImplementedError(
            "--freeze_proposition_queries is not supported with --query_level "
            f"{args.query_level}"
        )
    if args.query_level == "proposition_segments" and args.agg == "weighted":
        raise ValueError("--agg weighted is not supported with --query_level proposition_segments")


def _checkpoint_state(
    model, image_encoder, text_encoder, optimizer, scheduler, epoch, metrics,
    best_epoch, best_score, generator, config, *,
    freeze_proposition_queries: bool, frozen_proposition_queries: torch.Tensor | None,
):
    state = {
        "model": model.state_dict(),
        "image_encoder": image_encoder.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "rng_state": capture_rng_state(generator),
        "config": config,
        "freeze_proposition_queries": freeze_proposition_queries,
    }
    if frozen_proposition_queries is not None:
        state["frozen_proposition_queries"] = frozen_proposition_queries.detach().cpu()
    return state


def main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.dataset != "skin":
        raise ValueError("main_proposition.py currently supports only --dataset skin")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.clip_ratio != 0:
        raise NotImplementedError("ClipLoss is reserved for a later explicit ablation")
    _validate_query_configuration(args)

    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    checkpoints = output / "checkpoints"
    for directory in (checkpoints, output / "predictions", output / "metrics", output / "checklist"):
        directory.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv or config.get("ICD_train_file")
    val_csv = args.val_csv or config.get("ICD_val_file")
    test_csv = args.test_csv or config.get("ICD_test_file")
    image_root = args.image_root or config.get("image_root")
    if not all((train_csv, val_csv, test_csv, image_root, args.bert_model_name)):
        raise ValueError("train/val/test CSV, image_root, and bert_model_name are required")
    labels = _labels(train_csv)
    if _labels(val_csv) != labels or _labels(test_csv) != labels:
        raise ValueError("train, validation, and test label columns must match exactly")

    kb = load_proposition_kb(args.kb_path, labels)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=True)
    knowledge_tokens = prepare_knowledge_tokens(tokenizer, kb)
    if args.query_level == "proposition_segments":
        content_counts = knowledge_tokens.alignment_content_mask.sum(dim=1).cpu().numpy()
        _atomic_json(
            output / "segment_query_stats.json",
            {
                "num_alignment_texts": len(kb.unique_alignment_texts),
                "content_tokens_total": int(content_counts.sum()),
                "content_tokens_min": int(content_counts.min()),
                "content_tokens_median": float(np.median(content_counts)),
                "content_tokens_max": int(content_counts.max()),
            },
        )
    if args.query_level == "disease":
        _atomic_json(
            output / "disease_level_lengths.json",
            {
                "max_allowed_length": 512,
                "lengths": dict(zip(kb.disease_ids, knowledge_tokens.disease_lengths)),
                "truncated_diseases": [
                    disease for disease, length in zip(kb.disease_ids, knowledge_tokens.disease_lengths)
                    if length > 512
                ],
            },
        )

    workers = int(config.get("num_workers", 2))
    test_workers = int(config.get("test_num_workers", workers))
    train_dataset = Skin_Train_Dataset(train_csv, config["image_res"], image_root)
    val_dataset = Skin_Test_Dataset(val_csv, config["image_res"], image_root)
    test_dataset = Skin_Test_Dataset(test_csv, config["image_res"], image_root)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=RandomSampler(train_dataset),
        num_workers=workers, pin_memory=True, drop_last=True, worker_init_fn=utils.seed_worker,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["test_batch_size"], sampler=SequentialSampler(val_dataset),
        num_workers=test_workers, pin_memory=True, drop_last=False, worker_init_fn=utils.seed_worker,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["test_batch_size"], sampler=SequentialSampler(test_dataset),
        num_workers=test_workers, pin_memory=True, drop_last=False, worker_init_fn=utils.seed_worker,
    )

    image_encoder = ModelRes(args.image_encoder_name, args.embed_dim).to(device)
    text_encoder = Text_Encoder_Bert(args.bert_model_name, embed_dim=args.embed_dim).to(device)
    model = PropositionModel(
        args.embed_dim, kb.num_diseases, kb.prop_disease, kb.prop_polarity,
        tau_att=args.tau_att, aggregation=args.agg, query_level=args.query_level,
        prop_align_index=kb.align_unique_index,
        num_align_texts=len(kb.unique_alignment_texts),
        use_label_branch=not args.no_label_branch,
    ).to(device)
    pecl = None if args.no_pecl else PECLLoss(
        kb.disease_to_align_unique_indices,
        num_prototypes=len(kb.unique_alignment_texts),
        negative_ratio=args.negative_ratio,
        text_temperature=args.temperature_text,
        image_temperature=args.temperature_vision,
        text_weight=args.contrast_ratio_text,
        image_weight=args.contrast_ratio_vision,
        detach_text_prototypes=args.pecl_detach_text_prototypes,
    ).to(device)
    optimizer_config = config["optimizer"]
    decay, no_decay = [], []
    for module in (model, image_encoder, text_encoder):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(optimizer_config.get("weight_decay", 0.0))},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(optimizer_config["lr"]),
    )
    scheduler, _ = create_scheduler(utils.AttrDict(config["schedular"]), optimizer)
    generator = torch.Generator(device=device.type).manual_seed(args.seed + 1)

    start_epoch = 0
    best_epoch = None
    best_score = -math.inf
    latest = checkpoints / "latest.pt"
    resume_state = None
    if latest.is_file():
        resume_state = torch.load(latest, map_location=device, weights_only=False)
        _validate_resume_freeze_config(resume_state, args.freeze_proposition_queries)

    frozen_unique_pooled = None
    if args.freeze_proposition_queries:
        if resume_state is not None:
            frozen_unique_pooled = resume_state["frozen_proposition_queries"].detach().cpu()
        else:
            frozen_unique_pooled = _build_frozen_proposition_queries(
                args.bert_model_name,
                knowledge_tokens.unique,
                embed_dim=args.embed_dim,
                device=device,
            )
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        image_encoder.load_state_dict(resume_state["image_encoder"])
        text_encoder.load_state_dict(resume_state["text_encoder"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["lr_scheduler"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_epoch = resume_state.get("best_epoch")
        best_score = float(resume_state.get("best_validation_score", -math.inf))
        restore_rng_state(resume_state["rng_state"], generator)

    max_epoch = int(config["schedular"]["epochs"])
    epoch_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, max_epoch):
        started = time.perf_counter()
        image_encoder.train()
        text_encoder.train()
        train_result = train_one_epoch(
            model, pecl,
            _make_stream(
                train_loader, image_encoder, text_encoder, tokenizer, kb, knowledge_tokens,
                device, args, training=True, frozen_unique_pooled=frozen_unique_pooled,
            ),
            optimizer,
            accumulation_steps=int(config.get("grad_accumulation_steps", 1)),
            lambda_label=0.0 if args.no_label_branch else args.lambda_label,
            beta_pecl=0.0 if args.no_pecl else args.beta_pecl,
            pecl_generator=generator,
            query_chunk_size=args.query_chunk_size,
            scheduler=scheduler,
            epoch=epoch,
            warmup_iterations=int(config["schedular"].get("warmup_epochs", 0)),
        )
        image_encoder.eval()
        text_encoder.eval()
        validation = evaluate(
            model, pecl,
            _make_stream(
                val_loader, image_encoder, text_encoder, tokenizer, kb, knowledge_tokens,
                device, args, frozen_unique_pooled=frozen_unique_pooled,
            ),
            lambda_label=0.0 if args.no_label_branch else args.lambda_label,
            beta_pecl=0.0 if args.no_pecl else args.beta_pecl,
            pecl_generator=generator,
            query_chunk_size=args.query_chunk_size,
        )
        epoch_number = epoch + 1
        summary = save_validation_artifacts(validation, output, epoch_number, topk=args.topk)
        score = float(summary["validation_score"])
        if score > best_score:
            best_score, best_epoch = score, epoch_number
            is_best = True
        else:
            is_best = False
        record = {
            "epoch": epoch_number,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **{f"train_loss_{key}": value for key, value in train_result.losses.items()},
            **summary,
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_metrics_history(output, record)
        checkpoint = _checkpoint_state(
            model, image_encoder, text_encoder, optimizer, scheduler, epoch, record,
            best_epoch, best_score, generator, config,
            freeze_proposition_queries=args.freeze_proposition_queries,
            frozen_proposition_queries=frozen_unique_pooled,
        )
        save_checkpoint(latest, checkpoint)
        if is_best:
            save_checkpoint(checkpoints / "best.pt", checkpoint)
        epoch_times.append(time.perf_counter() - started)
        if epoch + 1 < max_epoch:
            scheduler.step(epoch + 1)

    best_path = checkpoints / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"best checkpoint not found: {best_path}")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    image_encoder.load_state_dict(best["image_encoder"])
    text_encoder.load_state_dict(best["text_encoder"])
    image_encoder.eval()
    text_encoder.eval()
    inference_started = time.perf_counter()
    test_result = evaluate(
        model, pecl,
        _make_stream(
            test_loader, image_encoder, text_encoder, tokenizer, kb, knowledge_tokens,
            device, args, frozen_unique_pooled=frozen_unique_pooled,
        ),
        lambda_label=0.0 if args.no_label_branch else args.lambda_label,
        beta_pecl=0.0 if args.no_pecl else args.beta_pecl,
        pecl_generator=generator,
        query_chunk_size=args.query_chunk_size,
    )
    inference_seconds = time.perf_counter() - inference_started
    test_image_paths = _dataset_image_paths(test_dataset)
    save_final_metrics(
        test_result, output, best_epoch=int(best["best_epoch"]),
        best_validation_score=float(best["best_validation_score"]), topk=args.topk,
        image_paths=test_image_paths,
    )
    checklist_generator = torch.Generator().manual_seed(args.seed)
    selected_attention_samples = torch.randperm(
        len(test_dataset), generator=checklist_generator
    )[: min(args.full_attention_samples, len(test_dataset))].sort().values
    checklist_result = collect_checklist(
        model,
        pecl,
        _make_stream(
            test_loader, image_encoder, text_encoder, tokenizer, kb, knowledge_tokens,
            device, args, frozen_unique_pooled=frozen_unique_pooled,
        ),
        lambda_label=0.0 if args.no_label_branch else args.lambda_label,
        beta_pecl=0.0 if args.no_pecl else args.beta_pecl,
        pecl_generator=generator,
        query_chunk_size=args.query_chunk_size,
        top_evidence=args.attention_topk,
        full_attention_sample_indices=selected_attention_samples,
    )
    checklist_paths = save_checklist_artifacts(
        checklist_result,
        output,
        kb,
        image_paths=test_image_paths,
        top_evidence=args.attention_topk,
        aggregation_weights=model.pathway.aggregation_weights(),
        query_level=args.query_level,
    )

    run_config = {
        "arguments": vars(args),
        "yaml_config": config,
        "run_id": args.run_id,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "sha256": {"kb": _sha256(args.kb_path), "train_csv": _sha256(train_csv),
                   "val_csv": _sha256(val_csv), "test_csv": _sha256(test_csv)},
        "sample_counts": {"train": len(train_dataset), "validation": len(val_dataset), "test": len(test_dataset)},
        "num_diseases": kb.num_diseases,
        "num_propositions": kb.num_propositions,
        "query_level": args.query_level,
        "segment_query_tokens_total": (
            int(knowledge_tokens.alignment_content_mask.sum())
            if args.query_level == "proposition_segments" else None
        ),
        "knowledge_max_length": knowledge_tokens.max_length,
        "measured_knowledge_max_length": knowledge_tokens.measured_max_length,
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "trainable_parameters": {
            "image_encoder": _trainable(image_encoder), "text_encoder": _trainable(text_encoder),
            "proposition_model": _trainable(model),
        },
        "runtime": {
            "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else None,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "mean_test_seconds_per_sample": inference_seconds / len(test_dataset),
        },
        "best_epoch": int(best["best_epoch"]),
        "best_validation_score": float(best["best_validation_score"]),
        "freeze_proposition_queries": args.freeze_proposition_queries,
        "pecl_detach_text_prototypes": args.pecl_detach_text_prototypes,
        "disease_level_prediction_uses_proposition_level_pecl": args.query_level == "disease" and not args.no_pecl,
        "checklist_artifacts": checklist_paths,
    }
    _atomic_json(output / "run_config.json", run_config)
    if args.query_level in ("proposition", "proposition_segments"):
        kb.write_proposition_index(output / "checklist" / "proposition_index.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="skin")
    parser.add_argument("--config", default="configs/skin.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_id", default="P1_s42")
    parser.add_argument("--train_csv", default="")
    parser.add_argument("--val_csv", default="")
    parser.add_argument("--test_csv", default="")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--kb_path", default="kb_builder/outputs/final/kb_propositions.json")
    parser.add_argument("--image_encoder_name", default="resnet50")
    parser.add_argument("--bert_model_name", required=True)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--tau_att", type=float, default=0.1)
    parser.add_argument("--agg", choices=("mean", "weighted"), default="mean")
    parser.add_argument(
        "--query_level",
        choices=("proposition", "disease", "proposition_segments"),
        default="proposition",
    )
    parser.add_argument("--query_chunk_size", type=int)
    parser.add_argument("--attention_topk", type=int, default=10)
    parser.add_argument("--full_attention_samples", type=int, default=20)
    parser.add_argument("--lambda_label", type=float, default=1.0)
    parser.add_argument("--beta_pecl", type=float, default=1.0)
    parser.add_argument("--no_label_branch", action="store_true")
    parser.add_argument("--no_pecl", action="store_true")
    parser.add_argument("--freeze_proposition_queries", action="store_true")
    parser.add_argument("--pecl_detach_text_prototypes", action="store_true")
    parser.add_argument("--negative_ratio", type=int, default=5)
    parser.add_argument("--temperature_text", type=float, default=0.5)
    parser.add_argument("--temperature_vision", type=float, default=2.0)
    parser.add_argument("--contrast_ratio_text", type=float, default=0.1)
    parser.add_argument("--contrast_ratio_vision", type=float, default=0.001)
    parser.add_argument("--clip_ratio", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.topk = sorted(set(parsed.topk))
    if not parsed.topk or min(parsed.topk) < 1:
        raise ValueError("--topk values must be positive")
    if parsed.attention_topk <= 0:
        raise ValueError("--attention_topk must be positive")
    with Path(parsed.config).open("r", encoding="utf-8") as handle:
        yaml_config = yaml.safe_load(handle)
    main(parsed, yaml_config)

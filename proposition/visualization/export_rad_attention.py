"""Export original RAD test-set cross-attention from a trained checkpoint.

This read-only utility loads a completed ``main_rad.py`` run and replays the
SkinCAP test forward pass with ``return_atten=True``. It does not retrain or
modify the original RAD architecture. Replayed probabilities are checked against
``predictions/test_best.npz`` before evidence is written.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer

from dataset.test_dataset import Skin_Test_Dataset
from engine.train_rad import get_text_features_bert
from models.clip_tqn import ModelRes, TQN_Model_fusion, Text_Encoder_Bert

LABEL_SEQUENCE_LENGTH = 16
GUIDELINE_SEQUENCE_LENGTH = 16
TEXT_SEQUENCE_LENGTH = 512
BRANCHES = ("guideline", "label")


def _labels(csv_path: str | Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns[2:])


def _guidelines(path: str | Path, labels: list[str]) -> list[str]:
    """Read and validate the guideline text in the same order as RAD labels."""

    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            if index >= len(labels) or record["query"] != labels[index]:
                expected = labels[index] if index < len(labels) else "<no label>"
                raise RuntimeError(
                    f"Guideline {index} ({record['query']!r}) does not match label {expected!r}."
                )
            texts.append(record["guideline_1_content"])
    if len(texts) != len(labels):
        raise RuntimeError("Guideline count does not match the SkinCAP label count.")
    return texts


def _entropy(attention: torch.Tensor) -> torch.Tensor:
    """Normalized Shannon entropy over every unmasked RAD memory position."""

    safe = attention.clamp_min(torch.finfo(attention.dtype).tiny)
    return -(safe * safe.log()).sum(dim=-1) / float(np.log(attention.shape[-1]))


def _image_paths(dataset: Any) -> list[str]:
    root = Path(dataset.image_root) if getattr(dataset, "image_root", None) else None
    return [
        str(Path(str(path)) if Path(str(path)).is_absolute() or root is None else root / str(path))
        for path in dataset.img_path_list
    ]


def _validate_saved_predictions(
    run_dir: Path, arrays: dict[str, np.ndarray], tolerance: float
) -> str:
    """Stop if replay does not exactly reproduce the stored original-RAD scores."""

    saved_path = run_dir / "predictions" / "test_best.npz"
    if not saved_path.is_file():
        return "predictions/test_best.npz not found; replay not verified"

    with np.load(saved_path) as saved:
        for branch, saved_key in (("label", "pred_label"), ("guideline", "pred_guideline")):
            if saved_key not in saved:
                raise RuntimeError(f"{saved_path} does not contain {saved_key}.")
            saved_values = saved[saved_key]
            replayed = arrays[f"{branch}_pred"]
            if saved_values.shape != replayed.shape:
                raise RuntimeError(
                    f"Replayed {branch} prediction shape {replayed.shape} does not match "
                    f"{saved_path.name} shape {saved_values.shape}."
                )
            difference = float(np.abs(saved_values - replayed).max())
            if difference > tolerance:
                raise RuntimeError(
                    f"Replayed {branch} probabilities differ from {saved_path} by {difference:.2e}; "
                    "check checkpoint, configuration, test CSV, tokenizer, and test batch size."
                )
    return f"matched {saved_path.name} within {tolerance:g}"


@torch.no_grad()
def export(args: argparse.Namespace) -> Path:
    run_dir = Path(args.rad_run_dir)
    config = yaml.safe_load(Path(args.rad_config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    batch_size = int(args.batch_size or config["test_batch_size"])

    labels = _labels(args.test_csv)
    guideline_texts = _guidelines(args.guideline_path, labels)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=True)

    image_encoder = ModelRes(args.image_encoder_name, args.embed_dim).to(device)
    text_encoder = Text_Encoder_Bert(args.bert_model_name, embed_dim=args.embed_dim).to(device)
    model = TQN_Model_fusion(embed_dim=args.embed_dim).to(device)
    model_guideline = TQN_Model_fusion(embed_dim=args.embed_dim).to(device)

    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for module, key in (
        (model, "model"),
        (model_guideline, "model_guideline"),
        (image_encoder, "image_encoder"),
        (text_encoder, "text_encoder"),
    ):
        if key not in state:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain {key!r}.")
        module.load_state_dict(state[key])
        module.eval()

    dataset = Skin_Test_Dataset(args.test_csv, config["image_res"], args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=int(config.get("test_num_workers", 0)),
        pin_memory=True,
        drop_last=False,
    )

    # Static disease and guideline queries: exactly the construction in valid_on_ICD.
    _, label_hidden = get_text_features_bert(
        text_encoder, labels, tokenizer, device, max_length=args.max_length
    )
    _, guideline_hidden = get_text_features_bert(
        text_encoder, guideline_texts, tokenizer, device, max_length=args.max_length
    )
    queries = {
        "label": (model, label_hidden[:, :LABEL_SEQUENCE_LENGTH, :]),
        "guideline": (model_guideline, guideline_hidden[:, :GUIDELINE_SEQUENCE_LENGTH, :]),
    }

    selected_full_indices = np.empty(0, dtype=np.int64)
    if args.proposition_evidence:
        with np.load(args.proposition_evidence, allow_pickle=False) as evidence:
            selected_full_indices = evidence["full_attention_sample_indices"].astype(np.int64)
    selected_full_set = set(selected_full_indices.tolist())

    collected: dict[str, list[np.ndarray]] = {
        "gt": [],
        "caption_input_ids": [],
        "caption_attention_mask": [],
        "query_token_position": [],
    }
    for branch in BRANCHES:
        for name in (
            "pred",
            "top_indices",
            "top_weights",
            "entropy",
            "image_mass",
            "caption_mass",
            "pad_mass",
        ):
            collected[f"{branch}_{name}"] = []
    full_attention: dict[str, list[np.ndarray]] = {branch: [] for branch in BRANCHES}

    image_token_count: int | None = None
    offset = 0
    for sample in loader:
        image = sample["image"].to(device, non_blocking=True)
        labels_batch = sample["label"].float()
        image_features, _ = image_encoder(image)
        tokens = tokenizer(
            list(sample["entity"]),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        _, text_hidden = text_encoder.encode_text(tokens)
        fusion_features = torch.cat(
            (image_features, text_hidden[:, :TEXT_SEQUENCE_LENGTH, :]), dim=1
        )
        n_image = int(image_features.shape[1])
        if image_token_count is None:
            image_token_count = n_image
        elif image_token_count != n_image:
            raise RuntimeError("Image encoder produced a varying number of patch tokens.")

        caption_mask = tokens["attention_mask"][:, :TEXT_SEQUENCE_LENGTH].bool()
        current = int(image.shape[0])
        for branch, (decoder, query) in queries.items():
            logits, attention = decoder(query, fusion_features, return_atten=True)
            attention = attention.float()
            probabilities = torch.sigmoid(logits)[:, :, 0]
            top_weights, top_indices = torch.topk(
                attention, k=min(args.attention_topk, attention.shape[-1]), dim=-1
            )
            image_mass = attention[..., :n_image].sum(dim=-1)
            caption_mass = (attention[..., n_image:] * caption_mask[:, None, :]).sum(dim=-1)

            collected[f"{branch}_pred"].append(probabilities.cpu().numpy())
            collected[f"{branch}_top_indices"].append(top_indices.to(torch.int32).cpu().numpy())
            collected[f"{branch}_top_weights"].append(top_weights.cpu().numpy())
            collected[f"{branch}_entropy"].append(_entropy(attention).cpu().numpy())
            collected[f"{branch}_image_mass"].append(image_mass.cpu().numpy())
            collected[f"{branch}_caption_mass"].append(caption_mass.cpu().numpy())
            collected[f"{branch}_pad_mass"].append((1.0 - image_mass - caption_mass).cpu().numpy())
            for local_index in range(current):
                if offset + local_index in selected_full_set:
                    full_attention[branch].append(
                        attention[local_index].to(torch.float16).cpu().numpy()
                    )

        collected["gt"].append(labels_batch.numpy())
        collected["caption_input_ids"].append(tokens["input_ids"].cpu().numpy())
        collected["caption_attention_mask"].append(
            tokens["attention_mask"].to(torch.int8).cpu().numpy()
        )
        # RAD repeats its 16 query embeddings to the local batch size on every batch.
        collected["query_token_position"].append(
            (np.arange(current) % LABEL_SEQUENCE_LENGTH).astype(np.int16)
        )
        offset += current

    if image_token_count is None:
        raise RuntimeError("The test loader produced no samples.")
    arrays = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    replay_note = _validate_saved_predictions(run_dir, arrays, args.replay_tolerance)

    metrics_path = run_dir / "final_metrics.json"
    if metrics_path.is_file():
        fusion_weights = np.asarray(
            json.loads(metrics_path.read_text(encoding="utf-8"))["fusion_weights"], dtype=np.float32
        )
        arrays["fused_pred"] = (
            arrays["label_pred"] * fusion_weights[None, :]
            + arrays["guideline_pred"] * (1.0 - fusion_weights[None, :])
        )
        arrays["fusion_weights"] = fusion_weights

    saved_full_indices = np.asarray(
        [index for index in sorted(selected_full_set) if 0 <= index < offset], dtype=np.int64
    )
    if len(saved_full_indices) != len(selected_full_set):
        raise IndexError("A requested proposition evidence sample index is outside the RAD test set.")
    arrays["full_attention_sample_indices"] = saved_full_indices
    for branch in BRANCHES:
        arrays[f"{branch}_full_attention"] = (
            np.stack(full_attention[branch])
            if full_attention[branch]
            else np.zeros((0, 0, 0), dtype=np.float16)
        )
    arrays["image_token_count"] = np.full(offset, image_token_count, dtype=np.int32)
    arrays["image_paths"] = np.asarray(_image_paths(dataset), dtype=str)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "checklist"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rad_attention_evidence.npz"
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, output_path)

    schema = {
        "artifact": output_path.name,
        "model": "original RAD (main_rad.py), unchanged",
        "attention_definition": "last decoder-layer cross-attention, averaged over 4 heads",
        "branches": list(BRANCHES),
        "query_count": len(labels),
        "disease_ids": labels,
        "memory_layout": {
            "image": [0, image_token_count],
            "caption": [image_token_count, image_token_count + TEXT_SEQUENCE_LENGTH],
        },
        "caption_pad_masked": False,
        "query_token_position": "local test-batch position modulo 16",
        "test_batch_size": batch_size,
        "replay_check": replay_note,
        "fields": {
            "{branch}_pred": "sigmoid probability per disease (N, C)",
            "{branch}_top_indices / _top_weights": "top memory positions per disease query (N, C, K)",
            "{branch}_entropy": "normalized attention entropy over all memory positions (N, C)",
            "{branch}_image_mass / _caption_mass / _pad_mass": "attention mass per disease query (N, C)",
            "{branch}_full_attention": "selected full attention (K, C, L), float16",
            "fused_pred": "class-wise fusion using final_metrics.json weights (N, C)",
        },
    }
    output_path.with_suffix(".schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RAD attention exported to {output_path} ({replay_note})")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rad_run_dir", required=True, type=Path)
    parser.add_argument("--rad_config", required=True, type=Path)
    parser.add_argument("--test_csv", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--bert_model_name", required=True)
    parser.add_argument("--guideline_path", default="guideline/qwen_maxtoken2k_skincap47_4sources.jsonl")
    parser.add_argument("--proposition_evidence", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--image_encoder_name", default="resnet50")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--attention_topk", type=int, default=10)
    parser.add_argument("--replay_tolerance", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    export(build_parser().parse_args())

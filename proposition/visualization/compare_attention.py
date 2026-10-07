"""Compare saved original-RAD and proposition attention on the same SkinCAP cases.

The script reads saved evidence only; it never loads a checkpoint. RAD produces a
single query map per disease. The proposition panel is the uniform average of all
proposition-query maps assigned to the selected disease.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_attention as renderer  # noqa: E402

STREAMS = ("rad_guideline", "rad_label", "proposition")


def _edge_positions(image_token_count: int) -> tuple[set[int], set[int]]:
    side = math.isqrt(image_token_count)
    if side * side != image_token_count:
        raise ValueError("Image token count must form a square patch grid.")
    bottom_row = set(range(image_token_count - side, image_token_count))
    border = {
        row * side + column
        for row in range(side)
        for column in range(side)
        if row in (0, side - 1) or column in (0, side - 1)
    }
    return bottom_row, border


def _top_shares(
    top_indices: np.ndarray, caption_mask: np.ndarray, image_token_count: int
) -> dict[str, float]:
    """Classify each top-K memory index as image, valid caption, or caption PAD."""

    if top_indices.ndim < 2 or caption_mask.ndim != 2:
        raise ValueError("Top attention indices and caption masks must include a sample axis.")
    bottom_row, border = _edge_positions(image_token_count)
    flat = top_indices.reshape(top_indices.shape[0], -1)
    if flat.size == 0:
        raise ValueError("No top-attention positions were supplied.")

    is_image = flat < image_token_count
    caption_positions = np.clip(flat - image_token_count, 0, caption_mask.shape[1] - 1)
    valid_caption = (~is_image) & (
        np.take_along_axis(caption_mask, caption_positions, axis=1) > 0
    )
    total = float(flat.size)
    return {
        "top_k_on_image": float(is_image.sum() / total),
        "top_k_on_caption": float(valid_caption.sum() / total),
        "top_k_on_pad": float(((~is_image) & (~valid_caption)).sum() / total),
        "top_k_on_bottom_image_row": float(np.isin(flat, list(bottom_row)).sum() / total),
        "top_k_on_image_border": float(np.isin(flat, list(border)).sum() / total),
    }


def _masses(
    attention: np.ndarray, caption_mask: np.ndarray, image_token_count: int
) -> dict[str, float]:
    """Compute mean attention masses for full maps shaped ``(K, L)``."""

    image_mass = attention[:, :image_token_count].sum(axis=-1)
    caption_mass = (attention[:, image_token_count:] * caption_mask).sum(axis=-1)
    return {
        "mass_image": float(image_mass.mean()),
        "mass_caption": float(caption_mass.mean()),
        "mass_pad": float(np.clip(1.0 - image_mass - caption_mass, 0.0, None).mean()),
    }


def _validate_artifacts(
    proposition: dict[str, np.ndarray], rad: dict[str, np.ndarray], disease_count: int
) -> None:
    required_prop = {
        "image_paths",
        "gt",
        "image_token_count",
        "caption_input_ids",
        "caption_attention_mask",
        "pred_proposition",
        "attention_entropy",
        "attention_top_indices",
        "full_attention_sample_indices",
        "full_attention",
    }
    required_rad = {
        "image_paths",
        "gt",
        "image_token_count",
        "query_token_position",
        "full_attention_sample_indices",
        "guideline_pred",
        "label_pred",
        "guideline_full_attention",
        "label_full_attention",
    }
    missing_prop = sorted(required_prop - proposition.keys())
    missing_rad = sorted(required_rad - rad.keys())
    if missing_prop:
        raise KeyError(f"Proposition evidence is missing: {', '.join(missing_prop)}")
    if missing_rad:
        raise KeyError(f"RAD evidence is missing: {', '.join(missing_rad)}")
    if not np.array_equal(proposition["image_paths"], rad["image_paths"]):
        raise ValueError("The evidence artifacts do not list identical test images in identical order.")
    if not np.array_equal(proposition["gt"], rad["gt"]):
        raise ValueError("The evidence artifacts do not have identical ground-truth labels.")
    if proposition["gt"].shape[1] != disease_count:
        raise ValueError("Disease index does not match the prediction/ground-truth class count.")
    if not np.array_equal(proposition["image_token_count"], rad["image_token_count"]):
        raise ValueError("RAD and proposition image-token counts do not match.")


def compare(args: argparse.Namespace) -> Path:
    proposition_path = Path(args.proposition_evidence)
    checklist_dir = proposition_path.parent
    proposition_index = renderer._read_json(checklist_dir / "proposition_index.json")
    diseases = [
        entry["disease_id"]
        for entry in renderer._read_json(checklist_dir / "disease_index.json")
    ]
    proposition_diseases = np.asarray(
        [entry["disease_index"] for entry in proposition_index], dtype=np.int64
    )

    with np.load(proposition_path, allow_pickle=False) as values:
        proposition = dict(values)
    with np.load(args.rad_evidence, allow_pickle=False) as values:
        rad = dict(values)
    _validate_artifacts(proposition, rad, len(diseases))

    image_token_count = int(proposition["image_token_count"][0])
    caption_mask = proposition["caption_attention_mask"].astype(np.float32)
    truth = proposition["gt"].argmax(axis=1)
    case_count = len(truth)
    case_positions = np.arange(case_count)
    proposition_rows = [np.flatnonzero(proposition_diseases == disease) for disease in truth]
    if any(rows.size == 0 for rows in proposition_rows):
        raise ValueError("At least one ground-truth disease has no proposition record.")

    summary: dict[str, Any] = {
        "test_cases": int(case_count),
        "image_tokens": image_token_count,
        "computed_on": "query of each test case's ground-truth disease",
    }
    for branch in ("guideline", "label"):
        top_indices = rad[f"{branch}_top_indices"][case_positions, truth]
        summary[f"rad_{branch}"] = {
            "mean_entropy": float(rad[f"{branch}_entropy"][case_positions, truth].mean()),
            "mass_image": float(rad[f"{branch}_image_mass"][case_positions, truth].mean()),
            "mass_caption": float(rad[f"{branch}_caption_mass"][case_positions, truth].mean()),
            "mass_pad": float(rad[f"{branch}_pad_mass"][case_positions, truth].mean()),
            **_top_shares(top_indices, caption_mask, image_token_count),
        }

    proposition_entropy = [
        proposition["attention_entropy"][case_index, rows].mean()
        for case_index, rows in enumerate(proposition_rows)
    ]
    proposition_top_shares = [
        _top_shares(
            proposition["attention_top_indices"][case_index, rows].reshape(1, -1),
            caption_mask[case_index : case_index + 1],
            image_token_count,
        )
        for case_index, rows in enumerate(proposition_rows)
    ]
    summary["proposition"] = {
        "mean_entropy": float(np.mean(proposition_entropy)),
        **{
            key: float(np.mean([shares[key] for shares in proposition_top_shares]))
            for key in proposition_top_shares[0]
        },
    }

    proposition_positions = {
        int(sample): index
        for index, sample in enumerate(proposition["full_attention_sample_indices"])
    }
    rad_positions = {
        int(sample): index for index, sample in enumerate(rad["full_attention_sample_indices"])
    }
    common_samples = sorted(proposition_positions.keys() & rad_positions.keys())
    if common_samples:
        proposition_full_maps = np.stack(
            [
                proposition["full_attention"][proposition_positions[sample]][
                    proposition_diseases == truth[sample]
                ]
                .astype(np.float32)
                .mean(axis=0)
                for sample in common_samples
            ]
        )
        summary["proposition"].update(
            _masses(
                proposition_full_maps,
                caption_mask[common_samples],
                image_token_count,
            )
        )
        summary["full_attention_cases"] = common_samples

    output_dir = Path(args.output_dir) if args.output_dir else checklist_dir / "attention_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attention_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=True)
    selected_samples = args.samples or common_samples
    for sample_index in selected_samples:
        if sample_index not in proposition_positions or sample_index not in rad_positions:
            raise ValueError(
                f"Sample {sample_index} has no full attention in both artifacts; "
                f"choose one of {common_samples}."
            )
        disease_index = int(truth[sample_index])
        maps = {
            "rad_guideline": rad["guideline_full_attention"][
                rad_positions[sample_index], disease_index
            ].astype(np.float32),
            "rad_label": rad["label_full_attention"][
                rad_positions[sample_index], disease_index
            ].astype(np.float32),
            "proposition": proposition["full_attention"][
                proposition_positions[sample_index]
            ][proposition_diseases == disease_index]
            .astype(np.float32)
            .mean(axis=0),
        }
        probabilities = {
            "rad_guideline": float(rad["guideline_pred"][sample_index, disease_index]),
            "rad_label": float(rad["label_pred"][sample_index, disease_index]),
            "proposition": float(proposition["pred_proposition"][sample_index, disease_index]),
        }
        top_predictions = {
            "rad_guideline": diseases[int(rad["guideline_pred"][sample_index].argmax())],
            "rad_label": diseases[int(rad["label_pred"][sample_index].argmax())],
            "proposition": diseases[int(proposition["pred_proposition"][sample_index].argmax())],
        }

        sample_dir = output_dir / f"sample_{sample_index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        cards: list[str] = []
        for stream in STREAMS:
            attention = maps[stream]
            image_path = sample_dir / f"{stream}.png"
            renderer._heatmap_overlay(
                str(proposition["image_paths"][sample_index]),
                attention[:image_token_count],
                image_path,
                args.alpha,
            )
            tokens = renderer._caption_tokens(
                tokenizer,
                proposition["caption_input_ids"][sample_index],
                proposition["caption_attention_mask"][sample_index],
                attention[image_token_count:],
            )
            mass = _masses(
                attention[None],
                caption_mask[sample_index : sample_index + 1],
                image_token_count,
            )
            if stream.startswith("rad"):
                extra = (
                    f"<br><b>RAD query token position:</b> "
                    f"{int(rad['query_token_position'][sample_index])}"
                )
            else:
                extra = (
                    f"<br><b>Aggregation:</b> uniform mean of "
                    f"{int((proposition_diseases == disease_index).sum())} proposition maps"
                )
            cards.append(
                f"""
                <article class=\"card\">
                  <h2>{html.escape(stream)}</h2>
                  <p><b>P({html.escape(diseases[disease_index])})</b> = {probabilities[stream]:.4f}
                  <br><b>Top-1:</b> {html.escape(top_predictions[stream])}
                  <br><b>Attention mass:</b> image {mass['mass_image']:.2f} · caption {mass['mass_caption']:.2f} · PAD {mass['mass_pad']:.2f}{extra}</p>
                  <img src=\"{image_path.name}\" alt=\"{html.escape(stream)} attention\">
                  <p class=\"caption\">{renderer._caption_html(tokens)}</p>
                </article>"""
            )
        report = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Attention comparison {sample_index}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
.card {{ display:inline-block; vertical-align:top; width:32%; min-width:320px; margin:0 1% 16px 0; padding:12px; box-sizing:border-box; border:1px solid #cbd5e1; border-radius:8px; background:#f8fafc; }}
.card img {{ width:100%; border-radius:5px; }} .caption {{ line-height:2.1; background:white; padding:8px; }}
</style></head><body>
<h1>Sample {sample_index} — true disease: {html.escape(diseases[disease_index])}</h1>
<p>Heatmap and caption colours are min–max normalized within each panel. Compare the printed attention masses for absolute allocation.</p>
{''.join(cards)}
</body></html>"""
        (sample_dir / "report.html").write_text(report, encoding="utf-8")

    print(f"Attention comparison written to {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposition_evidence", required=True, type=Path)
    parser.add_argument("--rad_evidence", required=True, type=Path)
    parser.add_argument("--bert_model_name", required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--samples", nargs="+", type=int)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser


if __name__ == "__main__":
    compare(build_parser().parse_args())

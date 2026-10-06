"""Render UI-ready proposition attention reports from a saved test evidence artifact.

Run this after ``main_proposition.py`` has written checklist artifacts. Rendering
uses saved tensors only; it never loads a model checkpoint or retrains a model.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
from transformers import AutoTokenizer


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _heatmap_overlay(image_path: str, patch_attention: np.ndarray, output_path: Path, alpha: float) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    patch_count = patch_attention.size
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError(f"image patch count {patch_count} is not a square grid")
    heat = Image.fromarray(np.uint8(_normalize(patch_attention).reshape(side, side) * 255))
    heat = heat.resize((width, height), Image.Resampling.BICUBIC)
    color = Image.fromarray(
        np.uint8(matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3] * 255)
    )
    Image.blend(image, color, alpha).save(output_path)


def _caption_tokens(tokenizer: Any, ids: np.ndarray, mask: np.ndarray, attention: np.ndarray) -> list[tuple[str, float]]:
    valid = int(np.asarray(mask, dtype=bool).sum())
    raw_tokens = tokenizer.convert_ids_to_tokens(ids[:valid].tolist(), skip_special_tokens=False)
    special = set(tokenizer.all_special_tokens)
    result: list[tuple[str, float]] = []
    for token, weight in zip(raw_tokens, attention[:valid], strict=True):
        if token not in special:
            result.append((token, float(weight)))
    return result


def _caption_html(tokens: list[tuple[str, float]]) -> str:
    if not tokens:
        return "<em>No non-special caption tokens.</em>"
    values = np.asarray([weight for _, weight in tokens], dtype=np.float32)
    normalized = _normalize(values)
    spans = []
    for (token, weight), intensity in zip(tokens, normalized, strict=True):
        color = f"rgba(220, 38, 38, {0.10 + 0.80 * float(intensity):.3f})"
        spans.append(
            f'<span title="attention={weight:.6f}" style="background:{color}; padding:2px 3px; margin:1px; border-radius:3px">{html.escape(token)}</span>'
        )
    return " ".join(spans)


def _selected_propositions(
    disease_index: int,
    compatibility: np.ndarray,
    kb: list[dict[str, Any]],
    maximum: int,
) -> list[int]:
    candidates = [index for index, record in enumerate(kb) if record["disease_index"] == disease_index]
    return sorted(candidates, key=lambda index: float(compatibility[index]), reverse=True)[:maximum]


def render(
    evidence_path: Path,
    tokenizer_name: str,
    output_dir: Path,
    *,
    sample_indices: list[int] | None,
    diseases_per_sample: int,
    propositions_per_disease: int,
    alpha: float,
) -> None:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    checklist_dir = evidence_path.parent
    schema = _read_json(evidence_path.with_suffix(".schema.json"))
    if schema["query_level"] != "proposition":
        raise ValueError("attention renderer currently requires proposition-level query artifacts")
    kb = _read_json(checklist_dir / "proposition_index.json")
    diseases = _read_json(checklist_dir / "disease_index.json")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

    with np.load(evidence_path, allow_pickle=False) as values:
        image_paths = values["image_paths"].astype(str)
        predictions = values["pred_proposition"]
        ground_truth = values["gt"]
        support = values["support"]
        compatibility = values["compatibility"]
        full_attention = values["full_attention"]
        full_attention_indices = values["full_attention_sample_indices"]
        attention_by_sample = {
            int(sample): full_attention[position]
            for position, sample in enumerate(full_attention_indices)
        }
        entropy = values["attention_entropy"]
        input_ids = values["caption_input_ids"]
        masks = values["caption_attention_mask"]
        image_token_count = int(values["image_token_count"][0])

        if sample_indices is None:
            sample_indices = sorted(attention_by_sample)
        for sample_index in sample_indices:
            if not 0 <= sample_index < len(image_paths):
                raise IndexError(f"sample index out of range: {sample_index}")
            if sample_index not in attention_by_sample:
                raise ValueError(
                    f"sample {sample_index} has no stored full attention; choose one of "
                    f"{sorted(attention_by_sample)}"
                )
            attention = attention_by_sample[sample_index]
            sample_dir = output_dir / f"sample_{sample_index:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            disease_order = np.argsort(-predictions[sample_index])[:diseases_per_sample]
            truth_index = int(np.argmax(ground_truth[sample_index]))
            cards: list[str] = []

            for rank, disease_position in enumerate(disease_order, start=1):
                disease = diseases[int(disease_position)]
                rows: list[str] = []
                for proposition_index in _selected_propositions(
                    int(disease_position), compatibility[sample_index], kb, propositions_per_disease
                ):
                    record = kb[proposition_index]
                    overlay = sample_dir / f"d{disease_position:02d}_p{proposition_index:04d}.png"
                    _heatmap_overlay(
                        image_paths[sample_index], attention[proposition_index, :image_token_count], overlay, alpha
                    )
                    caption = _caption_html(_caption_tokens(
                        tokenizer,
                        input_ids[sample_index],
                        masks[sample_index],
                        attention[proposition_index, image_token_count:],
                    ))
                    polarity = "positive" if record["polarity"] == 1 else "negative"
                    rows.append(f"""
                    <article class=\"proposition\">
                      <h4>{html.escape(record['canonicalDescription'])}</h4>
                      <p><b>Category:</b> {html.escape(record['category'])}; <b>polarity:</b> {polarity};
                      <b>support:</b> {support[sample_index, proposition_index]:.4f};
                      <b>compatibility:</b> {compatibility[sample_index, proposition_index]:.4f};
                      <b>attention entropy:</b> {entropy[sample_index, proposition_index]:.4f}</p>
                      <img src=\"{overlay.name}\" alt=\"Image attention heatmap\">
                      <p class=\"caption\">{caption}</p>
                      <p><b>Source:</b> {html.escape(record['sourceReference'])}</p>
                    </article>""")
                truth = " ✓ ground truth" if int(disease_position) == truth_index else ""
                cards.append(f"""
                <section class=\"disease\">
                  <h2>#{rank} {html.escape(disease['disease_id'])} — probability {predictions[sample_index, disease_position]:.4f}{truth}</h2>
                  {''.join(rows)}
                </section>""")

            source_image = Path(image_paths[sample_index]).resolve().as_uri()
            report = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Attention report {sample_index}</title>
<style>
body {{ font-family: Arial, sans-serif; color:#1f2937; margin:28px; max-width:1500px; }}
.disease {{ border-top:2px solid #1d4ed8; margin-top:24px; padding-top:8px; }}
.proposition {{ display:inline-block; vertical-align:top; width:31%; min-width:340px; margin:0 1% 18px 0; padding:12px; box-sizing:border-box; background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; }}
.proposition img {{ display:block; width:100%; max-width:450px; border-radius:5px; }}
.caption {{ line-height:2.15; background:white; padding:8px; min-height:52px; }}
h4 {{ margin:0; color:#7f1d1d; }}
</style></head><body>
<h1>Proposition attention report — sample {sample_index}</h1>
<p><b>Original image:</b> <a href=\"{source_image}\">{html.escape(image_paths[sample_index])}</a></p>
<p><b>Ground-truth disease:</b> {html.escape(diseases[truth_index]['disease_id'])}</p>
{''.join(cards)}
</body></html>"""
            (sample_dir / "report.html").write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--bert_model_name", required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--samples", nargs="+", type=int)
    parser.add_argument("--diseases_per_sample", type=int, default=3)
    parser.add_argument("--propositions_per_disease", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    destination = arguments.output_dir or arguments.evidence.parent / "rendered_attention"
    render(
        arguments.evidence,
        arguments.bert_model_name,
        destination,
        sample_indices=arguments.samples,
        diseases_per_sample=arguments.diseases_per_sample,
        propositions_per_disease=arguments.propositions_per_disease,
        alpha=arguments.alpha,
    )

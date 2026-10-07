"""Render standalone qualitative HTML reports from original-RAD attention evidence.

Run after ``export_rad_attention.py``. Rendering reads the evidence artifact only;
it never loads a model checkpoint or changes trained model parameters.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from render_attention import _caption_html, _caption_tokens, _heatmap_overlay

BRANCHES = ("guideline", "label")


def _read_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require(values: Any, names: set[str]) -> None:
    missing = sorted(names - set(values.files))
    if missing:
        raise KeyError(f"RAD evidence is missing: {', '.join(missing)}")


def render(
    evidence_path: Path,
    tokenizer_name: str,
    output_dir: Path,
    *,
    sample_indices: list[int] | None,
    alpha: float,
) -> list[Path]:
    """Create one self-contained HTML report and two heatmaps per selected case."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    schema = _read_schema(evidence_path.with_suffix(".schema.json"))
    diseases = [str(value) for value in schema["disease_ids"]]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(evidence_path, allow_pickle=False) as values:
        _require(
            values,
            {
                "image_paths",
                "gt",
                "image_token_count",
                "caption_input_ids",
                "caption_attention_mask",
                "query_token_position",
                "full_attention_sample_indices",
                "guideline_pred",
                "label_pred",
                "guideline_full_attention",
                "label_full_attention",
            },
        )
        full_positions = {
            int(sample): position
            for position, sample in enumerate(values["full_attention_sample_indices"])
        }
        selected = sample_indices if sample_indices is not None else sorted(full_positions)
        if not selected:
            raise ValueError(
                "No complete attention maps were stored. Re-run export_rad_attention.py "
                "with --full_attention_indices or --full_attention_first_n."
            )

        image_token_count = int(values["image_token_count"][0])
        caption_mask = values["caption_attention_mask"].astype(bool)
        label_prediction = values["label_pred"]
        guideline_prediction = values["guideline_pred"]
        fused_prediction = (
            values["fused_pred"]
            if "fused_pred" in values.files
            else (label_prediction + guideline_prediction) / 2.0
        )
        reports: list[Path] = []
        for sample_index in selected:
            if sample_index not in full_positions:
                raise ValueError(
                    f"Sample {sample_index} has no complete attention; choose one of "
                    f"{sorted(full_positions)}."
                )
            if not 0 <= sample_index < len(values["image_paths"]):
                raise IndexError(f"Sample index outside evidence: {sample_index}")

            truth = int(values["gt"][sample_index].argmax())
            top_diseases = np.argsort(-fused_prediction[sample_index])[:3]
            predictions = "".join(
                "<tr>"
                f"<td>{rank}</td><td>{html.escape(diseases[int(disease)])}</td>"
                f"<td>{fused_prediction[sample_index, disease]:.4f}</td>"
                f"<td>{label_prediction[sample_index, disease]:.4f}</td>"
                f"<td>{guideline_prediction[sample_index, disease]:.4f}</td>"
                "</tr>"
                for rank, disease in enumerate(top_diseases, start=1)
            )

            position = full_positions[sample_index]
            sample_dir = output_dir / f"sample_{sample_index:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            cards: list[str] = []
            for branch in BRANCHES:
                attention = values[f"{branch}_full_attention"][position, truth].astype(np.float32)
                true_probability = float(values[f"{branch}_pred"][sample_index, truth])
                image_mass = float(attention[:image_token_count].sum())
                caption_mass = float(
                    (attention[image_token_count:] * caption_mask[sample_index]).sum()
                )
                pad_mass = max(0.0, 1.0 - image_mass - caption_mass)
                heatmap = sample_dir / f"{branch}.png"
                _heatmap_overlay(
                    str(values["image_paths"][sample_index]),
                    attention[:image_token_count],
                    heatmap,
                    alpha,
                )
                caption = _caption_html(
                    _caption_tokens(
                        tokenizer,
                        values["caption_input_ids"][sample_index],
                        caption_mask[sample_index],
                        attention[image_token_count:],
                    )
                )
                cards.append(
                    f"""
                    <article class=\"card\">
                      <h2>{html.escape(branch.title())} decoder</h2>
                                            <p><b>P(true disease)</b> = {true_probability:.4f}
                      <br><b>Attention mass:</b> image {image_mass:.3f} · caption {caption_mass:.3f} · PAD {pad_mass:.3f}
                      <br><b>RAD query token position:</b> {int(values['query_token_position'][sample_index])}</p>
                      <img src=\"{heatmap.name}\" alt=\"{html.escape(branch)} attention heatmap\">
                      <p class=\"caption\">{caption}</p>
                    </article>"""
                )

            report = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>RAD attention {sample_index}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; }} th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }}
.card {{ display: inline-block; vertical-align: top; width: 47%; min-width: 320px; margin: 0 2% 20px 0; padding: 12px; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 8px; background: #f8fafc; }}
.card img {{ width: 100%; border-radius: 5px; }} .caption {{ line-height: 2.1; background: white; padding: 8px; }}
</style></head><body>
<h1>Original RAD — sample {sample_index}</h1>
<p><b>True disease:</b> {html.escape(diseases[truth])}<br><b>Image:</b> {html.escape(str(values['image_paths'][sample_index]))}<br>
Heatmap and caption colours are min–max normalized within each panel. Use printed attention mass for absolute allocation. PAD mass is shown because original RAD does not mask caption padding in the decoder.</p>
<h2>Top-3 diagnoses (fused)</h2>
<table><tr><th>Rank</th><th>Disease</th><th>Fused</th><th>Label</th><th>Guideline</th></tr>{predictions}</table>
{''.join(cards)}
</body></html>"""
            report_path = sample_dir / "report.html"
            report_path.write_text(report, encoding="utf-8")
            reports.append(report_path)

    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--bert_model_name", required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--samples", nargs="+", type=int)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    destination = arguments.output_dir or arguments.evidence.parent / "rendered_rad_attention"
    for created in render(
        arguments.evidence,
        arguments.bert_model_name,
        destination,
        sample_indices=arguments.samples,
        alpha=arguments.alpha,
    ):
        print(f"RAD attention report written to {created}")

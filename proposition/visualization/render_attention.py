"""Render UI-ready proposition attention reports from a saved test evidence artifact.

Run this after ``main_proposition.py`` has written checklist artifacts. Rendering
uses saved tensors only; it never loads a model checkpoint or retrains a model.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
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
        if token in special:
            continue
        # Reconstruct a WordPiece such as "mole" + "##s" for readable captions.
        if token.startswith("##") and result:
            previous_token, previous_weight = result[-1]
            result[-1] = (previous_token + token[2:], previous_weight + float(weight))
        else:
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
    ordered = sorted(candidates, key=lambda index: float(compatibility[index]), reverse=True)
    return ordered if maximum <= 0 else ordered[:maximum]


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


def _viewer_template(serialized_data: str) -> str:
        """Return a self-contained interactive proposition case viewer document."""

        # Keep the data inline so the viewer works from local folders and Colab
        # IFrames, where fetch() may be restricted for file URLs.
        return """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proposition clinical case viewer</title>
<style>
:root { --ink:#172033; --muted:#657089; --line:#dce3ee; --panel:#fff; --ground:#f5f7fb; --blue:#1f5bd8; --red:#c23938; --green:#087f5b; }
* { box-sizing:border-box; }
body { margin:0; background:var(--ground); color:var(--ink); font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
header { position:sticky; top:0; z-index:5; display:flex; align-items:center; gap:14px; min-height:64px; padding:10px max(20px,calc((100vw - 1540px)/2)); background:#10244a; color:#fff; box-shadow:0 2px 11px #0015364d; }
header h1 { margin:0 auto 0 0; font-size:17px; }
header small { display:block; color:#bfd1f5; font-weight:400; }
button, select, input { font:inherit; }
button, select { border:1px solid var(--line); border-radius:7px; background:#fff; padding:7px 9px; color:var(--ink); cursor:pointer; }
button:hover { border-color:var(--blue); }
.nav { white-space:nowrap; }
.nav button { margin-left:5px; }
main { max-width:1540px; margin:20px auto; padding:0 20px 30px; display:grid; grid-template-columns:minmax(420px,1.1fr) minmax(450px,1fr); gap:20px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:13px; box-shadow:0 3px 12px #13224a0b; padding:17px; }
h2, h3, h4, p { margin-top:0; }
h2 { font-size:18px; }
h3 { font-size:15px; margin-bottom:9px; }
.case-meta { color:var(--muted); font-size:13px; }
.image-wrap { position:relative; display:inline-block; width:100%; max-height:560px; overflow:hidden; background:#111827; border-radius:10px; text-align:center; }
#case-image { display:block; max-width:100%; max-height:560px; margin:auto; }
#heatmap { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.controls { display:flex; gap:13px; align-items:center; flex-wrap:wrap; margin:12px 0 4px; color:var(--muted); }
.controls label { display:flex; align-items:center; gap:5px; }
input[type="range"] { accent-color:var(--blue); }
.caption { line-height:2.25; min-height:52px; padding:9px; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; }
.token { border-radius:4px; padding:2px 3px; margin:1px; display:inline-block; }
.diagnoses { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; margin:9px 0 15px; }
.diagnosis { text-align:left; min-height:72px; border-left:4px solid #b8c5d9; }
.diagnosis.active { border-color:var(--blue); background:#edf3ff; }
.diagnosis .prob { font-weight:700; font-size:18px; color:var(--blue); }
.truth { color:var(--green); font-weight:650; }
.toolbar { display:flex; gap:9px; flex-wrap:wrap; align-items:center; margin-bottom:10px; }
.toolbar label { color:var(--muted); }
#props { display:grid; gap:8px; max-height:650px; overflow:auto; padding-right:4px; }
.prop { padding:11px; border:1px solid var(--line); border-radius:9px; background:#fff; cursor:pointer; }
.prop:hover, .prop.active { border-color:var(--blue); background:#f3f7ff; }
.prop-title { display:flex; align-items:flex-start; gap:8px; justify-content:space-between; font-weight:650; }
.badge { white-space:nowrap; border-radius:99px; padding:2px 7px; color:#35506d; background:#e8eff8; font-size:11px; font-weight:600; }
.negative { background:#fff0ef; color:#9d2922; }
.metrics { display:flex; gap:10px; flex-wrap:wrap; margin-top:7px; color:var(--muted); font-size:12px; }
.focus { margin-top:15px; border-top:1px solid var(--line); padding-top:14px; }
.source, .note { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
.note { margin-top:12px; }
@media (max-width:1000px) {
    main { grid-template-columns:1fr; }
    header { position:static; flex-wrap:wrap; }
    header h1 { min-width:100%; }
}
</style>
</head>
<body>
<header>
    <h1>Clinical case viewer <small>Proposition-model attention from saved best-checkpoint evidence</small></h1>
    <label>Ca bệnh <select id="case-select"></select></label>
    <span class="nav"><button id="previous" title="Ca trước (←)">←</button><button id="next" title="Ca sau (→)">→</button></span>
</header>
<main>
    <section class="panel">
        <h2 id="case-title">Ca bệnh</h2>
        <p class="case-meta" id="case-meta"></p>
        <div class="image-wrap"><img id="case-image" alt="Ảnh bệnh nhân"><canvas id="heatmap"></canvas></div>
        <div class="controls">
            <label><input id="show-heat" type="checkbox" checked> Hiện attention</label>
            <label>Opacity <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.62"></label>
            <label><input id="show-grid" type="checkbox"> Lưới patch</label>
        </div>
        <h3>Caption và attention của mệnh đề đang chọn</h3>
        <div class="caption" id="caption"></div>
        <p class="note">Màu heatmap và caption được chuẩn hoá min-max trong mệnh đề đang xem. Đây là cross-attention đã lưu, không phải vùng tổn thương được gán nhãn.</p>
    </section>
    <section class="panel">
        <h2>Chẩn đoán</h2>
        <p class="case-meta">Chọn một bệnh để xem các mệnh đề quy về bệnh đó.</p>
        <div id="diagnoses" class="diagnoses"></div>
        <div class="toolbar">
            <label>Nhóm <select id="category"><option value="all">Tất cả</option></select></label>
            <label>Sắp xếp <select id="sort"><option value="compatibility">Compatibility</option><option value="support">Support</option><option value="entropy">Entropy thấp</option></select></label>
        </div>
        <div id="props"></div>
        <div class="focus"><h3 id="focus-title">Mệnh đề</h3><p id="focus-details"></p><p class="source" id="focus-source"></p></div>
    </section>
</main>
<script>
const DATA = __CASE_DATA__;
const state = { caseIndex:0, diseaseIndex:0, propIndex:0, category:"all", sort:"compatibility", heat:true, grid:false, opacity:.62 };
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[char]));
const currentCase = () => DATA.cases[state.caseIndex];
const currentDisease = () => currentCase().diseases[state.diseaseIndex];
function currentProp() {
    const props = currentDisease().propositions;
    return props.find(prop => prop.id === state.propIndex) || props[0];
}
function normalize(values) {
    const minimum = Math.min(...values), maximum = Math.max(...values);
    return values.map(value => maximum > minimum ? (value - minimum) / (maximum - minimum) : 0);
}
function drawOverlay() {
    const image = $("case-image"), canvas = $("heatmap"), prop = currentProp();
    if (!image.complete || !prop) return;
    const width = image.clientWidth, height = image.clientHeight;
    canvas.width = Math.max(1, Math.round(width * devicePixelRatio));
    canvas.height = Math.max(1, Math.round(height * devicePixelRatio));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    context.scale(devicePixelRatio, devicePixelRatio);
    context.clearRect(0, 0, width, height);
    if (!state.heat) return;
    const values = normalize(prop.img);
    const side = Math.round(Math.sqrt(values.length));
    if (side * side !== values.length) return;
    const cellWidth = width / side, cellHeight = height / side;
    context.globalAlpha = state.opacity;
    values.forEach((value, index) => {
        context.fillStyle = `hsl(${(1 - value) * 240}, 92%, 48%)`;
        context.fillRect((index % side) * cellWidth, Math.floor(index / side) * cellHeight, cellWidth + 1, cellHeight + 1);
    });
    if (state.grid) {
        context.globalAlpha = .5;
        context.strokeStyle = "#ffffff";
        context.lineWidth = .5;
        for (let index = 0; index <= side; index++) {
            context.beginPath(); context.moveTo(index * cellWidth, 0); context.lineTo(index * cellWidth, height); context.stroke();
            context.beginPath(); context.moveTo(0, index * cellHeight); context.lineTo(width, index * cellHeight); context.stroke();
        }
    }
}
function renderEvidence() {
    const prop = currentProp();
    if (!prop) return;
    const values = normalize(prop.cap.map(item => item.w));
    $("caption").innerHTML = prop.cap.length
        ? prop.cap.map((item, index) => `<span class="token" title="attention=${item.w.toFixed(6)}" style="background:rgba(214,52,52,${.08 + .77 * values[index]})">${esc(item.t)}</span>`).join(" ")
        : "<em>Không có caption token hợp lệ.</em>";
    const polarity = prop.polarity > 0 ? "Khẳng định" : "Phủ định";
    $("focus-title").textContent = prop.text;
    $("focus-details").innerHTML = `<b>Nhóm:</b> ${esc(prop.category)} · <b>Polarity:</b> ${polarity}<br><b>Support:</b> ${prop.support.toFixed(4)} · <b>Compatibility:</b> ${prop.compatibility.toFixed(4)} · <b>Entropy:</b> ${prop.entropy.toFixed(4)}`;
    $("focus-source").innerHTML = `<b>Nguồn guideline:</b> ${esc(prop.source || "Không có")}`;
    drawOverlay();
}
function renderProps() {
    const selected = currentProp();
    let props = currentDisease().propositions.filter(prop => state.category === "all" || prop.category === state.category);
    props.sort((left, right) => state.sort === "entropy" ? left.entropy - right.entropy : right[state.sort] - left[state.sort]);
    $("props").innerHTML = props.map(prop => `<article class="prop ${selected && prop.id === selected.id ? "active" : ""}" data-id="${prop.id}"><div class="prop-title"><span>${esc(prop.text)}</span><span class="badge ${prop.polarity < 0 ? "negative" : ""}">${esc(prop.category)}</span></div><div class="metrics"><span>support ${prop.support.toFixed(3)}</span><span>compatibility ${prop.compatibility.toFixed(3)}</span><span>entropy ${prop.entropy.toFixed(3)}</span></div></article>`).join("") || "<p>Không có mệnh đề thuộc bộ lọc này.</p>";
    document.querySelectorAll(".prop[data-id]").forEach(node => {
        node.onclick = () => { state.propIndex = Number(node.dataset.id); renderProps(); renderEvidence(); };
    });
}
function populateCategories() {
    const select = $("category");
    const previous = state.category;
    const categories = [...new Set(currentDisease().propositions.map(prop => prop.category))].sort();
    select.innerHTML = '<option value="all">Tất cả</option>' + categories.map(category => `<option value="${esc(category)}">${esc(category)}</option>`).join("");
    state.category = categories.includes(previous) ? previous : "all";
    select.value = state.category;
}
function renderDiagnoses() {
    $("diagnoses").innerHTML = currentCase().diseases.map((disease, index) => `<button class="diagnosis ${index === state.diseaseIndex ? "active" : ""}" data-index="${index}"><b>#${index + 1} ${esc(disease.name)}</b><br><span class="prob">${(disease.prob * 100).toFixed(1)}%</span>${disease.truth ? '<br><span class="truth">✓ nhãn thật</span>' : ""}</button>`).join("");
    document.querySelectorAll(".diagnosis").forEach(node => {
        node.onclick = () => {
            state.diseaseIndex = Number(node.dataset.index);
            state.propIndex = currentDisease().propositions[0]?.id;
            populateCategories();
            renderDiagnoses();
            renderProps();
            renderEvidence();
        };
    });
}
function renderCase() {
    const item = currentCase();
    state.diseaseIndex = 0;
    state.propIndex = item.diseases[0]?.propositions[0]?.id;
    $("case-title").textContent = `Ca ${item.index} — ${item.gt}`;
    $("case-meta").textContent = `Ảnh: ${item.imageName} · chỉ gồm các ca có full attention đã lưu`;
    $("case-image").src = item.image;
    $("case-image").onload = drawOverlay;
    populateCategories();
    renderDiagnoses();
    renderProps();
    renderEvidence();
    $("case-select").value = String(state.caseIndex);
}
$("case-select").onchange = event => { state.caseIndex = Number(event.target.value); renderCase(); };
$("previous").onclick = () => { state.caseIndex = (state.caseIndex + DATA.cases.length - 1) % DATA.cases.length; renderCase(); };
$("next").onclick = () => { state.caseIndex = (state.caseIndex + 1) % DATA.cases.length; renderCase(); };
$("category").onchange = event => { state.category = event.target.value; renderProps(); };
$("sort").onchange = event => { state.sort = event.target.value; renderProps(); };
$("show-heat").onchange = event => { state.heat = event.target.checked; drawOverlay(); };
$("show-grid").onchange = event => { state.grid = event.target.checked; drawOverlay(); };
$("opacity").oninput = event => { state.opacity = Number(event.target.value); drawOverlay(); };
window.onresize = drawOverlay;
window.onkeydown = event => {
    if (event.target.matches("input,select")) return;
    if (event.key === "ArrowLeft") $("previous").click();
    if (event.key === "ArrowRight") $("next").click();
};
$("case-select").innerHTML = DATA.cases.map((item, index) => `<option value="${index}">Ca ${item.index}: ${esc(item.gt)}</option>`).join("");
renderCase();
</script>
</body>
</html>""".replace("__CASE_DATA__", serialized_data)


def render_viewer(
    evidence_path: Path,
    tokenizer_name: str,
    output_dir: Path,
    *,
    sample_indices: list[int] | None,
    diseases_per_sample: int,
    propositions_per_disease: int,
) -> Path:
    """Create an interactive, patient-centred viewer from saved artifacts."""

    checklist_dir = evidence_path.parent
    schema = _read_json(evidence_path.with_suffix(".schema.json"))
    if schema["query_level"] != "proposition":
        raise ValueError("case viewer requires proposition-level query artifacts")
    kb = _read_json(checklist_dir / "proposition_index.json")
    diseases = _read_json(checklist_dir / "disease_index.json")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

    with np.load(evidence_path, allow_pickle=False) as values:
        image_paths = values["image_paths"].astype(str)
        predictions = values["pred_proposition"]
        ground_truth = values["gt"]
        support = values["support"]
        compatibility = values["compatibility"]
        entropy = values["attention_entropy"]
        full_attention = values["full_attention"]
        full_attention_indices = values["full_attention_sample_indices"].astype(int)
        input_ids = values["caption_input_ids"]
        masks = values["caption_attention_mask"]
        image_token_count = int(values["image_token_count"][0])
        attention_by_sample = {
            int(sample): full_attention[position]
            for position, sample in enumerate(full_attention_indices)
        }

        selected = sorted(attention_by_sample) if sample_indices is None else sample_indices
        if not selected:
            raise ValueError("the evidence artifact has no samples with full attention")

        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        cases: list[dict[str, Any]] = []
        for sample_index in selected:
            if not 0 <= sample_index < len(image_paths):
                raise IndexError(f"sample index out of range: {sample_index}")
            if sample_index not in attention_by_sample:
                raise ValueError(f"sample {sample_index} has no stored full attention")

            source = Path(image_paths[sample_index])
            if not source.is_file():
                raise FileNotFoundError(f"source image does not exist: {source}")
            asset_name = f"sample_{sample_index:04d}{source.suffix.lower() or '.png'}"
            shutil.copy2(source, assets_dir / asset_name)

            truth_index = int(np.argmax(ground_truth[sample_index]))
            ranking = np.argsort(-predictions[sample_index])[:diseases_per_sample]
            disease_entries: list[dict[str, Any]] = []
            for disease_position in ranking:
                proposition_entries: list[dict[str, Any]] = []
                for proposition_index in _selected_propositions(
                    int(disease_position),
                    compatibility[sample_index],
                    kb,
                    propositions_per_disease,
                ):
                    record = kb[proposition_index]
                    attention = attention_by_sample[sample_index][proposition_index].astype(np.float32)
                    caption = _caption_tokens(
                        tokenizer,
                        input_ids[sample_index],
                        masks[sample_index],
                        attention[image_token_count:],
                    )
                    proposition_entries.append({
                        "id": int(proposition_index),
                        "text": record["canonicalDescription"],
                        "category": record["category"],
                        "polarity": int(record["polarity"]),
                        "support": round(float(support[sample_index, proposition_index]), 6),
                        "compatibility": round(float(compatibility[sample_index, proposition_index]), 6),
                        "entropy": round(float(entropy[sample_index, proposition_index]), 6),
                        "source": record.get("sourceReference", ""),
                        "img": np.round(attention[:image_token_count], 7).tolist(),
                        "cap": [
                            {"t": token, "w": round(float(weight), 7)}
                            for token, weight in caption
                        ],
                    })
                disease_entries.append({
                    "name": diseases[int(disease_position)]["disease_id"],
                    "prob": round(float(predictions[sample_index, disease_position]), 7),
                    "truth": bool(int(disease_position) == truth_index),
                    "propositions": proposition_entries,
                })
            cases.append({
                "index": int(sample_index),
                "image": f"assets/{asset_name}",
                "imageName": source.name,
                "gt": diseases[truth_index]["disease_id"],
                "diseases": disease_entries,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "model": "proposition",
            "full_attention_cases": len(cases),
            "image_token_count": image_token_count,
        },
        "cases": cases,
    }
    # Prevent an unlikely literal closing script tag in guideline descriptions.
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    viewer_path = output_dir / "index.html"
    viewer_path.write_text(_viewer_template(serialized), encoding="utf-8")
    return viewer_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--bert_model_name", required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--samples", nargs="+", type=int)
    parser.add_argument("--diseases_per_sample", type=int, default=3)
    parser.add_argument("--propositions_per_disease", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--format",
        choices=("static", "viewer"),
        default="static",
        help="'static' writes one report per case; 'viewer' writes an interactive index.html.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    destination = arguments.output_dir or arguments.evidence.parent / "rendered_attention"
    if arguments.format == "viewer":
        viewer = render_viewer(
            arguments.evidence,
            arguments.bert_model_name,
            destination,
            sample_indices=arguments.samples,
            diseases_per_sample=arguments.diseases_per_sample,
            propositions_per_disease=arguments.propositions_per_disease,
        )
        print(f"Interactive case viewer written to {viewer}")
    else:
        render(
            arguments.evidence,
            arguments.bert_model_name,
            destination,
            sample_indices=arguments.samples,
            diseases_per_sample=arguments.diseases_per_sample,
            propositions_per_disease=arguments.propositions_per_disease,
            alpha=arguments.alpha,
        )

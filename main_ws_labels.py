"""WS_03 command-line runner: masked NLI, optional ConceptCLIP, A2b, survey.

Default execution is deliberately safe: it creates/requires alias review,
runs no LLM API calls, and never writes to the immutable WS_02 directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

from proposition.weak.labels import (
    ABSENT, LLM_ABSENT, LLM_GROUP_CODES, LLM_NOT_MENTIONED, LLM_NOT_SENT,
    LLM_PRESENT, PRESENT, UNKNOWN, WeakLabels, load_propositions,
    mask_disease_aliases, merge_labels, read_alias_csv, sha256_file,
    split_sentences, suggested_alias_rows, write_alias_csv,
)
from proposition.weak.llm_verify import LlmCell, dry_run_report, verify_cells
from proposition.weak.reporting import write_survey

LOG = logging.getLogger("ws03")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ws.yaml"))
    parser.add_argument("--teacher", choices=("text", "image", "both"), default="both")
    parser.add_argument("--llm-mode", choices=("none", "dry-run", "run"), default="dry-run")
    parser.add_argument("--approve-llm", action="store_true", help="Required together with --llm-mode run.")
    parser.add_argument("--alias-csv", type=Path, default=None, help="Approved alias CSV. Defaults to output/disease_alias_v1.csv.")
    parser.add_argument("--prepare-aliases", action="store_true", help="Create a conservative alias draft then stop.")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--survey-only", action="store_true", help="Read existing labels.npz and recreate survey files only.")
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def _load_cases(cfg: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    cases: list[dict[str, str]] = []
    disease_columns: list[str] = []
    for split in ("train", "val", "test"):
        path = Path(cfg["skincap"][f"{split}_csv"])
        _require_file(path, f"SkinCAP {split} CSV")
        frame = pd.read_csv(path, dtype=str).fillna("")
        if frame.empty or frame.shape[1] < 2:
            raise ValueError(f"{path} must have image filename and English caption columns")
        case_column, caption_column = frame.columns[:2]
        if split == "train":
            disease_columns = list(frame.columns[2:])
        for _, item in frame.iterrows():
            case_id = str(item[case_column])
            if not case_id:
                raise ValueError(f"Blank case_id in {path}")
            positives = [name for name in disease_columns if name in item and str(item[name]) in {"1", "1.0", "True", "true"}]
            cases.append({"case_id": case_id, "split": split, "caption": str(item[caption_column]), "disease": positives[0] if len(positives) == 1 else ""})
    duplicate = [case for case, count in Counter(case["case_id"] for case in cases).items() if count > 1]
    if duplicate:
        raise ValueError(f"case_id must be globally unique across splits; duplicates: {duplicate[:5]}")
    return cases, disease_columns


def _nli_labels(cases: list[dict[str, str]], propositions, aliases: list[str], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(cfg["teacher_text_model"])
    _require_file(model_path / "config.json", "Teacher config")
    label_map_path = model_path.parent / "label_map.json"
    _require_file(label_map_path, "Teacher label_map")
    mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
    if mapping != {"entailment": 0, "neutral": 1, "contradiction": 2}:
        raise ValueError(f"Unexpected teacher label map: {mapping}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    n_cases, n_props = len(cases), len(propositions)
    label = np.full((n_cases, n_props), UNKNOWN, dtype=np.int8)
    p_entail = np.zeros((n_cases, n_props), dtype=np.float16)
    p_contra = np.zeros((n_cases, n_props), dtype=np.float16)
    best_sentence = np.full((n_cases, n_props), -1, dtype=np.int16)
    masked_sentences: dict[str, list[str]] = {}
    options = cfg["text_labeler"]
    autocast_enabled = bool(options.get("fp16", False)) and device.type == "cuda"
    batch_size = int(options["batch_size"])

    for row, case in enumerate(tqdm(cases, desc="WS_03 NLI")):
        sentences = split_sentences(mask_disease_aliases(case["caption"], aliases))
        masked_sentences[case["case_id"]] = sentences
        if not sentences:
            continue
        tasks = [(column, h_index, sentence_index, sentence, hypothesis) for column, proposition in enumerate(propositions) for h_index, hypothesis in enumerate(proposition.hypotheses) for sentence_index, sentence in enumerate(sentences)]
        ent = [[0.0 for _ in proposition.hypotheses] for proposition in propositions]
        con = [[0.0 for _ in proposition.hypotheses] for proposition in propositions]
        best = [[-1 for _ in proposition.hypotheses] for proposition in propositions]
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            encoded = tokenizer([item[3] for item in batch], [item[4] for item in batch], truncation=True, max_length=int(options["max_length"]), padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().numpy()
            for task, prob in zip(batch, probabilities):
                column, h_index, sentence_index = task[:3]
                if float(prob[0]) > ent[column][h_index]:
                    ent[column][h_index] = float(prob[0]); best[column][h_index] = sentence_index
                con[column][h_index] = max(con[column][h_index], float(prob[2]))
        for column, proposition in enumerate(propositions):
            p_entail[row, column] = max(ent[column])
            p_contra[row, column] = max(con[column])
            winning = int(np.argmax(ent[column]))
            best_sentence[row, column] = best[column][winning]
            if any(value >= float(options["entailment_threshold"]) for value in ent[column]):
                label[row, column] = PRESENT
            elif all(value >= float(options["contradiction_threshold"]) for value in con[column]):
                label[row, column] = ABSENT
    return label, p_entail, p_contra, best_sentence, masked_sentences


def _image_labels(cases: list[dict[str, str]], propositions, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """ConceptCLIP implementation for normal CLIP-compatible remote-code APIs."""
    from transformers import AutoModel, AutoProcessor

    opts = cfg["image_labeler"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(opts["model_id"], trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(opts["model_id"], trust_remote_code=True)
    applicable = [(index, prop) for index, prop in enumerate(propositions) if prop.image_applicable]
    n_cases, n_props = len(cases), len(propositions)
    label = np.full((n_cases, n_props), UNKNOWN, dtype=np.int8)
    scores = np.full((n_cases, n_props), np.nan, dtype=np.float16)
    prompts = [text for _, prop in applicable for hypothesis in prop.sub_hypotheses for text in (f"a clinical photo of skin with {hypothesis}", f"this skin lesion shows {hypothesis}", f"a clinical photo of skin without {hypothesis}", f"this skin lesion does not show {hypothesis}")]
    try:
        tokens = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            text_features = model.get_text_features(**tokens)
    except AttributeError as error:
        raise RuntimeError("ConceptCLIP remote code lacks get_text_features; inspect its model card/API before running image teacher.") from error
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    image_root = Path(cfg["skincap"]["image_root"])
    for row, case in enumerate(tqdm(cases, desc="WS_03 ConceptCLIP")):
        image_path = Path(case["case_id"])
        image_path = image_path if image_path.is_absolute() else image_root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image for {case['case_id']} is missing: {image_path}")
        try:
            image_inputs = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")
            image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
            with torch.inference_mode():
                image_feature = model.get_image_features(**image_inputs)
        except AttributeError as error:
            raise RuntimeError("ConceptCLIP remote code lacks get_image_features; inspect its model card/API before running image teacher.") from error
        image_feature = torch.nn.functional.normalize(image_feature.float(), dim=-1)[0]
        offset = 0
        for column, prop in applicable:
            hypothesis_scores: list[float] = []
            for _ in prop.sub_hypotheses:
                pos = (image_feature @ text_features[offset:offset + 2].T).mean()
                neg = (image_feature @ text_features[offset + 2:offset + 4].T).mean()
                logit_scale = getattr(model, "logit_scale", torch.tensor(1.0, device=device))
                scale = logit_scale.exp() if torch.is_tensor(logit_scale) else 1.0
                hypothesis_scores.append(float(torch.softmax(scale * torch.stack((pos, neg)), dim=0)[0]))
                offset += 4
            score = max(hypothesis_scores)
            scores[row, column] = score
            if score >= float(opts["positive_threshold"]): label[row, column] = PRESENT
            elif all(value <= float(opts["negative_threshold"]) for value in hypothesis_scores): label[row, column] = ABSENT
    return label, scores


def _select_llm_cells(cases, propositions, nli, p_entail, p_contra, best_sentence, sentences, cfg):
    opts = cfg["llm_verify"]; terms = tuple(str(term).casefold() for term in opts["negation_terms"])
    selected: list[LlmCell] = []
    groups = np.zeros(nli.shape, dtype=np.int8)
    for row, case in enumerate(cases):
        for column, proposition in enumerate(propositions):
            value = nli[row, column]; group = ""
            sentence_index = int(best_sentence[row, column])
            evidence = sentences[case["case_id"]][sentence_index].casefold() if sentence_index >= 0 else ""
            has_negation = any(term in evidence for term in terms)
            if value == PRESENT and proposition.category in {"symptom", "history"} and opts.get("group_d", False): group = "D"
            elif value == PRESENT and has_negation and opts.get("group_a", False): group = "A"
            elif value == UNKNOWN and .5 <= float(p_contra[row, column]) < float(cfg["text_labeler"]["contradiction_threshold"]) and opts.get("group_b", False): group = "B"
            elif value == UNKNOWN and .5 <= float(p_entail[row, column]) < float(cfg["text_labeler"]["entailment_threshold"]) and opts.get("group_c", False): group = "C"
            if group:
                groups[row, column] = LLM_GROUP_CODES[group]
                selected.append(LlmCell(row, column, case["case_id"], proposition.proposition_id, group, proposition.sub_hypotheses, " ".join(sentences[case["case_id"]])))
    return selected, groups


def _apply_llm(nli: np.ndarray, cells: Iterable[LlmCell], answers: dict[tuple[int, int], dict[str, Any]], groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = nli.copy(); verdicts = np.full(nli.shape, LLM_NOT_SENT, dtype=np.int8); invalid = np.zeros(nli.shape, dtype=np.int8)
    for cell in cells:
        result = answers.get((cell.row, cell.column), {"verdict": "NOT_MENTIONED", "quote_invalid": 0})
        verdict = result["verdict"]; invalid[cell.row, cell.column] = int(result.get("quote_invalid", 0))
        verdicts[cell.row, cell.column] = {"NOT_MENTIONED": LLM_NOT_MENTIONED, "PRESENT": LLM_PRESENT, "ABSENT": LLM_ABSENT}.get(verdict, LLM_NOT_MENTIONED)
        if cell.group in {"A", "D"}: label[cell.row, cell.column] = {"PRESENT": PRESENT, "ABSENT": ABSENT}.get(verdict, UNKNOWN)
        elif cell.group == "B" and verdict == "ABSENT": label[cell.row, cell.column] = ABSENT
        elif cell.group == "C" and verdict == "PRESENT": label[cell.row, cell.column] = PRESENT
    return label, verdicts, invalid


def main() -> None:
    args = parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _require_file(args.config, "Config")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(cfg["weak_labels_dir"]); stage2 = Path(cfg["propositions_dir"])
    if output.resolve() == stage2.resolve() or stage2 in output.parents:
        raise ValueError("weak_labels_dir must not be inside the immutable WS_02 proposition directory")
    propositions = load_propositions(cfg["propositions_final"], cfg["propositions_review"])
    cases, disease_columns = _load_cases(cfg)
    if args.survey_only:
        from proposition.weak.labels import load_weak_labels
        labels = load_weak_labels(output / "labels.npz")
        survey = write_survey(labels, propositions, output / "survey", case_diseases=[case["disease"] for case in cases])
        LOG.info("Survey written: %s", survey); return
    alias_path = args.alias_csv or output / "disease_alias_v1.csv"
    if args.prepare_aliases:
        if alias_path.exists(): raise FileExistsError(f"Refusing to overwrite alias draft: {alias_path}")
        write_alias_csv(alias_path, suggested_alias_rows(disease_columns))
        LOG.info("Alias draft written for review: %s. Review it, then rerun without --prepare-aliases.", alias_path); return
    _require_file(alias_path, "Approved alias CSV")
    output.mkdir(parents=True, exist_ok=True)
    aliases = read_alias_csv(alias_path)
    text, p_entail, p_contra, best_sentence, sentences = _nli_labels(cases, propositions, aliases, cfg) if args.teacher in {"text", "both"} else (np.full((len(cases), len(propositions)), UNKNOWN, np.int8), np.zeros((len(cases), len(propositions)), np.float16), np.zeros((len(cases), len(propositions)), np.float16), np.full((len(cases), len(propositions)), -1, np.int16), {case["case_id"]: split_sentences(mask_disease_aliases(case["caption"], aliases)) for case in cases})
    cells, groups = _select_llm_cells(cases, propositions, text, p_entail, p_contra, best_sentence, sentences, cfg)
    (output / "sentences_masked.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8")
    case_split = {case["case_id"]: case["split"] for case in cases}
    dry_rows = []
    for group in ("A", "B", "C", "D"):
        for split in ("train", "val", "test"):
            selected = [cell for cell in cells if cell.group == group and case_split[cell.case_id] == split]
            if selected:
                dry_rows.append({"group": group, "split": split, "n_cells": len(selected), "n_cases": len({cell.case_id for cell in selected}), "estimated_calls": len({cell.case_id for cell in selected}), "estimated_input_tokens": sum(len(cell.masked_caption) // 4 + sum(len(hypothesis) // 4 for hypothesis in cell.sub_hypotheses) for cell in selected)})
    dry = pd.DataFrame(dry_rows, columns=("group", "split", "n_cells", "n_cases", "estimated_calls", "estimated_input_tokens")); dry.to_csv(output / "llm_dry_run_report.csv", index=False)
    LOG.info("LLM dry run: %d selected cells / %d case calls", len(cells), len({cell.case_id for cell in cells}))
    if args.llm_mode == "run":
        answers = verify_cells(cells, settings=cfg["llm_verify"], prompt_path=cfg["llm_verify"]["prompt_path"], cache_path=output / "llm_cache.jsonl", approve=args.approve_llm)
        text_final, verdicts, invalid = _apply_llm(text, cells, answers, groups)
    else:
        text_final, verdicts, invalid = text.copy(), np.full(text.shape, LLM_NOT_SENT, np.int8), np.zeros(text.shape, np.int8)
    image, p_img = _image_labels(cases, propositions, cfg) if args.teacher in {"image", "both"} else (np.full(text.shape, UNKNOWN, np.int8), np.full(text.shape, np.nan, np.float16))
    arrays = {"case_id": np.asarray([case["case_id"] for case in cases]), "split": np.asarray([case["split"] for case in cases]), "proposition_id": np.asarray([prop.proposition_id for prop in propositions]), "label_text_nli": text, "label_text": text_final, "llm_group": groups, "llm_verdict": verdicts, "llm_quote_invalid": invalid, "label_img": image, "label_final": merge_labels(text_final, image, args.teacher), "p_entail": p_entail, "p_contra": p_contra, "p_img": p_img, "best_sentence_idx": best_sentence}
    labels = WeakLabels(arrays); labels.save(output / "labels.npz")
    meta = {"stage": "WS_03_weak_labels", "teacher": args.teacher, "llm_mode": args.llm_mode, "seed": cfg["seed"], "n_cases": len(cases), "n_propositions": len(propositions), "propositions_final_sha256": sha256_file(cfg["propositions_final"]), "propositions_review_sha256": sha256_file(cfg["propositions_review"]), "teacher_text_model": cfg["teacher_text_model"], "thresholds": {"text": cfg["text_labeler"], "image": cfg["image_labeler"]}}
    (output / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    survey = write_survey(labels, propositions, output / "survey", case_diseases=[case["disease"] for case in cases])
    LOG.info("WS_03 complete: %s; survey: %s", output / "labels.npz", survey)


if __name__ == "__main__":
    main()
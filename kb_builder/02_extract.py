"""Bước 2: gọi LLM để giữ/loại từng đơn vị nguồn và trích mệnh đề (1 lần gọi cho mỗi bệnh).

Chạy:  python 02_extract.py                 # tất cả bệnh chưa có kết quả
       python 02_extract.py --only acne psoriasis
       python 02_extract.py --force         # chạy lại cả những bệnh đã có

Đầu ra:
  outputs/extract/<slug>.prompt.txt     prompt đã gửi (để lưu vết)
  outputs/extract/<slug>.response.txt   phản hồi thô của LLM
  outputs/extract/<slug>.meta.json      model, provider, token, thời gian, phiên bản prompt
  outputs/extract/<slug>.json           JSON đã parse và kiểm tra cấu trúc
"""
import argparse
import datetime
import json
from collections import OrderedDict

from common import (
    CATEGORIES, EXCLUDE_REASONS, extract_json, load_config, load_prompt, read_jsonl, resolve, slug,
)
from llm_client import call_llm


def format_units(units):
    lines = []
    for u in units:
        path = u["section"] + (f" > {u['subsection']}" if u["subsection"] else "")
        lines.append(f"[{u['unit_id']}] ({path}) {u['text']}")
    return "\n".join(lines)


def validate(data, units, disease):
    """Kiểm tra cấu trúc; trả về danh sách lỗi (rỗng nếu hợp lệ)."""
    errs = []
    if data.get("disease") != disease:
        errs.append(f"disease mismatch: {data.get('disease')!r}")
    got = [u.get("unit_id") for u in data.get("units", [])]
    want = [u["unit_id"] for u in units]
    if got != want:
        errs.append(f"unit ids/order mismatch: missing={sorted(set(want) - set(got))} extra={sorted(set(got) - set(want))}")
    for u in data.get("units", []):
        uid, dec, props = u.get("unit_id"), u.get("decision"), u.get("propositions", [])
        if dec == "KEEP":
            if not props:
                errs.append(f"{uid}: KEEP without propositions")
            if u.get("exclude_reason") not in (None, ""):
                errs.append(f"{uid}: KEEP with exclude_reason")
        elif dec == "EXCLUDE":
            if props:
                errs.append(f"{uid}: EXCLUDE with propositions")
            if u.get("exclude_reason") not in EXCLUDE_REASONS:
                errs.append(f"{uid}: bad exclude_reason {u.get('exclude_reason')!r}")
        else:
            errs.append(f"{uid}: bad decision {dec!r}")
        for p in props:
            if p.get("polarity") not in (1, -1):
                errs.append(f"{uid}: bad polarity {p.get('polarity')!r}")
            if p.get("category") not in CATEGORIES:
                errs.append(f"{uid}: bad category {p.get('category')!r}")
            if not str(p.get("canonical", "")).strip():
                errs.append(f"{uid}: empty canonical")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="chỉ chạy các bệnh này (tên đúng như trong guideline)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = cfg["_out"] / "extract"
    out_dir.mkdir(parents=True, exist_ok=True)
    version = cfg["prompts"]["extract"]
    system, user_tpl = load_prompt("extract", version)
    synonyms = json.load(open(resolve(cfg, "synonyms_path"), encoding="utf-8"))

    units = read_jsonl(cfg["_out"] / "units.jsonl")
    by_disease = OrderedDict()
    for u in units:
        by_disease.setdefault(u["disease"], []).append(u)
    labels = list(by_disease)
    missing_syn = [d for d in labels if d not in synonyms]
    if missing_syn:
        raise ValueError(f"synonyms.json thiếu: {missing_syn}")

    todo = args.only or labels
    for disease in todo:
        if disease not in by_disease:
            raise ValueError(f"Unknown disease: {disease}")
        s = slug(disease)
        if (out_dir / f"{s}.json").exists() and not args.force:
            print(f"skip {disease} (đã có)")
            continue
        names = synonyms[disease]["names"] + synonyms[disease]["abbreviations"]
        user = user_tpl.format(
            disease=disease,
            target_names=", ".join(names),
            other_labels=", ".join(d for d in labels if d != disease),
            units=format_units(by_disease[disease]),
        )
        (out_dir / f"{s}.prompt.txt").write_text(f"### SYSTEM\n{system}\n\n### USER\n{user}", encoding="utf-8")
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        text, meta = call_llm(system, user, cfg["llm"])
        (out_dir / f"{s}.response.txt").write_text(text, encoding="utf-8")
        meta.update({"disease": disease, "prompt": f"extract_{version}", "requested_model": cfg["llm"]["model"],
                     "temperature": cfg["llm"].get("temperature"), "started_utc": started})
        (out_dir / f"{s}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            data = extract_json(text)
        except Exception as e:
            print(f"!! {disease}: không parse được JSON ({e}). Xem {s}.response.txt")
            continue
        errs = validate(data, by_disease[disease], disease)
        if errs:
            print(f"!! {disease}: {len(errs)} lỗi cấu trúc, ví dụ: {errs[:3]}")
            (out_dir / f"{s}.errors.json").write_text(json.dumps(errs, indent=2, ensure_ascii=False), encoding="utf-8")
            continue
        (out_dir / f"{s}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        n_keep = sum(u["decision"] == "KEEP" for u in data["units"])
        n_prop = sum(len(u["propositions"]) for u in data["units"])
        print(f"ok {disease}: keep {n_keep}/{len(data['units'])} units, {n_prop} propositions")


if __name__ == "__main__":
    main()

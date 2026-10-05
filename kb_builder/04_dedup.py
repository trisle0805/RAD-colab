"""Bước 4: gộp mệnh đề trùng trong cùng một bệnh.

(a) Gộp tự động các mệnh đề trùng chữ sau chuẩn hóa và cùng polarity.
(b) Gọi LLM tìm các nhóm cùng nghĩa (LLM chỉ trả về id, không viết văn bản mới).
    Bỏ qua (b) bằng --no-llm.

Đầu ra:
  outputs/dedup/<slug>.response.txt, <slug>.meta.json   (nếu dùng LLM)
  outputs/props_dedup.jsonl                             mệnh đề sau gộp, đã có proposition_id
"""
import argparse
import datetime
import json

from common import extract_json, load_config, load_prompt, norm_text, read_jsonl, slug, write_jsonl
from llm_client import call_llm


def collect(data, units_by_id):
    props = []
    for u in data["units"]:
        for k, p in enumerate(u["propositions"]):
            props.append({
                "tmp_id": f"{u['unit_id']}#{k}",
                "canonical": p["canonical"].strip(),
                "polarity": p["polarity"],
                "category": p["category"],
                "unit_ids": [u["unit_id"]],
            })
    return props


def merge_group(members, rep):
    unit_ids = []
    for m in members:
        for uid in m["unit_ids"]:
            if uid not in unit_ids:
                unit_ids.append(uid)
    return {"tmp_id": rep["tmp_id"], "canonical": rep["canonical"], "polarity": rep["polarity"],
            "category": rep["category"], "unit_ids": unit_ids,
            "merged_from": sorted({t for m in members for t in m.get("merged_from", [m["tmp_id"]])})}


def exact_dedup(props):
    seen, out = {}, []
    for p in props:
        key = (norm_text(p["canonical"]), p["polarity"])
        if key in seen:
            i = seen[key]
            out[i] = merge_group([out[i], p], out[i])
        else:
            seen[key] = len(out)
            out.append(dict(p, merged_from=[p["tmp_id"]]))
    return out


def llm_dedup(disease, props, cfg, out_dir):
    version = cfg["prompts"]["dedup"]
    system, user_tpl = load_prompt("dedup", version)
    listing = "\n".join(f"[{p['tmp_id']}] ({p['polarity']:+d}, {p['category']}) {p['canonical']}" for p in props)
    user = user_tpl.format(disease=disease, propositions=listing)
    s = slug(disease)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    text, meta = call_llm(system, user, cfg["llm"])
    (out_dir / f"{s}.prompt.txt").write_text(f"### SYSTEM\n{system}\n\n### USER\n{user}", encoding="utf-8")
    (out_dir / f"{s}.response.txt").write_text(text, encoding="utf-8")
    meta.update({"disease": disease, "prompt": f"dedup_{version}", "requested_model": cfg["llm"]["model"],
                 "temperature": cfg["llm"].get("temperature"), "started_utc": started})
    (out_dir / f"{s}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    groups = extract_json(text).get("groups", [])

    by_id = {p["tmp_id"]: p for p in props}
    used, merged, rejected = set(), [], []
    for g in groups:
        mem = [m for m in g.get("members", []) if m in by_id and m not in used]
        rep = g.get("representative")
        if len(mem) < 2 or rep not in mem or len({by_id[m]["polarity"] for m in mem}) != 1:
            rejected.append(g)
            continue
        used.update(mem)
        merged.append(merge_group([by_id[m] for m in mem], by_id[rep]))
    if rejected:
        print(f"   {disease}: bỏ qua {len(rejected)} nhóm không hợp lệ")
    keep = [p for p in props if p["tmp_id"] not in used]
    # Giữ thứ tự xuất hiện trong guideline
    order = {p["tmp_id"]: i for i, p in enumerate(props)}
    return sorted(keep + merged, key=lambda p: order[p["tmp_id"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="chỉ gộp trùng chữ, không gọi LLM")
    args = ap.parse_args()
    cfg = load_config()
    units = read_jsonl(cfg["_out"] / "units.jsonl")
    units_by_id = {u["unit_id"]: u for u in units}
    diseases = list(dict.fromkeys(u["disease"] for u in units))
    out_dir = cfg["_out"] / "dedup"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in diseases:
        path = cfg["_out"] / "extract" / f"{slug(d)}.json"
        if not path.exists():
            print(f"skip {d}: chưa có kết quả trích")
            continue
        props = collect(json.load(open(path, encoding="utf-8")), units_by_id)
        n0 = len(props)
        props = exact_dedup(props)
        n1 = len(props)
        if not args.no_llm and len(props) > 1:
            props = llm_dedup(d, props, cfg, out_dir)
        print(f"{d:<35} {n0:>4} -> {n1:>4} (trùng chữ) -> {len(props):>4} (sau gộp)")
        for i, p in enumerate(props, start=1):
            rows.append({
                "proposition_id": f"{slug(d)}_P{i:03d}",
                "disease_id": d,
                "canonicalDescription": p["canonical"],
                "polarity": p["polarity"],
                "category": p["category"],
                "sourceUnitIds": p["unit_ids"],
                "sourceExcerpt": [units_by_id[u]["text"] for u in p["unit_ids"]],
                "mergedFrom": p["merged_from"],
            })
    write_jsonl(cfg["_out"] / "props_dedup.jsonl", rows)
    print(f"Tổng: {len(rows)} mệnh đề -> {cfg['_out'] / 'props_dedup.jsonl'}")


if __name__ == "__main__":
    main()

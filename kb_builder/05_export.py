"""Bước 5: xuất bảng duyệt cho người, và chốt KB sau khi duyệt.

  python 05_export.py review      -> outputs/review/review_propositions.csv
                                     outputs/review/review_excluded_units.csv
  (mở bằng Excel/Google Sheets, điền cột action ..., lưu lại dạng CSV UTF-8, giữ nguyên tên file)
  python 05_export.py finalize    -> outputs/final/kb_propositions.json   (file KB dùng cho mô hình)
                                     outputs/final/kb_stats.md, kb_stats.json
                                     outputs/final/review_log.csv

Cột action:
  review_propositions.csv  : để trống = giữ | edit = sửa theo new_* | delete = bỏ
  review_excluded_units.csv: để trống = giữ việc loại | restore = khôi phục thành mệnh đề theo restore_*
"""
import argparse
import csv
import json
from collections import Counter, defaultdict

from checks import check_proposition, format_flags
from common import (CATEGORIES, clean_markdown, load_config, load_guidelines, norm_text, read_jsonl, resolve,
                    slug)

PROP_COLS = ["proposition_id", "disease_id", "category", "polarity", "canonicalDescription", "sourceExcerpt",
             "sourceUnitIds", "flags", "action", "new_canonical", "new_polarity", "new_category", "note"]
EXCL_COLS = ["unit_id", "disease", "section", "subsection", "text", "exclude_reason",
             "action", "restore_canonical", "restore_polarity", "restore_category", "note"]


def review(cfg, synonyms):
    props = read_jsonl(cfg["_out"] / "props_dedup.jsonl")
    units = read_jsonl(cfg["_out"] / "units.jsonl")
    out = cfg["_out"] / "review"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "review_propositions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PROP_COLS)
        w.writeheader()
        for p in props:
            flags = check_proposition(p["canonicalDescription"], p["polarity"], p["category"], p["disease_id"], synonyms)
            w.writerow({"proposition_id": p["proposition_id"], "disease_id": p["disease_id"],
                        "category": p["category"], "polarity": p["polarity"],
                        "canonicalDescription": p["canonicalDescription"],
                        "sourceExcerpt": " || ".join(p["sourceExcerpt"]),
                        "sourceUnitIds": ";".join(p["sourceUnitIds"]), "flags": format_flags(flags)})
    unit_by_id = {u["unit_id"]: u for u in units}
    with open(out / "review_excluded_units.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EXCL_COLS)
        w.writeheader()
        for d in dict.fromkeys(u["disease"] for u in units):
            path = cfg["_out"] / "extract" / f"{slug(d)}.json"
            if not path.exists():
                continue
            for u in json.load(open(path, encoding="utf-8"))["units"]:
                if u["decision"] == "EXCLUDE":
                    src = unit_by_id[u["unit_id"]]
                    w.writerow({"unit_id": u["unit_id"], "disease": d, "section": src["section"],
                                "subsection": src["subsection"], "text": src["text"],
                                "exclude_reason": u["exclude_reason"]})
    print(f"Đã xuất bảng duyệt vào {out}")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def apply_review(cfg):
    units = {u["unit_id"]: u for u in read_jsonl(cfg["_out"] / "units.jsonl")}
    props = {p["proposition_id"]: p for p in read_jsonl(cfg["_out"] / "props_dedup.jsonl")}
    log, final = [], []
    for r in _read_csv(cfg["_out"] / "review" / "review_propositions.csv"):
        p = dict(props[r["proposition_id"]])
        action = (r.get("action") or "").strip().lower()
        if action == "delete":
            log.append({"type": "delete", "id": r["proposition_id"], "before": p["canonicalDescription"],
                        "after": "", "note": r.get("note", "")})
            continue
        if action == "edit":
            before = f"{p['canonicalDescription']} ({p['polarity']:+d}, {p['category']})"
            if r["new_canonical"].strip():
                p["canonicalDescription"] = r["new_canonical"].strip()
            if r["new_polarity"].strip():
                p["polarity"] = int(r["new_polarity"])
            if r["new_category"].strip():
                p["category"] = r["new_category"].strip()
            log.append({"type": "edit", "id": r["proposition_id"], "before": before,
                        "after": f"{p['canonicalDescription']} ({p['polarity']:+d}, {p['category']})",
                        "note": r.get("note", "")})
        elif action:
            raise SystemExit(f"action không hợp lệ: {action!r} ({r['proposition_id']})")
        final.append(p)
    for r in _read_csv(cfg["_out"] / "review" / "review_excluded_units.csv"):
        if (r.get("action") or "").strip().lower() != "restore":
            continue
        p = {"disease_id": r["disease"], "canonicalDescription": r["restore_canonical"].strip(),
             "polarity": int(r["restore_polarity"]), "category": r["restore_category"].strip(),
             "sourceUnitIds": [r["unit_id"]], "sourceExcerpt": [units[r["unit_id"]]["text"]]}
        final.append(p)
        log.append({"type": "restore", "id": r["unit_id"], "before": f"EXCLUDED({r['exclude_reason']})",
                    "after": f"{p['canonicalDescription']} ({p['polarity']:+d}, {p['category']})",
                    "note": r.get("note", "")})
    return final, log


def finalize(cfg, synonyms, partial=False):
    out = cfg["_out"] / "final"
    out.mkdir(parents=True, exist_ok=True)
    final, log = apply_review(cfg)

    # Quy tắc 3-6: kiểm tra lại sau duyệt
    errors = []
    for p in final:
        for lvl, code, detail in check_proposition(p["canonicalDescription"], p["polarity"], p["category"],
                                                   p["disease_id"], synonyms):
            if lvl == "ERROR":
                errors.append(f"{p['disease_id']}: {code} {detail} | {p['canonicalDescription']}")
    # Quy tắc 7: sourceExcerpt phải nằm nguyên văn trong guideline gốc
    guide_text = {g["query"]: clean_markdown(g["guideline_1_content"]) for g in load_guidelines(cfg)}
    for p in final:
        for ex in p["sourceExcerpt"]:
            if ex not in guide_text[p["disease_id"]]:
                errors.append(f"{p['disease_id']}: EXCERPT_NOT_IN_GUIDELINE | {ex[:80]}")
    # Quy tắc 8: không còn mệnh đề trùng chữ trong cùng bệnh
    seen = Counter((p["disease_id"], norm_text(p["canonicalDescription"]), p["polarity"]) for p in final)
    errors += [f"{d}: DUPLICATE | {c}" for (d, c, _), n in seen.items() if n > 1]
    if errors:
        print("\n".join(errors[:50]))
        raise SystemExit(f"Còn {len(errors)} lỗi. Sửa trong bảng duyệt rồi chạy lại finalize.")

    labels = list(csv.reader(open(resolve(cfg, "label_csv"), encoding="utf-8-sig")))[0][2:]
    by_d = defaultdict(list)
    for p in final:
        by_d[p["disease_id"]].append(p)
    if partial:
        labels = [l for l in labels if by_d.get(l)]
    else:
        missing = [l for l in labels if not by_d.get(l)]
        unknown = sorted(set(by_d) - set(labels))
        if missing or unknown:
            raise SystemExit(f"Không khớp nhãn: bệnh chưa có mệnh đề={missing}, bệnh lạ={unknown}")

    kb = []
    for d in labels:
        props = sorted(by_d[d], key=lambda p: CATEGORIES.index(p["category"]))  # sort ổn định: giữ thứ tự trong guideline
        kb.append({
            "disease_id": d,
            "sourceReference": cfg["source_reference"].format(disease=d),
            "propositions": [{
                "proposition_id": f"{slug(d)}_P{i:03d}",
                "category": p["category"],
                "canonicalDescription": p["canonicalDescription"],
                "polarity": p["polarity"],
                "sourceExcerpt": p["sourceExcerpt"],
            } for i, p in enumerate(props, start=1)],
        })
    with open(out / "kb_propositions.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    with open(out / "review_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["type", "id", "before", "after", "note"])
        w.writeheader()
        w.writerows(log)
    write_stats(cfg, kb, log, out)
    print(f"KB: {sum(len(k['propositions']) for k in kb)} mệnh đề / {len(kb)} bệnh -> {out / 'kb_propositions.json'}")


def write_stats(cfg, kb, log, out):
    reasons, kept, total = Counter(), 0, 0
    for k in kb:
        for u in json.load(open(cfg["_out"] / "extract" / f"{slug(k['disease_id'])}.json", encoding="utf-8"))["units"]:
            total += 1
            if u["decision"] == "KEEP":
                kept += 1
            else:
                reasons[u["exclude_reason"]] += 1
    allp = [p for k in kb for p in k["propositions"]]
    counts = sorted(len(k["propositions"]) for k in kb)
    owners = defaultdict(set)
    for k in kb:
        for p in k["propositions"]:
            owners[norm_text(p["canonicalDescription"])].add(k["disease_id"])
    stats = {
        "n_diseases": len(kb),
        "n_source_units": total, "n_units_kept": kept, "units_excluded_by_reason": dict(reasons),
        "n_propositions": len(allp),
        "propositions_per_disease": {"min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1]},
        "n_negative_polarity": sum(p["polarity"] == -1 for p in allp),
        "category_counts": {c: sum(p["category"] == c for p in allp) for c in CATEGORIES},
        "n_unique_canonical": len(owners),
        "n_canonical_shared_by_2plus_diseases": sum(len(v) > 1 for v in owners.values()),
        "human_review": dict(Counter(l["type"] for l in log)),
    }
    json.dump(stats, open(out / "kb_stats.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    lines = ["# Thống kê KB", "",
             f"- Số bệnh: {stats['n_diseases']}",
             f"- Câu nguồn: {total} (giữ {kept}, bỏ {total - kept}; lý do bỏ: {dict(reasons)})",
             f"- Tổng mệnh đề: {len(allp)}; mỗi bệnh min/trung vị/max = "
             f"{counts[0]}/{counts[len(counts) // 2]}/{counts[-1]}",
             f"- Theo nhóm: {stats['category_counts']}",
             f"- Polarity -1: {stats['n_negative_polarity']}",
             f"- Mô tả khác nhau: {len(owners)}; dùng chung ở ≥2 bệnh: {stats['n_canonical_shared_by_2plus_diseases']}",
             f"- Duyệt của người: {stats['human_review']}", "",
             "| Bệnh | Mệnh đề | " + " | ".join(CATEGORIES) + " | Polarity -1 |",
             "|---|---|" + "---|" * len(CATEGORIES) + "---|"]
    for k in kb:
        ps = k["propositions"]
        lines.append(f"| {k['disease_id']} | {len(ps)} | " + " | ".join(str(sum(p['category'] == c for p in ps))
                                                                     for c in CATEGORIES)
                     + f" | {sum(p['polarity'] == -1 for p in ps)} |")
    (out / "kb_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["review", "finalize"])
    ap.add_argument("--partial", action="store_true", help="chạy thử khi chưa đủ 47 bệnh (bỏ kiểm tra khớp nhãn)")
    args = ap.parse_args()
    cfg = load_config()
    synonyms = json.load(open(resolve(cfg, "synonyms_path"), encoding="utf-8"))
    if args.mode == "review":
        review(cfg, synonyms)
    else:
        finalize(cfg, synonyms, args.partial)


if __name__ == "__main__":
    main()

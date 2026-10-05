"""Bước 3: kiểm tra tự động kết quả trích mệnh đề (không dùng LLM).

Đầu ra: outputs/check/flags.csv và tóm tắt in ra màn hình.
Các lỗi ERROR nên được xử lý (chạy lại bệnh đó với 02_extract.py --force --only ...,
hoặc sửa ở bước duyệt của người) trước khi chốt KB.
"""
import csv
import json
from collections import Counter

from checks import check_proposition
from common import load_config, read_jsonl, resolve, slug


def main():
    cfg = load_config()
    synonyms = json.load(open(resolve(cfg, "synonyms_path"), encoding="utf-8"))
    units = read_jsonl(cfg["_out"] / "units.jsonl")
    diseases = list(dict.fromkeys(u["disease"] for u in units))
    out_dir = cfg["_out"] / "check"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, level_count, code_count, reasons = [], Counter(), Counter(), Counter()
    missing = []
    for d in diseases:
        path = cfg["_out"] / "extract" / f"{slug(d)}.json"
        if not path.exists():
            missing.append(d)
            continue
        data = json.load(open(path, encoding="utf-8"))
        for u in data["units"]:
            if u["decision"] == "EXCLUDE":
                reasons[u["exclude_reason"]] += 1
            for k, p in enumerate(u["propositions"]):
                for lvl, code, detail in check_proposition(p["canonical"], p["polarity"], p["category"], d, synonyms):
                    level_count[lvl] += 1
                    code_count[f"{lvl}:{code}"] += 1
                    rows.append({"disease": d, "unit_id": u["unit_id"], "prop_index": k, "level": lvl,
                                 "code": code, "detail": detail, "canonical": p["canonical"],
                                 "polarity": p["polarity"]})

    with open(out_dir / "flags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["disease", "unit_id", "prop_index", "level", "code", "detail",
                                          "canonical", "polarity"])
        w.writeheader()
        w.writerows(rows)

    if missing:
        print(f"Chưa có kết quả trích cho {len(missing)} bệnh: {missing}")
    print("Đơn vị bị loại theo lý do:", dict(reasons))
    print("Số cờ theo mức:", dict(level_count))
    for k, v in sorted(code_count.items()):
        print(f"  {k}: {v}")
    print(f"Chi tiết: {out_dir / 'flags.csv'}")


if __name__ == "__main__":
    main()

"""Chạy pipeline KB cho một tập bệnh, từ segment đến xuất CSV review.

Bệnh chỉ được chọn một lần sẽ được thêm vào kết quả hiện có. Bệnh được chọn lại
sẽ chạy lại, ghi đè kết quả trích/xử lý và bị xóa trạng thái duyệt cũ.
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _slug(disease):
    from common import slug
    return slug(disease)


def completed_diseases(cfg, diseases):
    extract_dir = cfg["_out"] / "extract"
    return {d for d in diseases if (extract_dir / f"{_slug(d)}.json").exists()}


def snapshot_extract_mtimes(cfg, diseases):
    extract_dir = cfg["_out"] / "extract"
    return {
        disease: (extract_dir / f"{_slug(disease)}.json").stat().st_mtime
        for disease in diseases
        if (extract_dir / f"{_slug(disease)}.json").exists()
    }


def updated_diseases(cfg, before, diseases):
    extract_dir = cfg["_out"] / "extract"
    return [
        disease for disease in diseases
        if (path := extract_dir / f"{_slug(disease)}.json").exists()
        and path.stat().st_mtime > before.get(disease, 0)
    ]


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print("\n$", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def select_diseases(args, diseases, completed):
    if args.only:
        unknown = sorted(set(args.only) - set(diseases))
        if unknown:
            raise SystemExit(f"Không có bệnh trong guideline: {unknown}")
        return list(dict.fromkeys(args.only))

    candidates = [d for d in diseases if d not in completed]
    if args.random > len(candidates):
        raise SystemExit(
            f"Chỉ còn {len(candidates)} bệnh chưa xử lý; dùng --only để chạy lại bệnh đã có kết quả."
        )
    return random.Random(args.seed).sample(candidates, args.random)


def main():
    parser = argparse.ArgumentParser(
        description="Chạy segment, extract, check, dedup và review cho một tập bệnh."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--only", nargs="+", help="tên bệnh chính xác cần chạy/rerun")
    selection.add_argument("--random", type=int, metavar="N", help="chọn ngẫu nhiên N bệnh chưa xử lý")
    parser.add_argument("--seed", type=int, help="seed để tái lập lựa chọn ngẫu nhiên")
    parser.add_argument("--no-llm-dedup", action="store_true", help="chỉ gộp trùng chữ, không gọi LLM ở bước dedup")
    args = parser.parse_args()
    if args.random is not None and args.random <= 0:
        parser.error("--random phải lớn hơn 0")

    from common import load_config, load_guidelines

    cfg = load_config()
    diseases = [g["query"] for g in load_guidelines(cfg)]
    completed = completed_diseases(cfg, diseases)
    selected = select_diseases(args, diseases, completed)
    rerun = [d for d in selected if d in completed]

    print("Bệnh được chạy:", ", ".join(selected))
    if rerun:
        print("Sẽ ghi đè và yêu cầu duyệt lại:", ", ".join(rerun))

    run("01_segment.py")
    before_extract = snapshot_extract_mtimes(cfg, selected)
    run("02_extract.py", "--force", "--only", *selected)
    refreshed = updated_diseases(cfg, before_extract, selected)
    if not refreshed:
        raise SystemExit("Không có kết quả trích mới hợp lệ; review hiện có không bị thay đổi.")
    run("03_check.py")
    dedup_args = ["--no-llm"] if args.no_llm_dedup else []
    run("04_dedup.py", *dedup_args)
    run("05_export.py", "review", "--refreshed-diseases", *refreshed)
    print("\nĐã tạo/cập nhật CSV review. Sau khi duyệt, chạy:")
    print("python 05_export.py finalize --partial")


if __name__ == "__main__":
    main()

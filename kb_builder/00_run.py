"""Chạy pipeline KB cho một tập bệnh, từ segment đến xuất CSV review.

Mặc định chỉ chọn bệnh chưa có kết quả trích. Dùng --force để chọn và ghi đè
cả bệnh đã có kết quả; trạng thái duyệt cũ của bệnh đó sẽ bị xóa.

Dùng --dedup để chỉ chạy lại bước dedup cho bệnh được chọn, không extract lại.
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
        requested = list(dict.fromkeys(args.only))
        return requested if args.force or args.dedup else [d for d in requested if d not in completed]

    candidates = diseases if args.force or args.dedup else [d for d in diseases if d not in completed]
    if args.random > len(candidates):
        scope = "trong guideline" if args.force or args.dedup else "chưa xử lý"
        raise SystemExit(
            f"Chỉ có {len(candidates)} bệnh {scope}; giảm --random hoặc dùng --force."
        )
    return random.Random(args.seed).sample(candidates, args.random)


def main():
    parser = argparse.ArgumentParser(
        description="Chạy segment, extract, check, dedup và review cho một tập bệnh."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--only", nargs="+", help="tên bệnh chính xác cần chạy")
    selection.add_argument("--random", type=int, metavar="N", help="chọn ngẫu nhiên N bệnh chưa xử lý (trừ khi dùng --force)")
    parser.add_argument("--seed", type=int, help="seed để tái lập lựa chọn ngẫu nhiên")
    parser.add_argument(
        "--force",
        action="store_true",
        help="cho phép chọn và ghi đè bệnh đã có kết quả trích",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="chỉ chạy lại bước dedup cho bệnh được chọn, kể cả bệnh đã có kết quả trích",
    )
    parser.add_argument("--no-llm-dedup", action="store_true", help="chỉ gộp trùng chữ, không gọi LLM ở bước dedup")
    args = parser.parse_args()
    if args.random is not None and args.random <= 0:
        parser.error("--random phải lớn hơn 0")

    from common import load_config, load_guidelines

    cfg = load_config()
    diseases = [g["query"] for g in load_guidelines(cfg)]
    completed = completed_diseases(cfg, diseases)
    selected = select_diseases(args, diseases, completed)
    if not selected:
        raise SystemExit("Không có bệnh chưa xử lý được chọn. Dùng --force để extract lại hoặc --dedup để dedup lại bệnh đã có.")
    rerun = [d for d in selected if args.force and d in completed]

    print("Bệnh được chạy:", ", ".join(selected))
    if args.dedup:
        print("Chế độ dedup: không chạy segment, extract hoặc check.")
        dedup_args = ["--only", *selected]
        if args.no_llm_dedup:
            dedup_args.insert(0, "--no-llm")
        run("04_dedup.py", *dedup_args)
        run("05_export.py", "review", "--refreshed-diseases", *selected)
        print("\nĐã cập nhật CSV review cho các bệnh được dedup. Sau khi duyệt, chạy:")
        print("python 05_export.py finalize --partial")
        return
    if rerun:
        print("Sẽ ghi đè và yêu cầu duyệt lại:", ", ".join(rerun))

    run("01_segment.py")
    before_extract = snapshot_extract_mtimes(cfg, selected)
    extract_args = ["--force"] if args.force else []
    run("02_extract.py", *extract_args, "--only", *selected)
    refreshed = updated_diseases(cfg, before_extract, selected)
    if not refreshed:
        raise SystemExit("Không có kết quả trích mới hợp lệ; review hiện có không bị thay đổi.")
    run("03_check.py")
    dedup_args = ["--no-llm"] if args.no_llm_dedup else []
    run("04_dedup.py", *dedup_args, "--only", *refreshed)
    run("05_export.py", "review", "--refreshed-diseases", *refreshed)
    print("\nĐã tạo/cập nhật CSV review. Sau khi duyệt, chạy:")
    print("python 05_export.py finalize --partial")


if __name__ == "__main__":
    main()

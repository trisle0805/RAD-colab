"""Hàm dùng chung cho các bước của kb_builder."""
import json
import re
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

EXCLUDE_REASONS = {
    "MANAGEMENT",       # điều trị, xử trí, theo dõi, phòng ngừa
    "PROGNOSIS",        # tiên lượng, sống còn, nguy cơ tái phát / chuyển ác tính
    "EPIDEMIOLOGY",     # số liệu dịch tễ quần thể (tỉ lệ mắc, "most common ...")
    "OTHER_DISEASE",    # mô tả bệnh khác (chẩn đoán phân biệt, thực thể khác cùng tên)
    "NOT_DIAGNOSTIC",   # không mô tả dấu hiệu: câu về quy trình chẩn đoán, câu chung chung
}

# 5 nhóm mệnh đề, theo thứ tự hiển thị trong bảng kiểm
CATEGORIES = ["morphology", "location", "symptom", "history", "lab"]


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_out"] = (ROOT / cfg["output_dir"]).resolve()
    cfg["_out"].mkdir(parents=True, exist_ok=True)
    return cfg


def resolve(cfg, key):
    return (ROOT / cfg[key]).resolve()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_guidelines(cfg):
    rows = read_jsonl(resolve(cfg, "guideline_path"))
    names = [r["query"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("Guideline file has duplicated disease names.")
    return rows


def slug(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def norm_text(s):
    """Chuẩn hóa để so khớp: chữ thường, gạch nối thành khoảng trắng, gộp khoảng trắng."""
    s = s.lower().replace("-", " ").replace("’", "'")
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_json(text):
    """Lấy object JSON đầu tiên trong phản hồi LLM (bỏ code fence nếu có)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start:end + 1])


def load_prompt(name, version):
    base = ROOT / "prompts"
    system = (base / f"{name}_{version}.system.md").read_text(encoding="utf-8")
    user = (base / f"{name}_{version}.user.md").read_text(encoding="utf-8")
    return system, user


def clean_markdown(s):
    """Bỏ ký hiệu markdown (** và *in nghiêng*), giữ ký tự * nằm giữa chữ (vd. HLA-C*06:02)."""
    s = s.replace("**", "")
    s = re.sub(r"(?<![\w*])\*(\S[^*]*?)\*(?![\w*])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()

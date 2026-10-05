"""Bước 1: tách mỗi guideline thành các đơn vị nguồn (không dùng LLM).

- Dòng tiêu đề (#) và dòng nhãn mục (vd. "**Disease Description:**", "- **Cutaneous SCC:**")
  không phải đơn vị, chỉ dùng làm ngữ cảnh (section / subsection).
- Mỗi dòng bullet có nội dung = 1 đơn vị.
- Dòng đoạn văn (không phải bullet) được tách thành từng câu, mỗi câu = 1 đơn vị.
- Văn bản đơn vị giữ nguyên chữ của nguồn, chỉ bỏ ký hiệu markdown (** và dấu đầu dòng).

Đầu ra: outputs/units.jsonl
"""
import re

from common import clean_markdown, load_config, load_guidelines, slug, write_jsonl

LIST_RE = re.compile(r"^(\s*)(?:[-*•]|\d+\.)\s+(.*)$")
BOLD_WHOLE_RE = re.compile(r"^\*\*(.+?)\*\*\s*(:?)\s*$")

# Bảo vệ các dấu chấm không phải cuối câu khi tách câu.
ABBREV = [r"e\.g\.", r"i\.e\.", r"vs\.", r"et al\.", r"approx\.", r"Dr\.", r"St\.", r"No\."]


def clean(s):
    return clean_markdown(s)


def split_sentences(text):
    protected = text
    for i, a in enumerate(ABBREV):
        protected = re.sub(a, lambda m, i=i: m.group(0).replace(".", f"<DOT{i}>"), protected)
    # Tên chi viết tắt kiểu "P. acnes", "B. burgdorferi"
    protected = re.sub(r"\b([A-Z])\.\s(?=[a-z])", r"\1<DOTG> ", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", protected)
    out = []
    for p in parts:
        p = re.sub(r"<DOT\d+>", ".", p).replace("<DOTG>", ".")
        if p.strip():
            out.append(p.strip())
    return out


def header_label(body):
    """Trả về nhãn nếu dòng chỉ là nhãn mục in đậm, ngược lại None."""
    m = BOLD_WHOLE_RE.match(body.strip())
    if not m:
        return None
    label, colon = m.group(1).strip(), m.group(2)
    if colon or label.endswith(":"):
        return label.rstrip(":").strip()
    return None


def segment(disease_idx, disease, text):
    units, section, sub_stack = [], "Preamble", []  # sub_stack: [(indent, label)]
    n = 0
    for raw in text.split("\n"):
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            section, sub_stack = "Title", []
            continue
        m = LIST_RE.match(raw)
        if m:
            indent, body = len(m.group(1)), m.group(2)
            sub_stack = [(i, l) for i, l in sub_stack if i < indent]
            label = header_label(body)
            if label is not None:
                sub_stack.append((indent, label))
                continue
            sentences, is_list = [clean(body)], True
        else:
            label = header_label(raw)
            if label is not None:
                section, sub_stack = label, []
                continue
            sentences, is_list = split_sentences(clean(raw)), False
        for sent in sentences:
            n += 1
            units.append({
                "unit_id": f"{disease_idx:02d}-{n:03d}",
                "disease": disease,
                "disease_slug": slug(disease),
                "section": clean(section),
                "subsection": " > ".join(clean(l) for _, l in sub_stack),
                "text": sent,
                "is_list_item": is_list,
            })
    return units


def main():
    cfg = load_config()
    guides = load_guidelines(cfg)
    all_units = []
    for idx, g in enumerate(guides, start=1):
        units = segment(idx, g["query"], g["guideline_1_content"])
        if not units:
            raise ValueError(f"No units for {g['query']}")
        all_units.extend(units)
        print(f"{idx:02d} {g['query']:<35} {len(units):>3} units")
    write_jsonl(cfg["_out"] / "units.jsonl", all_units)
    print(f"Total: {len(all_units)} units -> {cfg['_out'] / 'units.jsonl'}")


if __name__ == "__main__":
    main()

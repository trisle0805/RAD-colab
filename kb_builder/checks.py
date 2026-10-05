"""Các phép kiểm tra tự động trên mệnh đề (dùng chung cho 03_check.py và 05_export.py)."""
import re

from common import CATEGORIES, norm_text

NEGATION = ["no", "not", "without", "absent", "absence", "lack", "lacks", "lacking", "negative",
            "normal", "spared", "sparing", "unremarkable", "none", "never"]
HEDGES = ["may", "can", "often", "usually", "typically", "commonly", "frequently", "sometimes",
          "occasionally", "rarely", "possibly", "might", "could"]


def _has_word(text, words):
    toks = set(norm_text(text).split())
    return sorted(w for w in words if w in toks)


def _name_hits(canonical, names, abbreviations):
    hits = []
    n = f" {norm_text(canonical)} "
    for name in names:
        if f" {norm_text(name)} " in n:
            hits.append(name)
    for ab in abbreviations:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(ab)}(?![A-Za-z0-9])", canonical):
            hits.append(ab)
    return hits


def check_proposition(canonical, polarity, category, disease, synonyms):
    """Trả về danh sách (mức, mã, chi tiết). Mức: ERROR | WARN | INFO."""
    flags = []
    c = canonical.strip()
    if not c:
        return [("ERROR", "EMPTY", "")]
    neg = _has_word(c, NEGATION)
    if neg:
        flags.append(("ERROR", "NEGATION_WORD", ",".join(neg)))
    hed = _has_word(c, HEDGES)
    if hed:
        flags.append(("ERROR", "HEDGE_WORD", ",".join(hed)))
    own = synonyms[disease]
    hits = _name_hits(c, own["names"], own["abbreviations"])
    if hits:
        flags.append(("ERROR", "LEAK_OWN_NAME", ",".join(hits)))
    other_hits = []
    for d, s in synonyms.items():
        if d == disease:
            continue
        h = [x for x in _name_hits(c, s["names"], s["abbreviations"]) if x not in hits]
        if h:
            other_hits.append(f"{d}:{'/'.join(h)}")
    if other_hits:
        flags.append(("WARN", "OTHER_LABEL_NAME", "; ".join(other_hits)))
    n_words = len(c.split())
    if n_words > 20:
        flags.append(("ERROR", "LENGTH", str(n_words)))
    elif n_words > 15:
        flags.append(("WARN", "LENGTH", str(n_words)))
    if c.endswith("."):
        flags.append(("WARN", "TRAILING_PERIOD", ""))
    if polarity not in (1, -1):
        flags.append(("ERROR", "BAD_POLARITY", str(polarity)))
    elif polarity == -1:
        flags.append(("INFO", "NEGATIVE_POLARITY", ""))
    if category not in CATEGORIES:
        flags.append(("ERROR", "BAD_CATEGORY", str(category)))
    return flags


def format_flags(flags):
    return " | ".join(f"{lvl}:{code}" + (f"({d})" if d else "") for lvl, code, d in flags)

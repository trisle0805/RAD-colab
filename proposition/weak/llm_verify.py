"""OpenAI-backed, cacheable A2b verification for WS_03.

The client is imported lazily so a dry run needs neither the package nor a
credential.  A real run is deliberately gated by the caller's explicit
``approve`` argument.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .labels import quote_is_supported


@dataclass(frozen=True)
class LlmCell:
    row: int
    column: int
    case_id: str
    proposition_id: str
    group: str
    sub_hypotheses: tuple[str, ...]
    masked_caption: str


def prompt_version(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl_cache(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    source = Path(path)
    if not source.is_file():
        return cache
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
                key = (str(entry["case_id"]), str(entry["proposition_id"]), str(entry["prompt_version"]))
                cache[key] = entry
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return cache


def dry_run_report(cells: Iterable[LlmCell]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cell in cells:
        grouped[(cell.group, "all")].add(cell.case_id)
    # The caller additionally saves per-split reports; this generic report is
    # intentionally only based on selected cells and contains no secrets.
    return [
        {"group": group, "split": split, "n_cases": len(case_ids), "estimated_calls": len(case_ids)}
        for (group, split), case_ids in sorted(grouped.items())
    ]


def _render_prompt(template: str, caption: str, cells: list[LlmCell]) -> str:
    signs = "\n".join(
        f"{index + 1}. {cell.proposition_id} | {' / '.join(cell.sub_hypotheses)}"
        for index, cell in enumerate(cells)
    )
    return template.replace("{masked_caption}", caption).replace("{numbered list: id | sub-hypotheses joined by \" / \"}", signs)


def _openai_client(settings: Mapping[str, Any]):
    if settings.get("provider") != "openai":
        raise ValueError("This implementation currently supports provider=openai only")
    key_env = str(settings.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"Missing {key_env}. Set it as a Colab Secret/environment variable.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package before a real LLM run.") from error
    base_url = os.environ.get(str(settings.get("base_url_env", "OPENAI_BASE_URL"))) or None
    return OpenAI(api_key=api_key, base_url=base_url)


def verify_cells(
    cells: Iterable[LlmCell], *, settings: Mapping[str, Any], prompt_path: str | Path,
    cache_path: str | Path, approve: bool,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Verify selected cells, one API call per case, with durable JSONL cache."""

    if not approve:
        raise RuntimeError("Real LLM calls require explicit approval (--approve-llm).")
    model = settings.get("model") or os.environ.get(str(settings.get("model_env", "OPENAI_MODEL")))
    if not model:
        raise ValueError("Set llm_verify.model or the OPENAI_MODEL environment variable before real OpenAI calls")
    template = Path(prompt_path).read_text(encoding="utf-8")
    version = prompt_version(prompt_path)
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache = load_jsonl_cache(cache_file)
    by_case: dict[str, list[LlmCell]] = defaultdict(list)
    for cell in cells:
        by_case[cell.case_id].append(cell)
    client = _openai_client(settings)
    results: dict[tuple[int, int], dict[str, Any]] = {}

    with cache_file.open("a", encoding="utf-8") as cache_handle:
        for case_id, case_cells in by_case.items():
            unresolved = [cell for cell in case_cells if (cell.case_id, cell.proposition_id, version) not in cache]
            answers: dict[str, Mapping[str, Any]] = {}
            for cell in case_cells:
                cached = cache.get((cell.case_id, cell.proposition_id, version))
                if cached:
                    answers[cell.proposition_id] = cached.get("result", {})
            if unresolved:
                prompt = _render_prompt(template, unresolved[0].masked_caption, unresolved)
                response = None
                for attempt in range(int(settings.get("max_retries", 3))):
                    try:
                        response = client.chat.completions.create(
                            model=str(model),
                            temperature=float(settings.get("temperature", 0)),
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"},
                        )
                        parsed = json.loads(response.choices[0].message.content or "{}")
                        raw_results = {str(item.get("id")): item for item in parsed.get("results", []) if isinstance(item, dict)}
                        break
                    except Exception:
                        if attempt + 1 == int(settings.get("max_retries", 3)):
                            raise
                        time.sleep(2 ** attempt)
                for cell in unresolved:
                    result = raw_results.get(cell.proposition_id, {})
                    entry = {
                        "case_id": cell.case_id, "proposition_id": cell.proposition_id,
                        "prompt_version": version, "group": cell.group, "model": model,
                        "result": result,
                    }
                    cache_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                    cache[(cell.case_id, cell.proposition_id, version)] = entry
                    answers[cell.proposition_id] = result
            for cell in case_cells:
                answer = answers.get(cell.proposition_id, {})
                verdict = str(answer.get("verdict", "NOT_MENTIONED")).upper()
                quote = str(answer.get("quote", ""))
                invalid = verdict in {"PRESENT", "ABSENT"} and not quote_is_supported(quote, cell.masked_caption)
                if verdict not in {"PRESENT", "ABSENT", "NOT_MENTIONED"} or invalid:
                    verdict = "NOT_MENTIONED"
                results[(cell.row, cell.column)] = {"verdict": verdict, "quote_invalid": int(invalid), "group": cell.group}
    return results
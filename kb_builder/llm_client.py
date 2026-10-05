"""Gọi LLM qua Anthropic hoặc OpenAI. API key đọc từ biến môi trường hoặc .env.
ANTHROPIC_API_KEY hoặc OPENAI_API_KEY."""
import os
import time
from pathlib import Path


def load_dotenv():
    """Nạp biến môi trường chưa được đặt từ kb_builder/.env, nếu file tồn tại."""
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ensure_env_loaded():
    load_dotenv()


def call_llm(system, user, llm_cfg):
    _ensure_env_loaded()
    provider = llm_cfg["provider"]
    if not llm_cfg.get("model"):
        raise ValueError("config.yaml: llm.model chưa được điền.")
    last_err = None
    for attempt in range(1, llm_cfg.get("max_retries", 3) + 1):
        try:
            if provider == "anthropic":
                return _anthropic(system, user, llm_cfg)
            if provider == "openai":
                return _openai(system, user, llm_cfg)
            raise ValueError(f"Unknown provider: {provider}")
        except ValueError:
            raise
        except Exception as e:  # lỗi mạng / rate limit: thử lại
            last_err = e
            time.sleep(10 * attempt)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def _anthropic(system, user, cfg):
    import anthropic
    client = anthropic.Anthropic(timeout=cfg.get("request_timeout", 600))
    kwargs = dict(
        model=cfg["model"],
        max_tokens=cfg.get("max_output_tokens", 16000),
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    resp = client.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    meta = {
        "provider": "anthropic",
        "model": resp.model,
        "stop_reason": resp.stop_reason,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
    }
    return text, meta


def _openai(system, user, cfg):
    import openai
    client_kwargs = {"timeout": cfg.get("request_timeout", 600)}
    if cfg.get("base_url"):
        client_kwargs["base_url"] = cfg["base_url"]
    client = openai.OpenAI(**client_kwargs)
    kwargs = dict(
        model=cfg["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=cfg.get("max_output_tokens", 16000),
    )
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    meta = {
        "provider": "openai",
        "model": resp.model,
        "stop_reason": choice.finish_reason,
        "usage": {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens},
    }
    return choice.message.content, meta

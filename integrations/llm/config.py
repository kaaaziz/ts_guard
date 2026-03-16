from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool
    provider: str

    groq_api_key: str | None
    groq_base_url: str
    model: str

    timeout_seconds: float
    temperature: float
    max_completion_tokens: int
    max_history_messages: int

    enable_fallback: bool

    @property
    def provider_normalized(self) -> str:
        return (self.provider or "rule_based").strip().lower()


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    return LLMSettings(
        enabled=_env_bool("TSGUARD_LLM_ENABLED", True),
        provider=os.getenv("TSGUARD_LLM_PROVIDER", "groq").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_base_url=os.getenv(
            "TSGUARD_GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        ).rstrip("/"),
        model=os.getenv("TSGUARD_LLM_MODEL", "llama-3.1-8b-instant").strip(),
        timeout_seconds=_env_float("TSGUARD_LLM_TIMEOUT_SECONDS", 20.0),
        temperature=_env_float("TSGUARD_LLM_TEMPERATURE", 0.2),
        max_completion_tokens=_env_int("TSGUARD_LLM_MAX_COMPLETION_TOKENS", 350),
        max_history_messages=_env_int("TSGUARD_LLM_MAX_HISTORY_MESSAGES", 8),
        enable_fallback=_env_bool("TSGUARD_LLM_ENABLE_FALLBACK", True),
    )
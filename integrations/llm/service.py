from __future__ import annotations

from typing import List, Dict, Any

import streamlit as st

from .config import get_llm_settings
from .context import (
    SystemSnapshot,
    build_llm_context_block,
    build_llm_system_prompt,
    generate_rule_based_reply,
)
from .providers import GroqProvider, LLMProviderError


def _history_to_provider_messages(
    chat_history: List[Dict[str, Any]],
    max_history_messages: int,
) -> List[Dict[str, str]]:
    cleaned: list[dict[str, str]] = []

    if not isinstance(chat_history, list):
        return cleaned

    tail = chat_history[-max_history_messages:] if max_history_messages > 0 else []

    for item in tail:
        role = str(item.get("role", "")).strip().lower()
        text = str(item.get("text", "")).strip()
        if role in {"user", "assistant"} and text:
            cleaned.append({"role": role, "content": text})

    return cleaned


def _store_runtime_debug(*, backend: str, error: str | None) -> None:
    try:
        st.session_state["_llm_backend_used"] = backend
        st.session_state["_llm_last_error"] = error
    except Exception:
        pass


def generate_chat_reply(
    *,
    user_text: str,
    snapshot: SystemSnapshot,
    chat_history: List[Dict[str, Any]],
) -> str:
    settings = get_llm_settings()

    provider_name = settings.provider_normalized
    history_messages = _history_to_provider_messages(
        chat_history=chat_history,
        max_history_messages=settings.max_history_messages,
    )

    system_prompt = build_llm_system_prompt()
    live_context = build_llm_context_block(user_text=user_text, snap=snapshot)

    if settings.enabled and provider_name == "groq" and settings.groq_api_key:
        provider = GroqProvider(settings=settings)
        try:
            reply = provider.generate(
                system_prompt=system_prompt,
                history_messages=history_messages,
                user_text=user_text,
                live_context=live_context,
            )
            _store_runtime_debug(backend="groq", error=None)
            return reply
        except LLMProviderError as exc:
            if not settings.enable_fallback:
                _store_runtime_debug(backend="groq_error", error=str(exc))
                return (
                    "The Groq assistant is currently unavailable and fallback is disabled.\n\n"
                    f"Error: {exc}"
                )

            fallback = generate_rule_based_reply(user_text, snapshot)
            _store_runtime_debug(backend="rule_based_fallback", error=str(exc))
            return fallback

    fallback_reason = None
    if not settings.enabled:
        fallback_reason = "LLM disabled by configuration."
    elif provider_name != "groq":
        fallback_reason = f"Provider '{settings.provider}' is not enabled in this build."
    elif not settings.groq_api_key:
        fallback_reason = "GROQ_API_KEY is missing."

    fallback = generate_rule_based_reply(user_text, snapshot)
    _store_runtime_debug(backend="rule_based", error=fallback_reason)
    return fallback
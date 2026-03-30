from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import requests

from .config import LLMSettings


class LLMProviderError(RuntimeError):
    pass


def _coerce_content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(x.strip() for x in chunks if str(x).strip()).strip()

    return str(content).strip()


@dataclass
class GroqProvider:
    settings: LLMSettings

    def generate(
        self,
        *,
        system_prompt: str,
        history_messages: List[Dict[str, str]],
        user_text: str,
        live_context: str,
    ) -> str:
        if not self.settings.groq_api_key:
            raise LLMProviderError("GROQ_API_KEY is missing.")

        endpoint = f"{self.settings.groq_base_url}/chat/completions"

        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        for msg in history_messages:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        final_user_content = (
            f"{live_context}\n\n"
            f"User question:\n{user_text.strip()}"
        )
        messages.append({"role": "user", "content": final_user_content})

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_completion_tokens": self.settings.max_completion_tokens,
            "n": 1,
        }

        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise LLMProviderError(f"Groq request failed: {exc}") from exc

        if response.status_code >= 400:
            try:
                err_json = response.json()
            except Exception:
                err_json = response.text
            raise LLMProviderError(
                f"Groq HTTP {response.status_code}: {err_json}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderError("Groq returned a non-JSON response.") from exc

        try:
            choice0 = data["choices"][0]
            message = choice0["message"]
            content = _coerce_content_to_text(message.get("content", ""))
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected Groq response shape: {data}"
            ) from exc

        if not content:
            raise LLMProviderError("Groq returned an empty message.")

        return content
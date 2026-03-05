from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base_provider import LLMProvider, ProviderResponse


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks).strip()

    return str(content)


class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    def __init__(
        self,
        model: str,
        default_temperature: float = 0.2,
        default_max_tokens: int = 2000,
        timeout_seconds: int = 120,
    ) -> None:
        super().__init__(model=model, default_temperature=default_temperature, default_max_tokens=default_max_tokens)
        self.timeout_seconds = int(timeout_seconds)

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterProvider")
        self.api_key = api_key

        self.referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        self.title = os.getenv("OPENROUTER_APP_TITLE", "agent-harness").strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title

        request = urllib.request.Request(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

        latency = time.perf_counter() - started

        raw_data = json.loads(raw_bytes.decode("utf-8"))
        choices = raw_data.get("choices", [])
        message: dict[str, Any] = {}
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {}) or {}

        usage = raw_data.get("usage", {}) or {}
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = prompt_tokens + completion_tokens

        return ProviderResponse(
            text=_extract_message_text(message),
            tool_calls=list(message.get("tool_calls", []) or []),
            token_usage=int(total_tokens or 0),
            latency=latency,
            model=str(raw_data.get("model") or self.model),
            raw=raw_data,
        )

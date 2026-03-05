from __future__ import annotations

from typing import Any

from .base_provider import LLMProvider, ProviderResponse
from .openrouter_provider import OpenRouterProvider


class MockProvider(LLMProvider):
    """Offline-safe provider used for local tests and development."""
    name = "mock"

    def __init__(self, model: str = "mock/default", default_temperature: float = 0.2, default_max_tokens: int = 2000) -> None:
        super().__init__(
            model=model,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        prompt_words = len((system_prompt + " " + user_prompt).split())
        return ProviderResponse(
            text="{}",
            tool_calls=[],
            token_usage=max(1, int(prompt_words / 0.75)),
            latency=0.0,
            model=self.model,
            raw={"provider": "mock", "temperature": temperature, "max_tokens": max_tokens},
        )


class ProviderFactory:
    @staticmethod
    def create(
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> LLMProvider:
        normalized = model_name.strip()
        if not normalized:
            normalized = "mock/default"

        if normalized.startswith("openrouter/"):
            upstream_model = normalized.split("openrouter/", 1)[1]
            if not upstream_model:
                raise ValueError("OpenRouter model must be provided after 'openrouter/' prefix")
            return OpenRouterProvider(
                model=upstream_model,
                default_temperature=temperature,
                default_max_tokens=max_tokens,
            )

        if normalized.startswith("mock/") or normalized == "mock":
            return MockProvider(
                model=normalized,
                default_temperature=temperature,
                default_max_tokens=max_tokens,
            )

        raise ValueError(
            f"Unsupported provider model '{model_name}'. Use 'openrouter/<model>' or 'mock/<name>'."
        )

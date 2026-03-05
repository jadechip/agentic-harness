from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderResponse:
    text: str
    tool_calls: list[dict[str, Any]]
    token_usage: int
    latency: float
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    name: str = "provider"

    def __init__(
        self,
        model: str,
        default_temperature: float = 0.2,
        default_max_tokens: int = 2000,
    ) -> None:
        self.model = model
        self.default_temperature = float(default_temperature)
        self.default_max_tokens = int(default_max_tokens)

    @property
    def provider_name(self) -> str:
        return self.name

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        raise NotImplementedError

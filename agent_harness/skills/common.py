from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SkillOutput:
    content: dict[str, Any]
    prompt: str
    tool_calls: list[dict[str, Any]]
    token_usage: int


def estimate_tokens(text: str) -> int:
    # Rough estimate: 1 token ~= 0.75 words for English-like text.
    words = max(1, len(text.split()))
    return int(words / 0.75)

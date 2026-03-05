from __future__ import annotations

import json
from typing import Any

from agent_harness.providers.base_provider import LLMProvider


def _bounded(score: float) -> float:
    return max(0.0, min(1.0, score))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None

    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                segment = stripped[start : idx + 1]
                try:
                    data = json.loads(segment)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    return None
    return None


class HeuristicLLMJudge:
    """Heuristic proxy for model-based judging.

    This keeps the runtime deterministic for local execution while preserving
    the contract shape of an LLM judge.
    """

    def evaluate(self, criteria: list[str], content: dict[str, Any]) -> dict[str, float]:
        text_blob = " ".join(str(v) for v in content.values() if isinstance(v, (str, int, float)))
        word_count = len(text_blob.split())
        structural_count = sum(1 for value in content.values() if isinstance(value, (list, dict)))

        richness = _bounded((word_count / 120.0) * 0.7 + (structural_count / 8.0) * 0.3)
        baseline = 0.45 + 0.45 * richness

        return {criterion: _bounded(baseline) for criterion in criteria}


class ProviderLLMJudge:
    """Model-backed judge using the configured LLM provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def evaluate(self, criteria: list[str], content: dict[str, Any]) -> dict[str, float]:
        system_prompt = (
            "You are an evaluation judge. Return only JSON. "
            "Output format: {\"scores\": {\"criterion\": 0.0-1.0}}"
        )
        user_prompt = (
            f"Criteria: {json.dumps(criteria)}\n"
            f"Artifact content: {json.dumps(content, sort_keys=True, default=str)}\n"
            "Score each criterion between 0 and 1."
        )

        response = self.provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=[],
            temperature=0.0,
            max_tokens=800,
        )
        parsed = _extract_json_object(response.text) or {}
        raw_scores = parsed.get("scores", {}) if isinstance(parsed.get("scores", {}), dict) else {}

        scores: dict[str, float] = {}
        for criterion in criteria:
            value = raw_scores.get(criterion, 0.5)
            try:
                scores[criterion] = _bounded(float(value))
            except (TypeError, ValueError):
                scores[criterion] = 0.5
        return scores

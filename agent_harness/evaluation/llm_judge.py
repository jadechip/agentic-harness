from __future__ import annotations

from typing import Any


def _bounded(score: float) -> float:
    return max(0.0, min(1.0, score))


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

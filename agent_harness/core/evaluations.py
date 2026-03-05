from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    name: str
    criteria: list[str]
    weights: dict[str, float]
    pass_threshold: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    contract_name: str
    score: float
    criterion_scores: dict[str, float]
    passed: bool
    feedback: list[str]


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    if not scores:
        return 0.0

    if not weights:
        values = list(scores.values())
        return sum(values) / len(values)

    weighted_total = 0.0
    weight_sum = 0.0
    for criterion, score in scores.items():
        weight = float(weights.get(criterion, 1.0))
        weighted_total += score * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0
    return weighted_total / weight_sum

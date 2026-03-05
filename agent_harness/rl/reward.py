from __future__ import annotations


def compute_reward(
    artifact_quality_score: float,
    token_usage: int,
    latency_seconds: float,
    token_cost_weight: float = 0.00002,
    latency_penalty_weight: float = 0.02,
) -> float:
    token_cost = token_usage * token_cost_weight
    latency_penalty = latency_seconds * latency_penalty_weight
    return artifact_quality_score - token_cost - latency_penalty

from __future__ import annotations


def compute_reward(
    artifact_quality_score: float,
    token_usage: int,
    latency_seconds: float,
    verification_pass_rate: float = 0.0,
    spec_compliance: float = 0.0,
    token_cost_weight: float = 0.00002,
    latency_penalty_weight: float = 0.02,
    verification_bonus_weight: float = 0.35,
    verification_failure_penalty_weight: float = 0.35,
    spec_compliance_bonus_weight: float = 0.2,
) -> float:
    token_cost = token_usage * token_cost_weight
    latency_penalty = latency_seconds * latency_penalty_weight

    verification_bonus = verification_pass_rate * verification_bonus_weight
    verification_failure_penalty = (1.0 - verification_pass_rate) * verification_failure_penalty_weight
    spec_bonus = spec_compliance * spec_compliance_bonus_weight

    return (
        artifact_quality_score
        + verification_bonus
        + spec_bonus
        - verification_failure_penalty
        - token_cost
        - latency_penalty
    )

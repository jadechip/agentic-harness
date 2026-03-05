from __future__ import annotations

from agent_harness.core.evaluations import EvaluationContract


def get_contract() -> EvaluationContract:
    return EvaluationContract(
        name="plan_quality",
        criteria=["phase_completeness", "spec_alignment", "risk_coverage"],
        weights={"phase_completeness": 4.0, "spec_alignment": 3.0, "risk_coverage": 3.0},
        pass_threshold=0.75,
    )

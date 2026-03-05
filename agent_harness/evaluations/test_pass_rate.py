from __future__ import annotations

from agent_harness.core.evaluations import EvaluationContract


def get_contract() -> EvaluationContract:
    return EvaluationContract(
        name="test_pass_rate",
        criteria=["test_pass_rate", "issue_reporting"],
        weights={"test_pass_rate": 8.0, "issue_reporting": 2.0},
        pass_threshold=0.8,
    )

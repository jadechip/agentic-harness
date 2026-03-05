from __future__ import annotations

from agent_harness.core.evaluations import EvaluationContract


def get_contract() -> EvaluationContract:
    return EvaluationContract(
        name="repo_map_quality",
        criteria=[
            "module_coverage",
            "entrypoint_detection",
            "dependency_accuracy",
            "architecture_summary_quality",
        ],
        weights={
            "module_coverage": 3.0,
            "entrypoint_detection": 2.0,
            "dependency_accuracy": 3.0,
            "architecture_summary_quality": 2.0,
        },
        pass_threshold=0.8,
    )

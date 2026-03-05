from __future__ import annotations

from agent_harness.core.evaluations import EvaluationContract


def get_contract() -> EvaluationContract:
    return EvaluationContract(
        name="patch_quality",
        criteria=["file_targeting", "patch_completeness", "repo_change_applied", "spec_alignment"],
        weights={
            "file_targeting": 2.0,
            "patch_completeness": 3.0,
            "repo_change_applied": 3.0,
            "spec_alignment": 2.0,
        },
        pass_threshold=0.75,
    )

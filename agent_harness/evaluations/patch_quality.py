from __future__ import annotations

from agent_harness.core.evaluations import EvaluationContract


def get_contract() -> EvaluationContract:
    return EvaluationContract(
        name="patch_quality",
        criteria=["file_targeting", "patch_completeness", "spec_alignment"],
        weights={"file_targeting": 3.0, "patch_completeness": 4.0, "spec_alignment": 3.0},
        pass_threshold=0.75,
    )

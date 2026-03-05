from __future__ import annotations

from dataclasses import dataclass

from agent_harness.core.evaluations import EvaluationContract, EvaluationResult, weighted_score

from .deterministic_checks import run_deterministic_checks
from .llm_judge import HeuristicLLMJudge


@dataclass(slots=True)
class EvaluationEngine:
    deterministic_weight: float = 0.7
    llm_weight: float = 0.3
    llm_judge: HeuristicLLMJudge = HeuristicLLMJudge()

    def evaluate(self, contract: EvaluationContract, artifact_content: dict) -> EvaluationResult:
        deterministic_scores = run_deterministic_checks(contract.criteria, artifact_content)
        llm_scores = self.llm_judge.evaluate(contract.criteria, artifact_content)

        merged_scores: dict[str, float] = {}
        for criterion in contract.criteria:
            d_score = deterministic_scores.get(criterion, 0.5)
            l_score = llm_scores.get(criterion, 0.5)
            merged_scores[criterion] = (
                self.deterministic_weight * d_score + self.llm_weight * l_score
            )

        final_score = weighted_score(merged_scores, contract.weights)
        passed = final_score >= contract.pass_threshold

        feedback: list[str] = []
        for criterion, score in merged_scores.items():
            if score < 0.75:
                feedback.append(f"Improve criterion '{criterion}' (current={score:.2f})")

        return EvaluationResult(
            contract_name=contract.name,
            score=final_score,
            criterion_scores=merged_scores,
            passed=passed,
            feedback=feedback,
        )

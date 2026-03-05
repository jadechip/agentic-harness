from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_harness.core.evaluations import EvaluationContract
from agent_harness.core.tasks import Task, TaskOutcome
from agent_harness.core.traces import TraceFactory
from agent_harness.evaluation.engine import EvaluationEngine
from agent_harness.runtime.context_builder import ContextBuilder
from agent_harness.runtime.executor import AgentExecutor, ExecutionCandidate
from agent_harness.store.artifact_store import ArtifactStore
from agent_harness.store.trace_store import TraceStore


@dataclass(slots=True)
class ImprovementLoop:
    executor: AgentExecutor
    context_builder: ContextBuilder
    evaluation_engine: EvaluationEngine
    artifact_store: ArtifactStore
    trace_store: TraceStore
    evaluation_contracts: dict[str, EvaluationContract]

    def run_task(self, run_id: str, task: Task, user_request: str, repo_path: Path) -> TaskOutcome:
        feedback: list[str] = []
        previous_artifact: dict | None = None
        contract = self._contract_for(task.evaluation)

        for attempt in range(1, task.retry_limit + 1):
            context = self.context_builder.build(
                run_id=run_id,
                task=task,
                user_request=user_request,
                feedback=feedback,
                prior_artifact=previous_artifact,
            )
            candidates = self.executor.run_task(
                task=task,
                context=context,
                user_request=user_request,
                repo_path=repo_path,
                feedback=feedback,
            )

            ranked = self._evaluate_and_store(run_id, task, attempt, candidates, contract)
            best_candidate, best_evaluation = ranked[0]

            self.artifact_store.mark_selected(run_id, task.name, best_candidate.artifact.id)

            if best_evaluation.passed:
                return TaskOutcome(
                    task_name=task.name,
                    artifact_id=best_candidate.artifact.id,
                    score=best_evaluation.score,
                    passed=True,
                    attempts=attempt,
                    feedback=best_evaluation.feedback,
                )

            feedback = best_evaluation.feedback
            previous_artifact = best_candidate.artifact.content

        return TaskOutcome(
            task_name=task.name,
            artifact_id=None,
            score=0.0,
            passed=False,
            attempts=task.retry_limit,
            feedback=feedback,
        )

    def _evaluate_and_store(
        self,
        run_id: str,
        task: Task,
        attempt: int,
        candidates: list[ExecutionCandidate],
        contract: EvaluationContract,
    ) -> list[tuple[ExecutionCandidate, object]]:
        ranked: list[tuple[ExecutionCandidate, object]] = []

        for candidate in candidates:
            evaluation = self.evaluation_engine.evaluate(contract=contract, artifact_content=candidate.artifact.content)
            self.artifact_store.save(
                run_id=run_id,
                task_name=task.name,
                artifact=candidate.artifact,
                attempt=attempt,
                candidate=candidate.sample_index,
                selected=False,
            )

            trace = TraceFactory.create(
                run_id=run_id,
                task_name=task.name,
                skill_name=task.skill,
                prompt=candidate.prompt,
                tool_calls=candidate.tool_calls,
                artifact_id=candidate.artifact.id,
                evaluation_score=evaluation.score,
                token_usage=candidate.token_usage,
                latency=candidate.latency,
            )
            self.trace_store.save(
                trace=trace,
                attempt=attempt,
                candidate=candidate.sample_index,
                passed=evaluation.passed,
                feedback=evaluation.feedback,
            )

            ranked.append((candidate, evaluation))

        ranked.sort(key=lambda pair: pair[1].score, reverse=True)
        return ranked

    def _contract_for(self, name: str) -> EvaluationContract:
        if name in self.evaluation_contracts:
            return self.evaluation_contracts[name]

        # Default fallback keeps unknown evaluations executable.
        return EvaluationContract(
            name=name,
            criteria=["non_empty"],
            weights={"non_empty": 1.0},
            pass_threshold=0.7,
        )

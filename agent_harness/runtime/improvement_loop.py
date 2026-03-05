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
    artifact_schemas: dict[str, dict] | None = None

    def run_task(self, run_id: str, task: Task, user_request: str, repo_path: Path) -> TaskOutcome:
        feedback: list[str] = []
        previous_artifact: dict | None = None
        previous_artifact_id: str | None = None
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
                artifact_schema=self._artifact_schema_for(task.produces),
                parent_artifact_id=previous_artifact_id,
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
            previous_artifact_id = best_candidate.artifact.id

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

            ranked.append((candidate, evaluation))

        ranked.sort(key=lambda pair: pair[1].score, reverse=True)
        selected_artifact_id = ranked[0][0].artifact.id

        for candidate, evaluation in ranked:
            trace = TraceFactory.create(
                run_id=run_id,
                task_name=task.name,
                skill_name=task.skill,
                provider_name=candidate.provider_name,
                model_name=candidate.model,
                temperature=candidate.temperature,
                max_tokens=candidate.max_tokens,
                evaluator_mode=self.evaluation_engine.mode,
                prompt=candidate.prompt,
                tool_calls=candidate.tool_calls,
                artifact_id=candidate.artifact.id,
                candidate_index=candidate.sample_index,
                selected=(candidate.artifact.id == selected_artifact_id),
                evaluation_score=evaluation.score,
                evaluation_breakdown=evaluation.criterion_scores,
                token_usage=candidate.token_usage,
                latency=candidate.latency,
            )
            self.trace_store.save(
                trace=trace,
                attempt=attempt,
                passed=evaluation.passed,
                feedback=evaluation.feedback,
            )

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

    def _artifact_schema_for(self, artifact_type: str) -> dict | None:
        if isinstance(self.artifact_schemas, dict):
            return self.artifact_schemas.get(artifact_type)
        return None

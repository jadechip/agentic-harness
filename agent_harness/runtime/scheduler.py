from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agent_harness.core.harness import HarnessConfig
from agent_harness.core.tasks import TaskOutcome, TaskGraph
from agent_harness.runtime.improvement_loop import ImprovementLoop
from agent_harness.store.artifact_store import ArtifactStore
from agent_harness.store.run_store import RunStore
from agent_harness.store.trace_store import TraceStore


@dataclass(slots=True)
class HarnessRunResult:
    run_id: str
    success: bool
    task_outcomes: list[TaskOutcome]
    feedback_cycles: int


@dataclass(slots=True)
class HarnessScheduler:
    harness: HarnessConfig
    run_store: RunStore
    artifact_store: ArtifactStore
    trace_store: TraceStore
    improvement_loop: ImprovementLoop
    provider_name: str
    model_name: str
    temperature: float
    max_tokens: int
    evaluation_mode: str = "deterministic"
    enforce_real_execution: bool = False

    def run(self, user_request: str, repo_path: str | Path) -> HarnessRunResult:
        repo = Path(repo_path).resolve()
        run_id = self.run_store.start_run(
            harness_name=self.harness.name,
            user_request=user_request,
            repo_path=str(repo),
            metadata={
                "settings": self.harness.settings,
                "provider": {
                    "name": self.provider_name,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                "evaluation_mode": self.evaluation_mode,
            },
        )

        main_order = self._main_execution_order()
        if not main_order:
            self.run_store.finish_run(run_id=run_id, status="failed", metadata={"error": "empty execution order"})
            return HarnessRunResult(run_id=run_id, success=False, task_outcomes=[], feedback_cycles=0)

        task_outcomes: list[TaskOutcome] = []
        max_feedback_cycles = int(self.harness.settings.get("max_feedback_cycles", 3))
        max_runtime_seconds = int(self.harness.settings.get("max_runtime_seconds", 1800))

        main_index_map = {name: idx for idx, name in enumerate(main_order)}
        feedback_cycles = 0
        started_at = time.monotonic()
        success = True

        current_task_name: str | None = main_order[0]
        while current_task_name is not None:
            if (time.monotonic() - started_at) > max_runtime_seconds:
                success = False
                break

            task = self.harness.tasks[current_task_name]
            outcome = self.improvement_loop.run_task(
                run_id=run_id,
                task=task,
                user_request=user_request,
                repo_path=repo,
            )
            task_outcomes.append(outcome)

            if outcome.passed:
                success_target = self.harness.feedback_target(current_task_name, "success")
                if success_target is not None:
                    current_task_name = success_target
                    continue

                if current_task_name in main_index_map:
                    next_idx = main_index_map[current_task_name] + 1
                    current_task_name = main_order[next_idx] if next_idx < len(main_order) else None
                else:
                    # Feedback-only task without explicit success route.
                    success = False
                    break
                continue

            failure_target = self.harness.feedback_target(current_task_name, "failure")
            if failure_target is None:
                success = False
                break

            if feedback_cycles >= max_feedback_cycles:
                success = False
                break
            feedback_cycles += 1
            current_task_name = failure_target

        execution_contract = self._execution_contract_status(run_id=run_id)
        if self.enforce_real_execution and (
            not execution_contract["repo_changed"] or not execution_contract["verification_ran"]
        ):
            success = False

        self.run_store.finish_run(
            run_id=run_id,
            status="success" if success else "failed",
            metadata={
                "feedback_cycles": feedback_cycles,
                "completed_tasks": [out.task_name for out in task_outcomes if out.passed],
                "provider": {
                    "name": self.provider_name,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                "evaluation_mode": self.evaluation_mode,
                "execution_contract": execution_contract,
            },
        )

        return HarnessRunResult(
            run_id=run_id,
            success=success,
            task_outcomes=task_outcomes,
            feedback_cycles=feedback_cycles,
        )

    def _main_execution_order(self) -> list[str]:
        if not self.harness.flow:
            return self.harness.execution_order()

        flow_nodes: set[str] = set()
        for source, target in self.harness.flow:
            flow_nodes.add(source)
            flow_nodes.add(target)

        if not flow_nodes:
            return self.harness.execution_order()

        sub_tasks = {name: self.harness.tasks[name] for name in flow_nodes if name in self.harness.tasks}
        sub_flow = [edge for edge in self.harness.flow if edge[0] in sub_tasks and edge[1] in sub_tasks]
        return TaskGraph(tasks=sub_tasks, flow=sub_flow).execution_order()

    def _execution_contract_status(self, run_id: str) -> dict[str, bool]:
        selected_artifacts = self.artifact_store.list_selected_for_run(run_id)

        repo_changed = False
        for artifact in selected_artifacts:
            if artifact.type != "ImplementationPatch":
                continue
            content = artifact.content or {}
            files_changed = content.get("files_changed", [])
            if isinstance(files_changed, list) and len(files_changed) > 0:
                repo_changed = True
                break

        verification_ran = False
        for artifact in selected_artifacts:
            if artifact.type != "QAReport":
                continue
            content = artifact.content or {}
            if bool(content.get("verification_ran", False)):
                verification_ran = True
                break

        return {
            "repo_changed": repo_changed,
            "verification_ran": verification_ran,
        }

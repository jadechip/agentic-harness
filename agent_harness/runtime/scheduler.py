from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agent_harness.core.harness import HarnessConfig
from agent_harness.core.tasks import TaskOutcome
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

    def run(self, user_request: str, repo_path: str | Path) -> HarnessRunResult:
        repo = Path(repo_path).resolve()
        run_id = self.run_store.start_run(
            harness_name=self.harness.name,
            user_request=user_request,
            repo_path=str(repo),
            metadata={"settings": self.harness.settings},
        )

        order = self.harness.execution_order()
        task_outcomes: list[TaskOutcome] = []
        max_feedback_cycles = int(self.harness.settings.get("max_feedback_cycles", 3))
        max_runtime_seconds = int(self.harness.settings.get("max_runtime_seconds", 1800))

        feedback_cycles = 0
        index = 0
        started_at = time.monotonic()
        success = True

        while index < len(order):
            if (time.monotonic() - started_at) > max_runtime_seconds:
                success = False
                break

            task_name = order[index]
            task = self.harness.tasks[task_name]
            outcome = self.improvement_loop.run_task(
                run_id=run_id,
                task=task,
                user_request=user_request,
                repo_path=repo,
            )
            task_outcomes.append(outcome)

            if outcome.passed:
                index += 1
                continue

            target = self.harness.feedback_target(task_name, "failure")
            if target is None:
                success = False
                break

            if feedback_cycles >= max_feedback_cycles:
                success = False
                break
            feedback_cycles += 1

            try:
                index = order.index(target)
            except ValueError:
                success = False
                break

        if index < len(order):
            success = False

        self.run_store.finish_run(
            run_id=run_id,
            status="success" if success else "failed",
            metadata={
                "feedback_cycles": feedback_cycles,
                "completed_tasks": [out.task_name for out in task_outcomes if out.passed],
            },
        )

        return HarnessRunResult(
            run_id=run_id,
            success=success,
            task_outcomes=task_outcomes,
            feedback_cycles=feedback_cycles,
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evaluations import EvaluationContract
from .skills import Skill
from .tasks import Task, TaskGraph


@dataclass(frozen=True, slots=True)
class FeedbackRule:
    source_task: str
    event: str
    target_task: str


@dataclass(slots=True)
class HarnessConfig:
    name: str
    tasks: dict[str, Task]
    flow: list[tuple[str, str]]
    feedback: list[FeedbackRule] = field(default_factory=list)
    skills: dict[str, Skill] = field(default_factory=dict)
    artifact_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    evaluations: dict[str, EvaluationContract] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def task_graph(self) -> TaskGraph:
        return TaskGraph(tasks=self.tasks, flow=self.flow)

    def execution_order(self) -> list[str]:
        return self.task_graph().execution_order()

    def feedback_target(self, task_name: str, event: str = "failure") -> str | None:
        for rule in self.feedback:
            if rule.source_task == task_name and rule.event == event:
                return rule.target_task
        return None

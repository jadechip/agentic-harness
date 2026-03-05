from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    skill: str
    context: list[str]
    produces: str
    evaluation: str
    samples: int = 1
    retry_limit: int = 3


@dataclass(slots=True)
class TaskGraph:
    tasks: dict[str, Task]
    flow: list[tuple[str, str]]

    def execution_order(self) -> list[str]:
        indegree: dict[str, int] = {name: 0 for name in self.tasks}
        outgoing: dict[str, set[str]] = {name: set() for name in self.tasks}

        for source, target in self.flow:
            if source not in self.tasks or target not in self.tasks:
                raise ValueError(f"Invalid edge {source}->{target}: unknown task name")
            if target not in outgoing[source]:
                outgoing[source].add(target)
                indegree[target] += 1

        queue = [name for name, degree in indegree.items() if degree == 0]
        ordered: list[str] = []

        while queue:
            task_name = queue.pop(0)
            ordered.append(task_name)
            for target in outgoing[task_name]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)

        if len(ordered) != len(self.tasks):
            raise ValueError("Flow graph contains a cycle")

        return ordered


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    task_name: str
    artifact_id: str | None
    score: float | None
    passed: bool
    attempts: int
    feedback: list[str] = field(default_factory=list)

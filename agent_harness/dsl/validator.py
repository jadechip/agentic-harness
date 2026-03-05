from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from agent_harness.core.harness import HarnessConfig


class HarnessValidationError(ValueError):
    pass


def _reachable_tasks(task_names: set[str], edges: list[tuple[str, str]]) -> set[str]:
    if not task_names:
        return set()

    incoming: dict[str, int] = {name: 0 for name in task_names}
    outgoing: dict[str, set[str]] = {name: set() for name in task_names}

    for source, target in edges:
        if source in task_names and target in task_names:
            outgoing[source].add(target)
            incoming[target] += 1

    roots = sorted(name for name, degree in incoming.items() if degree == 0)
    queue = deque(roots)
    visited: set[str] = set()
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for nxt in sorted(outgoing[current]):
            queue.append(nxt)
    return visited


def validate_harness(
    harness: HarnessConfig,
    available_skills: Iterable[str] | None = None,
    available_evaluations: Iterable[str] | None = None,
    available_artifact_schemas: Iterable[str] | None = None,
    require_provider_settings: bool = False,
) -> None:
    if not harness.name:
        raise HarnessValidationError("Harness name is required")

    if not harness.tasks:
        raise HarnessValidationError("Harness must define at least one task")

    if len(harness.tasks) > 1 and not harness.flow and not bool(harness.settings.get("allow_implicit_order", False)):
        raise HarnessValidationError(
            "Harness flow cannot be empty for multi-task harnesses unless settings.allow_implicit_order=true"
        )

    try:
        harness.execution_order()
    except ValueError as exc:
        raise HarnessValidationError(str(exc)) from exc

    task_names = set(harness.tasks)

    for source, target in harness.flow:
        if source not in task_names or target not in task_names:
            raise HarnessValidationError(f"Flow edge {source}->{target} references unknown task")

    for rule in harness.feedback:
        if rule.source_task not in task_names:
            raise HarnessValidationError(f"Feedback source task not found: {rule.source_task}")
        if rule.target_task not in task_names:
            raise HarnessValidationError(f"Feedback target task not found: {rule.target_task}")

    allow_unreachable = bool(harness.settings.get("allow_unreachable_tasks", False))
    if not allow_unreachable:
        flow_nodes: set[str] = set()
        for source, target in harness.flow:
            flow_nodes.add(source)
            flow_nodes.add(target)

        if harness.flow:
            not_in_flow = sorted(task_names - flow_nodes)
            if not_in_flow:
                raise HarnessValidationError(
                    "Tasks must be connected to flow DAG: " + ", ".join(not_in_flow)
                )

        reached = _reachable_tasks(flow_nodes or task_names, harness.flow)
        unreachable = sorted((flow_nodes or task_names) - reached)
        if unreachable:
            raise HarnessValidationError(
                "Unreachable tasks in flow DAG: "
                + ", ".join(unreachable)
                + ". Set settings.allow_unreachable_tasks=true to allow this."
            )

    known_skills = set(available_skills) if available_skills is not None else None
    if known_skills is not None:
        for task in harness.tasks.values():
            if task.skill not in known_skills:
                raise HarnessValidationError(f"Task '{task.name}' references unknown skill '{task.skill}'")

    known_evaluations = set(available_evaluations) if available_evaluations is not None else None
    if known_evaluations is not None:
        for task in harness.tasks.values():
            if task.evaluation not in known_evaluations:
                raise HarnessValidationError(
                    f"Task '{task.name}' references unknown evaluation '{task.evaluation}'"
                )

    known_artifacts = set(available_artifact_schemas) if available_artifact_schemas is not None else None
    if known_artifacts is not None:
        for task in harness.tasks.values():
            if task.produces not in known_artifacts:
                raise HarnessValidationError(
                    f"Task '{task.name}' references unknown artifact schema '{task.produces}'"
                )

    if require_provider_settings:
        model = str(harness.settings.get("model", "")).strip()
        if not model:
            raise HarnessValidationError(
                "Harness settings.model must be set unless mock mode is explicitly enabled"
            )

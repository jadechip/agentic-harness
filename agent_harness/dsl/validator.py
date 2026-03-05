from __future__ import annotations

from collections.abc import Iterable

from agent_harness.core.harness import HarnessConfig


class HarnessValidationError(ValueError):
    pass


def validate_harness(
    harness: HarnessConfig,
    available_skills: Iterable[str] | None = None,
    available_evaluations: Iterable[str] | None = None,
) -> None:
    if not harness.name:
        raise HarnessValidationError("Harness name is required")

    if not harness.tasks:
        raise HarnessValidationError("Harness must define at least one task")

    # Verifies all task references are valid and graph is acyclic.
    harness.execution_order()

    task_names = set(harness.tasks)

    for source, target in harness.flow:
        if source not in task_names or target not in task_names:
            raise HarnessValidationError(f"Flow edge {source}->{target} references unknown task")

    for rule in harness.feedback:
        if rule.source_task not in task_names:
            raise HarnessValidationError(f"Feedback source task not found: {rule.source_task}")
        if rule.target_task not in task_names:
            raise HarnessValidationError(f"Feedback target task not found: {rule.target_task}")

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

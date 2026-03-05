from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_harness.core.evaluations import EvaluationContract
from agent_harness.core.harness import FeedbackRule, HarnessConfig
from agent_harness.core.skills import Skill
from agent_harness.core.tasks import Task


class HarnessParserError(ValueError):
    pass


def _parse_edge(edge: str) -> tuple[str, str]:
    if "->" not in edge:
        raise HarnessParserError(f"Invalid flow edge: {edge}")
    source, target = edge.split("->", 1)
    source = source.strip()
    target = target.strip()
    if not source or not target:
        raise HarnessParserError(f"Invalid flow edge: {edge}")
    return source, target


def _parse_feedback_rule(raw: str) -> FeedbackRule:
    if "->" not in raw:
        raise HarnessParserError(f"Invalid feedback rule: {raw}")
    left, target = raw.split("->", 1)
    left = left.strip()
    target = target.strip()

    if "." in left:
        source_task, event = left.split(".", 1)
    else:
        source_task, event = left, "failure"

    if not source_task or not target:
        raise HarnessParserError(f"Invalid feedback rule: {raw}")

    return FeedbackRule(source_task=source_task.strip(), event=event.strip(), target_task=target)


def _parse_task(name: str, raw: dict[str, Any]) -> Task:
    if "skill" not in raw:
        raise HarnessParserError(f"Task '{name}' missing required field 'skill'")
    if "produces" not in raw:
        raise HarnessParserError(f"Task '{name}' missing required field 'produces'")

    evaluation = raw.get("evaluate", raw.get("evaluation"))
    if not evaluation:
        raise HarnessParserError(f"Task '{name}' missing required field 'evaluate'")

    context = raw.get("context", [])
    if context is None:
        context = []
    if not isinstance(context, list):
        raise HarnessParserError(f"Task '{name}' context must be a list")

    samples = int(raw.get("samples", 1))
    retry_limit = int(raw.get("retry_limit", raw.get("retries", 3)))

    return Task(
        name=name,
        skill=str(raw["skill"]),
        context=[str(v) for v in context],
        produces=str(raw["produces"]),
        evaluation=str(evaluation),
        samples=max(1, samples),
        retry_limit=max(1, retry_limit),
    )


def _parse_skill(name: str, raw: dict[str, Any]) -> Skill:
    return Skill(
        name=name,
        description=str(raw.get("description", "")),
        input_artifacts=[str(v) for v in raw.get("input_artifacts", [])],
        output_artifact=str(raw.get("output_artifact", "")),
        allowed_tools=[str(v) for v in raw.get("allowed_tools", [])],
    )


def _parse_evaluation_contract(name: str, raw: dict[str, Any]) -> EvaluationContract:
    criteria = [str(v) for v in raw.get("criteria", [])]
    weights = {str(k): float(v) for k, v in raw.get("weights", {}).items()}
    pass_threshold = float(raw.get("pass_threshold", 0.8))
    return EvaluationContract(name=name, criteria=criteria, weights=weights, pass_threshold=pass_threshold)


def parse_harness_yaml(path: str | Path) -> HarnessConfig:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    harness_name = data.get("harness")
    if not harness_name:
        raise HarnessParserError("Harness file missing top-level 'harness' name")

    raw_tasks = data.get("tasks") or {}
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise HarnessParserError("Harness file must define a non-empty 'tasks' mapping")

    tasks: dict[str, Task] = {}
    for task_name, raw_task in raw_tasks.items():
        if not isinstance(raw_task, dict):
            raise HarnessParserError(f"Task '{task_name}' definition must be a mapping")
        tasks[str(task_name)] = _parse_task(str(task_name), raw_task)

    flow: list[tuple[str, str]] = []
    for raw_edge in data.get("flow", []):
        flow.append(_parse_edge(str(raw_edge)))

    raw_feedback = data.get("feedback", [])
    feedback: list[FeedbackRule] = []
    if isinstance(raw_feedback, str):
        feedback.append(_parse_feedback_rule(raw_feedback))
    elif isinstance(raw_feedback, list):
        for rule in raw_feedback:
            feedback.append(_parse_feedback_rule(str(rule)))
    elif isinstance(raw_feedback, dict):
        for left, right in raw_feedback.items():
            feedback.append(_parse_feedback_rule(f"{left} -> {right}"))

    raw_skills = data.get("skills", {})
    skills: dict[str, Skill] = {}
    if isinstance(raw_skills, dict):
        for skill_name, raw_skill in raw_skills.items():
            skills[str(skill_name)] = _parse_skill(str(skill_name), raw_skill or {})

    raw_evaluations = data.get("evaluations", {})
    evaluations: dict[str, EvaluationContract] = {}
    if isinstance(raw_evaluations, dict):
        for eval_name, raw_contract in raw_evaluations.items():
            evaluations[str(eval_name)] = _parse_evaluation_contract(str(eval_name), raw_contract or {})

    artifact_schemas = data.get("artifacts", {})
    if artifact_schemas is None:
        artifact_schemas = {}

    settings = data.get("settings", {})
    if settings is None:
        settings = {}

    return HarnessConfig(
        name=str(harness_name),
        tasks=tasks,
        flow=flow,
        feedback=feedback,
        skills=skills,
        artifact_schemas={str(k): v for k, v in artifact_schemas.items()},
        evaluations=evaluations,
        settings=settings,
    )

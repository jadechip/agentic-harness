from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_harness.providers.base_provider import LLMProvider
from agent_harness.skills.common import SkillOutput, generate_structured_content
from agent_harness.tools.base_tool import ToolSandbox


def run(
    *,
    context: dict[str, Any],
    user_request: str,
    repo_path: Path,
    tools: ToolSandbox,
    provider: LLMProvider,
    sample_index: int,
    feedback: list[str] | None = None,
) -> SkillOutput:
    artifacts = context.get("artifacts", {})
    codebase_map = artifacts.get("CodebaseMap", {})
    modules = codebase_map.get("modules", []) if isinstance(codebase_map, dict) else []

    phases = [
        {
            "name": "Requirements and Scope",
            "tasks": [
                "Confirm acceptance criteria with explicit success metrics",
                "Identify impacted boundaries and interfaces",
            ],
        },
        {
            "name": "Design",
            "tasks": [
                "Design integration points and configuration strategy",
                "Define backward compatibility behavior",
            ],
        },
        {
            "name": "Implementation",
            "tasks": [
                "Implement code changes in target modules",
                "Add or update automated tests and fixtures",
            ],
        },
        {
            "name": "Verification",
            "tasks": [
                "Run test suite and smoke checks",
                "Record risks, unresolved issues, and rollback plan",
            ],
        },
    ]

    candidate_files = modules[: min(6, len(modules))]
    risks = [
        "Insufficient test coverage for changed paths",
        "Configuration drift between environments",
        "Unexpected dependency version mismatches",
    ]

    fallback_content = {
        "objective": user_request,
        "summary": (
            f"Plan for request '{user_request}' across {len(candidate_files)} likely impacted files "
            "with phased implementation and verification."
        ),
        "phases": phases,
        "candidate_files": candidate_files,
        "risks": risks,
        "test_strategy": "Prioritize integration tests around changed behavior and failure paths.",
    }

    artifact_schema = {
        "objective": "string",
        "summary": "string",
        "phases": [{"name": "string", "tasks": ["string"]}],
        "candidate_files": ["string"],
        "risks": ["string"],
        "test_strategy": "string",
    }

    llm_context = {
        "user_request": user_request,
        "sample_index": sample_index,
        "feedback": feedback or [],
        "codebase_map": codebase_map,
        "fallback_plan": fallback_content,
    }

    content, response, prompt = generate_structured_content(
        provider=provider,
        task_goal="Generate a high-quality implementation plan artifact.",
        artifact_name="PhasePlan",
        artifact_schema=artifact_schema,
        context_payload=llm_context,
        fallback_content=fallback_content,
    )

    return SkillOutput(
        content=content,
        prompt=prompt,
        tool_calls=list(response.tool_calls),
        token_usage=response.token_usage,
        model=response.model,
        latency=response.latency,
    )

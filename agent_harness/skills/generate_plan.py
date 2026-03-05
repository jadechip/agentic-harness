from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_harness.skills.common import SkillOutput, estimate_tokens
from agent_harness.tools.base_tool import ToolSandbox


def run(
    *,
    context: dict[str, Any],
    user_request: str,
    repo_path: Path,
    tools: ToolSandbox,
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
                "Identify impacted authentication and session boundaries",
            ],
        },
        {
            "name": "Design",
            "tasks": [
                "Design integration points and configuration strategy",
                "Define migration and backward compatibility behavior",
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
        "Insufficient test coverage for auth/session paths",
        "Configuration drift between environments",
        "Unexpected dependency or SDK version mismatches",
    ]

    summary = (
        f"Plan for request '{user_request}' across {len(candidate_files)} likely impacted files "
        f"with phased implementation and verification."
    )

    prompt = (
        "Create a high-quality PhasePlan artifact from available repository context.\n"
        f"User request: {user_request}\n"
        f"Candidate modules: {candidate_files}\n"
        f"Retry feedback: {feedback or []}\n"
        f"Sample index: {sample_index}"
    )

    content = {
        "objective": user_request,
        "summary": summary,
        "phases": phases,
        "candidate_files": candidate_files,
        "risks": risks,
        "test_strategy": "Prioritize integration tests around changed auth flows and failure paths.",
    }

    token_usage = estimate_tokens(prompt + " " + summary)
    return SkillOutput(content=content, prompt=prompt, tool_calls=[], token_usage=token_usage)

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
    plan = artifacts.get("PhasePlan", {})
    candidate_files = plan.get("candidate_files", []) if isinstance(plan, dict) else []

    files_to_modify = candidate_files[: min(5, len(candidate_files))]
    if not files_to_modify:
        files_to_modify = ["<discover during implementation>"]

    tool_calls: list[dict[str, Any]] = []
    if files_to_modify and files_to_modify[0] != "<discover during implementation>":
        read_result = tools.execute("file_read", {"path": files_to_modify[0], "max_chars": 1000}, allowed_tools=["file_read"])
        tool_calls.append(
            {
                "tool": "file_read",
                "input": {"path": files_to_modify[0], "max_chars": 1000},
                "success": read_result.success,
                "error": read_result.error,
            }
        )

    patch_plan = [
        f"Implement requested capability: {user_request}",
        "Update service integration boundaries and configuration handling",
        "Add/adjust tests for success and failure scenarios",
        "Document rollout and rollback considerations",
    ]

    summary = (
        f"Implementation plan sample {sample_index} targeting {len(files_to_modify)} file(s)."
    )

    prompt = (
        "Generate an ImplementationPatch artifact from planning context and feedback.\n"
        f"User request: {user_request}\n"
        f"Files to modify: {files_to_modify}\n"
        f"Retry feedback: {feedback or []}\n"
        f"Sample index: {sample_index}"
    )

    content = {
        "objective": user_request,
        "summary": summary,
        "files_to_modify": files_to_modify,
        "patch_plan": patch_plan,
        "estimated_complexity": "medium",
    }

    token_usage = estimate_tokens(prompt + " " + summary)
    return SkillOutput(content=content, prompt=prompt, tool_calls=tool_calls, token_usage=token_usage)

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
    qa_report = artifacts.get("QAReport", {})
    implementation_patch = artifacts.get("ImplementationPatch", {})

    prior_files = implementation_patch.get("files_to_modify", []) if isinstance(implementation_patch, dict) else []
    issues = qa_report.get("issues", []) if isinstance(qa_report, dict) else []

    patch_plan = [
        "Analyze failing checks and isolate root cause",
        "Apply targeted fix to highest-impact file",
        "Add regression coverage for the specific failure",
        "Re-run verification and capture residual risks",
    ]

    prompt = (
        "Generate a remediation ImplementationPatch based on QA failures.\n"
        f"User request: {user_request}\n"
        f"Known issues: {issues}\n"
        f"Retry feedback: {feedback or []}\n"
        f"Sample index: {sample_index}"
    )

    content = {
        "summary": "Remediation patch generated from verification failures",
        "files_to_modify": prior_files or ["<determine after root-cause analysis>"],
        "patch_plan": patch_plan,
        "failure_signals": issues,
        "estimated_complexity": "medium",
    }

    token_usage = estimate_tokens(prompt)
    return SkillOutput(content=content, prompt=prompt, tool_calls=[], token_usage=token_usage)

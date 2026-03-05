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
    qa_report = artifacts.get("QAReport", {})
    implementation_patch = artifacts.get("ImplementationPatch", {})

    prior_files = implementation_patch.get("files_to_modify", []) if isinstance(implementation_patch, dict) else []
    issues = qa_report.get("issues", []) if isinstance(qa_report, dict) else []

    fallback_content = {
        "objective": user_request,
        "summary": "Remediation patch generated from verification failures",
        "files_to_modify": prior_files or ["<determine after root-cause analysis>"],
        "patch_plan": [
            "Analyze failing checks and isolate root cause",
            "Apply targeted fix to highest-impact file",
            "Add regression coverage for the specific failure",
            "Re-run verification and capture residual risks",
        ],
        "failure_signals": issues,
        "estimated_complexity": "medium",
    }

    artifact_schema = {
        "objective": "string",
        "summary": "string",
        "files_to_modify": ["string"],
        "patch_plan": ["string"],
        "failure_signals": ["string"],
        "estimated_complexity": "string",
    }

    llm_context = {
        "user_request": user_request,
        "sample_index": sample_index,
        "feedback": feedback or [],
        "qa_report": qa_report,
        "previous_patch": implementation_patch,
        "fallback_patch": fallback_content,
    }

    content, response, prompt = generate_structured_content(
        provider=provider,
        task_goal="Debug verification failure and produce a remediation implementation patch.",
        artifact_name="ImplementationPatch",
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

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_harness.skills.common import SkillOutput, estimate_tokens
from agent_harness.tools.base_tool import ToolSandbox


def _derive_pass_rate(success: bool, output: str) -> tuple[int, int, float]:
    lowered = output.lower()
    if "no tests ran" in lowered:
        return 0, 0, 0.6

    if success:
        return 1, 0, 1.0

    # Coarse fallback for failed test command without parser-specific output.
    return 1, 1, 0.0


def run(
    *,
    context: dict[str, Any],
    user_request: str,
    repo_path: Path,
    tools: ToolSandbox,
    sample_index: int,
    feedback: list[str] | None = None,
) -> SkillOutput:
    tool_calls: list[dict[str, Any]] = []

    test_result = tools.execute("shell", {"command": "pytest -q"}, allowed_tools=["shell"])
    tool_calls.append(
        {
            "tool": "shell",
            "input": {"command": "pytest -q"},
            "success": test_result.success,
            "error": test_result.error,
            "output_preview": test_result.output[:2000],
        }
    )

    tests_ran, tests_failed, pass_rate = _derive_pass_rate(test_result.success, test_result.output)

    issues: list[str] = []
    if not test_result.success and test_result.error:
        issues.append(test_result.error)
    if not test_result.success and test_result.output:
        issues.append("Test command returned non-zero status")
    if "no tests ran" in test_result.output.lower():
        issues.append("No tests discovered by pytest")

    prompt = (
        "Verify behavior by running available checks and produce QAReport.\n"
        f"User request: {user_request}\n"
        f"Retry feedback: {feedback or []}\n"
        f"Sample index: {sample_index}"
    )

    content = {
        "tests_ran": tests_ran,
        "tests_failed": tests_failed,
        "pass_rate": pass_rate,
        "issues": issues,
        "recommendations": [
            "Address failing checks before merge",
            "Expand regression coverage for changed areas",
        ],
        "raw_output": test_result.output[:4000],
    }

    token_usage = estimate_tokens(prompt + " " + test_result.output)
    return SkillOutput(content=content, prompt=prompt, tool_calls=tool_calls, token_usage=token_usage)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_harness.providers.base_provider import LLMProvider
from agent_harness.skills.common import SkillOutput, generate_structured_content
from agent_harness.tools.base_tool import ToolSandbox


def _command_exists(tools: ToolSandbox, command: str) -> bool:
    result = tools.execute(
        "shell",
        {"command": f"command -v {command} >/dev/null 2>&1"},
        allowed_tools=["shell"],
    )
    return result.success


def _has_npm_test_script(repo_path: Path) -> bool:
    package_json = repo_path / "package.json"
    if not package_json.exists() or not package_json.is_file():
        return False
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
    return isinstance(scripts, dict) and isinstance(scripts.get("test"), str)


def _choose_verification_commands(tools: ToolSandbox, repo_path: Path) -> list[str]:
    if _command_exists(tools, "pytest"):
        return ["pytest -q"]

    if _command_exists(tools, "npm") and _has_npm_test_script(repo_path):
        return ["npm test --silent"]

    python_files = list(repo_path.rglob("*.py"))
    if python_files and _command_exists(tools, "python"):
        return ["python -m compileall -q ."]

    return ["ls -la"]


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
    commands = _choose_verification_commands(tools, repo_path)
    command_results: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    for command in commands:
        result = tools.execute(
            "shell",
            {"command": command},
            allowed_tools=["shell"],
        )
        record = {
            "command": command,
            "success": result.success,
            "returncode": result.metadata.get("returncode") if isinstance(result.metadata, dict) else None,
            "output": result.output[:4000],
            "error": result.error,
        }
        command_results.append(record)
        tool_calls.append(
            {
                "tool": "shell",
                "input": {"command": command},
                "success": result.success,
                "error": result.error,
                "metadata": result.metadata,
                "output_preview": result.output[:1000],
            }
        )

    verification_ran = len(command_results) > 0
    successful_commands = sum(1 for r in command_results if r.get("success"))
    pass_rate = (successful_commands / len(command_results)) if command_results else 0.0

    issues: list[str] = []
    failing_commands: list[str] = []
    for record in command_results:
        if record.get("success"):
            continue
        cmd = str(record.get("command", ""))
        failing_commands.append(cmd)
        error = str(record.get("error", "")).strip()
        if error:
            issues.append(f"{cmd}: {error}")
        else:
            issues.append(f"{cmd}: command failed")

    fallback_content = {
        "verification_ran": verification_ran,
        "commands_run": [r["command"] for r in command_results],
        "command_results": command_results,
        "tests_ran": len(command_results),
        "tests_failed": len(failing_commands),
        "pass_rate": pass_rate,
        "failing_commands": failing_commands,
        "issues": issues,
        "recommendations": [
            "Address failing verification commands before merge",
            "Expand regression coverage around changed files",
        ],
        "raw_output": "\n\n".join(
            [f"$ {r['command']}\n{r['output']}" for r in command_results]
        )[:5000],
    }

    artifact_schema = {
        "verification_ran": "boolean",
        "commands_run": ["string"],
        "command_results": [
            {
                "command": "string",
                "success": "boolean",
                "returncode": "integer",
                "output": "string",
                "error": "string",
            }
        ],
        "tests_ran": "integer",
        "tests_failed": "integer",
        "pass_rate": "float between 0 and 1",
        "failing_commands": ["string"],
        "issues": ["string"],
        "recommendations": ["string"],
        "raw_output": "string",
    }

    llm_context = {
        "user_request": user_request,
        "sample_index": sample_index,
        "feedback": feedback or [],
        "verification_commands": commands,
        "command_results": command_results,
        "fallback_report": fallback_content,
    }

    content, response, prompt = generate_structured_content(
        provider=provider,
        task_goal="Run repository verification and produce QA report with concrete failures.",
        artifact_name="QAReport",
        artifact_schema=artifact_schema,
        context_payload=llm_context,
        fallback_content=fallback_content,
    )

    # Verification facts come from actual command execution, not model interpretation.
    content["verification_ran"] = verification_ran
    content["commands_run"] = [r["command"] for r in command_results]
    content["command_results"] = command_results
    content["tests_ran"] = len(command_results)
    content["tests_failed"] = len(failing_commands)
    content["pass_rate"] = pass_rate
    content["failing_commands"] = failing_commands
    if not isinstance(content.get("issues"), list):
        content["issues"] = issues
    else:
        merged_issues = list(dict.fromkeys([*content.get("issues", []), *issues]))
        content["issues"] = merged_issues

    merged_tool_calls = tool_calls + list(response.tool_calls)
    return SkillOutput(
        content=content,
        prompt=prompt,
        tool_calls=merged_tool_calls,
        token_usage=response.token_usage,
        model=response.model,
        latency=response.latency,
    )

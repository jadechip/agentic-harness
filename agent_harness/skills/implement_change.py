from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_harness.providers.base_provider import LLMProvider
from agent_harness.skills.common import SkillOutput, generate_structured_content
from agent_harness.tools.base_tool import ToolSandbox


def _safe_repo_relpath(repo_path: Path, path_value: str) -> str:
    candidate = (repo_path / path_value).resolve()
    if repo_path not in candidate.parents and candidate != repo_path:
        raise ValueError(f"Edit path escapes repo: {path_value}")
    return str(candidate.relative_to(repo_path))


def _is_probably_binary(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        data = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\x00" in data


def _read_text(tools: ToolSandbox, rel_path: str) -> tuple[str, dict[str, Any]]:
    result = tools.execute(
        "file_read",
        {"path": rel_path, "max_chars": 1_000_000},
        allowed_tools=["file_read"],
    )
    call = {
        "tool": "file_read",
        "input": {"path": rel_path, "max_chars": 1_000_000},
        "success": result.success,
        "error": result.error,
        "metadata": result.metadata,
    }
    return (result.output if result.success else ""), call


def _write_text(tools: ToolSandbox, rel_path: str, text: str) -> dict[str, Any]:
    result = tools.execute(
        "file_edit",
        {"path": rel_path, "content": text, "append": False},
        allowed_tools=["file_edit"],
    )
    return {
        "tool": "file_edit",
        "input": {"path": rel_path, "append": False},
        "success": result.success,
        "error": result.error,
        "metadata": result.metadata,
    }


def _apply_edit(
    *,
    repo_path: Path,
    tools: ToolSandbox,
    edit: dict[str, Any],
) -> tuple[bool, str, list[dict[str, Any]], str]:
    tool_calls: list[dict[str, Any]] = []
    raw_path = str(edit.get("path", "")).strip()
    if not raw_path:
        return False, "", tool_calls, "missing path"

    try:
        rel_path = _safe_repo_relpath(repo_path, raw_path)
    except ValueError as exc:
        return False, "", tool_calls, str(exc)

    absolute = (repo_path / rel_path).resolve()
    if _is_probably_binary(absolute):
        return False, rel_path, tool_calls, "refusing to edit binary file"

    operation = str(edit.get("operation", "replace")).strip().lower()

    current_text, read_call = _read_text(tools, rel_path)
    tool_calls.append(read_call)

    if not read_call["success"] and operation != "overwrite":
        return False, rel_path, tool_calls, f"failed to read file for operation {operation}"

    new_text = current_text
    if operation == "replace":
        search = str(edit.get("search", ""))
        replace = str(edit.get("replace", ""))
        if not search:
            return False, rel_path, tool_calls, "replace operation missing search"
        if search in current_text:
            new_text = current_text.replace(search, replace)
        elif replace in current_text:
            return False, rel_path, tool_calls, "idempotent: replacement already applied"
        else:
            return False, rel_path, tool_calls, "search pattern not found"
    elif operation == "append":
        suffix = str(edit.get("content", ""))
        if not suffix:
            return False, rel_path, tool_calls, "append operation missing content"
        if suffix in current_text:
            return False, rel_path, tool_calls, "idempotent: content already present"
        new_text = current_text + suffix
    elif operation == "ensure_contains":
        content = str(edit.get("content", ""))
        if not content:
            return False, rel_path, tool_calls, "ensure_contains missing content"
        if content in current_text:
            return False, rel_path, tool_calls, "idempotent: content already present"
        if current_text and not current_text.endswith("\n"):
            new_text = current_text + "\n" + content
        else:
            new_text = current_text + content
    elif operation == "overwrite":
        content = str(edit.get("content", ""))
        if current_text == content:
            return False, rel_path, tool_calls, "idempotent: content unchanged"
        new_text = content
    else:
        return False, rel_path, tool_calls, f"unsupported operation '{operation}'"

    if new_text == current_text:
        return False, rel_path, tool_calls, "idempotent: no content change"

    write_call = _write_text(tools, rel_path, new_text)
    tool_calls.append(write_call)
    if not write_call["success"]:
        return False, rel_path, tool_calls, "failed to write updated file"

    return True, rel_path, tool_calls, "applied"


def _pick_target_file(repo_path: Path, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if not candidate or candidate.startswith("<"):
            continue
        path = (repo_path / candidate).resolve()
        if not path.exists() or not path.is_file():
            continue
        if _is_probably_binary(path):
            continue
        return candidate
    return None


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
    plan = artifacts.get("PhasePlan", {})
    qa_report = artifacts.get("QAReport", {})
    codebase_map = artifacts.get("CodebaseMap", {})

    plan_files = plan.get("candidate_files", []) if isinstance(plan, dict) else []
    codebase_modules = codebase_map.get("modules", []) if isinstance(codebase_map, dict) else []
    candidate_files = [*plan_files, *codebase_modules]

    target_file = _pick_target_file(repo_path, candidate_files)
    files_to_modify = [target_file] if target_file else []

    debug_mode = bool(feedback)
    debug_signals = qa_report.get("issues", []) if isinstance(qa_report, dict) else []

    edit_marker = f"# agent_harness change: {user_request.strip()}"
    fallback_edits: list[dict[str, Any]] = []
    if target_file:
        fallback_edits.append(
            {
                "path": target_file,
                "operation": "ensure_contains",
                "content": edit_marker + "\n",
            }
        )

    fallback_content = {
        "objective": user_request,
        "summary": (
            f"{'Debug' if debug_mode else 'Implementation'} patch candidate {sample_index} "
            f"for request '{user_request}'."
        ),
        "files_to_modify": files_to_modify,
        "patch_plan": [
            f"Implement requested capability: {user_request}",
            "Apply concrete file edits",
            "Run verification commands",
            "Capture diff summary and changed files",
        ],
        "edits": fallback_edits,
        "verification_commands": ["pytest -q"],
        "estimated_complexity": "medium",
    }

    artifact_schema = {
        "objective": "string",
        "summary": "string",
        "files_to_modify": ["string"],
        "patch_plan": ["string"],
        "edits": [
            {
                "path": "string",
                "operation": "replace|append|overwrite|ensure_contains",
                "search": "string",
                "replace": "string",
                "content": "string",
            }
        ],
        "verification_commands": ["string"],
        "estimated_complexity": "string",
    }

    llm_context = {
        "user_request": user_request,
        "sample_index": sample_index,
        "feedback": feedback or [],
        "debug_mode": debug_mode,
        "debug_signals": debug_signals,
        "phase_plan": plan,
        "qa_report": qa_report,
        "repo_modules": codebase_modules[:50],
        "fallback_patch": fallback_content,
    }

    content, response, prompt = generate_structured_content(
        provider=provider,
        task_goal=(
            "Produce and apply concrete repository edits. "
            "Return valid edit operations that can be executed directly."
        ),
        artifact_name="ImplementationPatch",
        artifact_schema=artifact_schema,
        context_payload=llm_context,
        fallback_content=fallback_content,
    )

    tool_calls: list[dict[str, Any]] = list(response.tool_calls)
    applied_files: list[str] = []
    edit_results: list[dict[str, Any]] = []

    edits = content.get("edits", [])
    if not isinstance(edits, list):
        edits = []

    for edit in edits:
        if not isinstance(edit, dict):
            continue
        changed, rel_path, calls, status = _apply_edit(repo_path=repo_path, tools=tools, edit=edit)
        tool_calls.extend(calls)
        if changed and rel_path:
            applied_files.append(rel_path)
        edit_results.append(
            {
                "path": rel_path,
                "operation": str(edit.get("operation", "")),
                "status": status,
                "changed": changed,
            }
        )

    unique_changed_files = sorted(set(applied_files))

    git_diff_snippet = ""
    if unique_changed_files:
        diff_result = tools.execute(
            "git",
            {"args": ["diff", "--", *unique_changed_files]},
            allowed_tools=["git"],
        )
        tool_calls.append(
            {
                "tool": "git",
                "input": {"args": ["diff", "--", *unique_changed_files]},
                "success": diff_result.success,
                "error": diff_result.error,
                "metadata": diff_result.metadata,
            }
        )
        if diff_result.success:
            git_diff_snippet = diff_result.output[:3000]

    content["files_changed"] = unique_changed_files
    content["commands_executed"] = [
        str(call.get("tool")) + (f": {call.get('input')!s}" if call.get("input") else "")
        for call in tool_calls
        if isinstance(call, dict) and call.get("tool") is not None
    ]
    content["edit_results"] = edit_results
    content["diff_summary"] = (
        f"Applied edits to {len(unique_changed_files)} file(s)."
        if unique_changed_files
        else "No file edits were applied."
    )
    content["git_diff_snippet"] = git_diff_snippet

    return SkillOutput(
        content=content,
        prompt=prompt,
        tool_calls=tool_calls,
        token_usage=response.token_usage,
        model=response.model,
        latency=response.latency,
    )

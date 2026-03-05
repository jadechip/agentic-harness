from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .base_tool import BaseTool, ToolResult


class GitTool(BaseTool):
    name = "git"

    _allowed_subcommands = {
        "status",
        "diff",
        "log",
        "show",
        "rev-parse",
        "branch",
        "ls-files",
    }

    def __init__(self, workspace: str | Path, timeout_seconds: int = 20) -> None:
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        args = payload.get("args", [])
        if isinstance(args, str):
            args = args.split()
        if not isinstance(args, list) or not args:
            return ToolResult(success=False, output="", error="Missing git arguments")

        subcommand = str(args[0])
        if subcommand not in self._allowed_subcommands:
            return ToolResult(success=False, output="", error=f"Blocked git subcommand '{subcommand}'")

        cmd = ["git", *[str(a) for a in args]]
        try:
            completed = subprocess.run(
                cmd,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Git command timed out")

        output = (completed.stdout or "") + (completed.stderr or "")
        return ToolResult(
            success=completed.returncode == 0,
            output=output.strip(),
            metadata={"returncode": completed.returncode, "command": cmd},
        )

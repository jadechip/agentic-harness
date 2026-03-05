from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from .base_tool import BaseTool, ToolResult


class ShellTool(BaseTool):
    name = "shell"

    _blocked_commands = {
        "rm",
        "sudo",
        "shutdown",
        "reboot",
        "mkfs",
        "dd",
        "killall",
    }

    _blocked_fragments = ["rm -rf /", ":(){:|:&};:"]

    def __init__(self, workspace: str | Path, timeout_seconds: int = 30) -> None:
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        command = str(payload.get("command", "")).strip()
        if not command:
            return ToolResult(success=False, output="", error="Missing command")

        if any(fragment in command for fragment in self._blocked_fragments):
            return ToolResult(success=False, output="", error="Blocked command fragment")

        try:
            head = shlex.split(command)[0]
        except Exception:
            return ToolResult(success=False, output="", error="Unable to parse command")

        if head in self._blocked_commands:
            return ToolResult(success=False, output="", error=f"Blocked command '{head}'")

        try:
            completed = subprocess.run(
                ["/bin/zsh", "-lc", command],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Command timed out")

        success = completed.returncode == 0
        output = (completed.stdout or "") + (completed.stderr or "")
        return ToolResult(success=success, output=output.strip(), metadata={"returncode": completed.returncode})

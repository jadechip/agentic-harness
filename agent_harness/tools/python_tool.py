from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base_tool import BaseTool, ToolResult


class PythonTool(BaseTool):
    name = "python"

    def __init__(self, workspace: str | Path, timeout_seconds: int = 20) -> None:
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        code = str(payload.get("code", "")).strip()
        if not code:
            return ToolResult(success=False, output="", error="Missing python code")

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            script_path = fh.name

        try:
            completed = subprocess.run(
                ["python3", script_path],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Python execution timed out")

        output = (completed.stdout or "") + (completed.stderr or "")
        return ToolResult(success=completed.returncode == 0, output=output.strip())

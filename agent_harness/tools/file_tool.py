from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_tool import BaseTool, ToolResult


class _BaseFileTool(BaseTool):
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()

    def _resolve(self, path_value: str) -> Path:
        candidate = (self.workspace / path_value).resolve()
        if self.workspace not in candidate.parents and candidate != self.workspace:
            raise ValueError("Path escapes workspace")
        return candidate


class FileReadTool(_BaseFileTool):
    name = "file_read"

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        path_value = str(payload.get("path", "")).strip()
        if not path_value:
            return ToolResult(success=False, output="", error="Missing file path")

        max_chars = int(payload.get("max_chars", 20000))

        try:
            path = self._resolve(path_value)
            if not path.exists() or not path.is_file():
                return ToolResult(success=False, output="", error="File does not exist")
            text = path.read_text(encoding="utf-8")
            return ToolResult(success=True, output=text[:max_chars])
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class FileEditTool(_BaseFileTool):
    name = "file_edit"

    def __init__(self, workspace: str | Path, max_edits: int = 200) -> None:
        super().__init__(workspace)
        self.max_edits = max_edits
        self._edits = 0

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        path_value = str(payload.get("path", "")).strip()
        content = payload.get("content")
        append = bool(payload.get("append", False))

        if not path_value:
            return ToolResult(success=False, output="", error="Missing file path")
        if content is None:
            return ToolResult(success=False, output="", error="Missing file content")
        if self._edits >= self.max_edits:
            return ToolResult(success=False, output="", error="Edit limit exceeded")

        try:
            path = self._resolve(path_value)
            path.parent.mkdir(parents=True, exist_ok=True)
            if append:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(str(content))
            else:
                path.write_text(str(content), encoding="utf-8")
            self._edits += 1
            return ToolResult(success=True, output=f"Wrote {path}")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

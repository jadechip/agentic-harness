from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResult:
    success: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    name: str

    @abstractmethod
    def execute(self, payload: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


class ToolSandbox:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def available_tools(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, tool_name: str, payload: dict[str, Any], allowed_tools: list[str] | None = None) -> ToolResult:
        if allowed_tools is not None and tool_name not in allowed_tools:
            return ToolResult(success=False, output="", error=f"Tool '{tool_name}' is not allowed for this skill")

        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(success=False, output="", error=f"Unknown tool '{tool_name}'")

        return tool.execute(payload)

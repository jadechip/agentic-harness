from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    input_artifacts: list[str]
    output_artifact: str
    allowed_tools: list[str]

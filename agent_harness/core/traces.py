from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Trace:
    run_id: str
    task_name: str
    skill_name: str
    prompt: str
    tool_calls: list[dict[str, Any]]
    artifact_id: str
    evaluation_score: float
    token_usage: int
    latency: float
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "skill_name": self.skill_name,
            "prompt": self.prompt,
            "tool_calls": self.tool_calls,
            "artifact_id": self.artifact_id,
            "evaluation_score": self.evaluation_score,
            "token_usage": self.token_usage,
            "latency": self.latency,
            "created_at": self.created_at.isoformat(),
        }


class TraceFactory:
    @staticmethod
    def create(
        run_id: str,
        task_name: str,
        skill_name: str,
        prompt: str,
        tool_calls: list[dict[str, Any]],
        artifact_id: str,
        evaluation_score: float,
        token_usage: int,
        latency: float,
    ) -> Trace:
        return Trace(
            run_id=run_id,
            task_name=task_name,
            skill_name=skill_name,
            prompt=prompt,
            tool_calls=tool_calls,
            artifact_id=artifact_id,
            evaluation_score=evaluation_score,
            token_usage=token_usage,
            latency=latency,
            created_at=utc_now(),
        )

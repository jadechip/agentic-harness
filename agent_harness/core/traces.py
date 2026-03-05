from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Trace:
    run_id: str
    task_name: str
    skill_name: str
    provider_name: str
    model_name: str
    temperature: float
    max_tokens: int
    evaluator_mode: str
    prompt: str
    tool_calls: list[dict[str, Any]]
    artifact_id: str
    candidate_index: int
    selected: bool
    evaluation_score: float
    evaluation_breakdown: dict[str, float] = field(default_factory=dict)
    token_usage: int = 0
    latency: float = 0.0
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "skill_name": self.skill_name,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "evaluator_mode": self.evaluator_mode,
            "prompt": self.prompt,
            "tool_calls": self.tool_calls,
            "artifact_id": self.artifact_id,
            "candidate_index": self.candidate_index,
            "selected": self.selected,
            "evaluation_score": self.evaluation_score,
            "evaluation_breakdown": self.evaluation_breakdown,
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
        provider_name: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        evaluator_mode: str,
        prompt: str,
        tool_calls: list[dict[str, Any]],
        artifact_id: str,
        candidate_index: int,
        selected: bool,
        evaluation_score: float,
        evaluation_breakdown: dict[str, float],
        token_usage: int,
        latency: float,
    ) -> Trace:
        return Trace(
            run_id=run_id,
            task_name=task_name,
            skill_name=skill_name,
            provider_name=provider_name,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            evaluator_mode=evaluator_mode,
            prompt=prompt,
            tool_calls=tool_calls,
            artifact_id=artifact_id,
            candidate_index=candidate_index,
            selected=selected,
            evaluation_score=evaluation_score,
            evaluation_breakdown=evaluation_breakdown,
            token_usage=token_usage,
            latency=latency,
            created_at=utc_now(),
        )

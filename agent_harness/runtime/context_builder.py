from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_harness.core.tasks import Task
from agent_harness.store.artifact_store import ArtifactStore


@dataclass(slots=True)
class ContextBuilder:
    artifact_store: ArtifactStore
    token_budget: int = 4000
    system_instructions: str = (
        "You are a harness-driven coding agent. Produce the required artifact deterministically from context."
    )

    def build(
        self,
        run_id: str,
        task: Task,
        user_request: str,
        feedback: list[str] | None = None,
        prior_artifact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        for artifact_type in task.context:
            artifact = self.artifact_store.get_latest_selected(run_id, artifact_type)
            if artifact is not None:
                artifacts[artifact_type] = artifact.content

        context = {
            "system_instructions": self.system_instructions,
            "task_name": task.name,
            "user_request": user_request,
            "artifacts": artifacts,
            "feedback": feedback or [],
            "previous_artifact": prior_artifact or {},
        }

        return self._apply_token_budget(context)

    def render_prompt(self, context: dict[str, Any]) -> str:
        return json.dumps(context, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def _apply_token_budget(self, context: dict[str, Any]) -> dict[str, Any]:
        rendered = self.render_prompt(context)
        words = len(rendered.split())
        if words <= self.token_budget:
            return context

        reduced = dict(context)
        reduced_artifacts: dict[str, Any] = {}
        for artifact_type, content in context.get("artifacts", {}).items():
            reduced_artifacts[artifact_type] = self._truncate_content(content)
        reduced["artifacts"] = reduced_artifacts
        return reduced

    def _truncate_content(self, value: Any) -> Any:
        if isinstance(value, str):
            return value[:1500]
        if isinstance(value, list):
            return [self._truncate_content(v) for v in value[:50]]
        if isinstance(value, dict):
            return {k: self._truncate_content(v) for k, v in list(value.items())[:50]}
        return value

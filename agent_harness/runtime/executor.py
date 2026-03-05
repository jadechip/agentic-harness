from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent_harness.core.artifacts import Artifact, ArtifactFactory
from agent_harness.providers.base_provider import LLMProvider
from agent_harness.core.tasks import Task
from agent_harness.skills.common import SkillOutput
from agent_harness.tools.base_tool import ToolSandbox


SkillCallable = Callable[..., SkillOutput]


@dataclass(slots=True)
class ExecutionCandidate:
    sample_index: int
    artifact: Artifact
    prompt: str
    tool_calls: list[dict]
    token_usage: int
    provider_name: str
    model: str
    temperature: float
    max_tokens: int
    latency: float


class AgentExecutor:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolSandbox,
        skill_registry: dict[str, SkillCallable] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.skill_registry = skill_registry or {}

    def register_skill(self, skill_name: str, fn: SkillCallable) -> None:
        self.skill_registry[skill_name] = fn

    def _resolve_skill(self, skill_name: str) -> SkillCallable:
        if skill_name in self.skill_registry:
            return self.skill_registry[skill_name]

        module_name = f"agent_harness.skills.{skill_name}"
        module = importlib.import_module(module_name)
        fn = getattr(module, "run", None)
        if fn is None:
            raise ValueError(f"Skill module '{module_name}' does not define run()")

        self.skill_registry[skill_name] = fn
        return fn

    def run_task(
        self,
        task: Task,
        context: dict,
        user_request: str,
        repo_path: Path,
        feedback: list[str],
        artifact_schema: dict | None = None,
        parent_artifact_id: str | None = None,
        schema_version: str = "1.0",
    ) -> list[ExecutionCandidate]:
        skill_fn = self._resolve_skill(task.skill)
        sample_count = max(1, task.samples)

        if sample_count == 1:
            return [
                self._run_single_candidate(
                    skill_fn=skill_fn,
                    task=task,
                    context=context,
                    user_request=user_request,
                    repo_path=repo_path,
                    feedback=feedback,
                    artifact_schema=artifact_schema,
                    parent_artifact_id=parent_artifact_id,
                    sample_index=1,
                    schema_version=schema_version,
                )
            ]

        candidates: list[ExecutionCandidate] = []
        errors: list[str] = []
        max_workers = min(sample_count, 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._run_single_candidate,
                    skill_fn,
                    task,
                    context,
                    user_request,
                    repo_path,
                    feedback,
                    artifact_schema,
                    parent_artifact_id,
                    sample_idx,
                    schema_version,
                )
                for sample_idx in range(1, sample_count + 1)
            ]
            for future in as_completed(futures):
                try:
                    candidates.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))

        if not candidates:
            detail = "; ".join(errors) if errors else "unknown error"
            raise RuntimeError(f"All task candidates failed for task '{task.name}': {detail}")

        candidates.sort(key=lambda c: c.sample_index)
        return candidates

    def _run_single_candidate(
        self,
        skill_fn: SkillCallable,
        task: Task,
        context: dict,
        user_request: str,
        repo_path: Path,
        feedback: list[str],
        artifact_schema: dict | None,
        parent_artifact_id: str | None,
        sample_index: int,
        schema_version: str,
    ) -> ExecutionCandidate:
        started = time.perf_counter()
        skill_output = skill_fn(
            context=context,
            user_request=user_request,
            repo_path=repo_path,
            tools=self.tools,
            provider=self.provider,
            sample_index=sample_index,
            feedback=feedback,
        )
        elapsed = time.perf_counter() - started
        latency = skill_output.latency if skill_output.latency > 0 else elapsed

        artifact = ArtifactFactory.create(
            artifact_type=task.produces,
            schema_version=schema_version,
            content=skill_output.content,
            produced_by_task=task.name,
            schema=artifact_schema,
            parent_artifact_id=parent_artifact_id,
            candidate_index=sample_index,
        )

        return ExecutionCandidate(
            sample_index=sample_index,
            artifact=artifact,
            prompt=skill_output.prompt,
            tool_calls=skill_output.tool_calls,
            token_usage=skill_output.token_usage,
            provider_name=self.provider.provider_name,
            model=skill_output.model,
            temperature=self.provider.default_temperature,
            max_tokens=self.provider.default_max_tokens,
            latency=latency,
        )

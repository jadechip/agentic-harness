from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_harness.providers.base_provider import LLMProvider, ProviderResponse


@dataclass(slots=True)
class SkillOutput:
    content: dict[str, Any]
    prompt: str
    tool_calls: list[dict[str, Any]]
    token_usage: int
    model: str
    latency: float


def estimate_tokens(text: str) -> int:
    words = max(1, len(text.split()))
    return int(words / 0.75)


def _extract_json_segment(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    # Direct JSON object.
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Fenced code block, typically ```json ... ```.
    fence_marker = "```"
    if fence_marker in stripped:
        parts = stripped.split(fence_marker)
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate

    # Best-effort bracket matching for first JSON object.
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]

    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    segment = _extract_json_segment(text)
    if segment is None:
        return None

    try:
        parsed = json.loads(segment)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def merge_with_fallback(generated: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
    if not generated:
        return dict(fallback)

    merged = dict(fallback)
    for key, value in generated.items():
        merged[key] = value
    return merged


def build_structured_prompts(
    *,
    task_goal: str,
    artifact_name: str,
    artifact_schema: dict[str, Any],
    context_payload: dict[str, Any],
) -> tuple[str, str, str]:
    system_prompt = (
        "You are an expert coding agent operating in a harness runtime. "
        "Return only valid JSON with no markdown, no prose, and no code fences."
    )

    user_prompt = (
        f"Task goal: {task_goal}\n"
        f"Artifact type: {artifact_name}\n"
        f"Return JSON matching this schema exactly:\n{json.dumps(artifact_schema, indent=2, sort_keys=True)}\n"
        f"Context:\n{json.dumps(context_payload, indent=2, sort_keys=True, default=str)}\n"
        "Constraints:\n"
        "- Use concrete, repository-grounded details from context\n"
        "- Keep field types consistent with schema\n"
        "- If unknown, use empty lists/objects or concise strings instead of null"
    )

    full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    return system_prompt, user_prompt, full_prompt


def generate_structured_content(
    *,
    provider: LLMProvider,
    task_goal: str,
    artifact_name: str,
    artifact_schema: dict[str, Any],
    context_payload: dict[str, Any],
    fallback_content: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ProviderResponse, str]:
    system_prompt, user_prompt, full_prompt = build_structured_prompts(
        task_goal=task_goal,
        artifact_name=artifact_name,
        artifact_schema=artifact_schema,
        context_payload=context_payload,
    )

    response = provider.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools or [],
        temperature=float(temperature if temperature is not None else provider.default_temperature),
        max_tokens=int(max_tokens if max_tokens is not None else provider.default_max_tokens),
    )

    generated = parse_json_object(response.text)
    content = merge_with_fallback(generated, fallback_content)
    return content, response, full_prompt

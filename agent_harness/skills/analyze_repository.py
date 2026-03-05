from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agent_harness.providers.base_provider import LLMProvider
from agent_harness.skills.common import SkillOutput, generate_structured_content
from agent_harness.tools.base_tool import ToolSandbox


LANGUAGE_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".rb": "Ruby",
    ".php": "PHP",
}

ENTRYPOINT_NAMES = {
    "main.py",
    "app.py",
    "server.py",
    "index.js",
    "index.ts",
    "main.go",
}

CONFIG_NAMES = {
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
}

IMPORT_PATTERN = re.compile(r"^\s*(?:from\s+([\w\.]+)|import\s+([\w\.]+))")


def _scan_python_dependencies(path: Path) -> list[str]:
    deps: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = IMPORT_PATTERN.match(line)
            if not match:
                continue
            dep = match.group(1) or match.group(2)
            if dep:
                deps.add(dep.split(".")[0])
    except Exception:
        return []
    return sorted(deps)


def run(
    *,
    context: dict[str, Any],
    user_request: str,
    repo_path: Path,
    tools: ToolSandbox,
    provider: LLMProvider,
    sample_index: int,
    feedback: list[str] | None = None,
) -> SkillOutput:
    modules: list[str] = []
    entrypoints: list[str] = []
    languages: set[str] = set()
    dependency_graph: dict[str, list[str]] = {}
    config_files: list[str] = []
    folder_counter: Counter[str] = Counter()

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        root_path = Path(root)
        for filename in files:
            rel = str((root_path / filename).relative_to(repo_path))
            ext = Path(filename).suffix.lower()

            if filename in CONFIG_NAMES:
                config_files.append(rel)

            if ext in LANGUAGE_EXTENSIONS:
                modules.append(rel)
                languages.add(LANGUAGE_EXTENSIONS[ext])
                top_folder = rel.split("/", 1)[0] if "/" in rel else "."
                folder_counter[top_folder] += 1

            if filename in ENTRYPOINT_NAMES:
                entrypoints.append(rel)

            if ext == ".py":
                deps = _scan_python_dependencies(root_path / filename)
                if deps:
                    dependency_graph[rel] = deps

    hot_folders = ", ".join(folder for folder, _ in folder_counter.most_common(3)) or "root"
    architecture_summary = (
        f"Repository has {len(modules)} source modules across {len(languages)} languages. "
        f"Primary areas: {hot_folders}. "
        f"Detected {len(entrypoints)} likely entrypoints and {len(config_files)} config files."
    )

    fallback_content = {
        "modules": sorted(modules),
        "entrypoints": sorted(entrypoints),
        "languages": sorted(languages),
        "dependency_graph": dependency_graph,
        "config_files": sorted(config_files),
        "architecture_summary": architecture_summary,
    }

    artifact_schema = {
        "modules": ["string"],
        "entrypoints": ["string"],
        "languages": ["string"],
        "dependency_graph": {"module_path": ["dependency_name"]},
        "config_files": ["string"],
        "architecture_summary": "string",
    }

    llm_context = {
        "user_request": user_request,
        "sample_index": sample_index,
        "feedback": feedback or [],
        "observed_repository_facts": fallback_content,
    }

    content, response, prompt = generate_structured_content(
        provider=provider,
        task_goal="Derive comprehensive structural understanding of the repository.",
        artifact_name="CodebaseMap",
        artifact_schema=artifact_schema,
        context_payload=llm_context,
        fallback_content=fallback_content,
    )

    return SkillOutput(
        content=content,
        prompt=prompt,
        tool_calls=list(response.tool_calls),
        token_usage=response.token_usage,
        model=response.model,
        latency=response.latency,
    )

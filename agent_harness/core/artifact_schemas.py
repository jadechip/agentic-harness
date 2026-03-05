from __future__ import annotations

from typing import Any


BUILTIN_ARTIFACT_SCHEMAS: dict[str, dict[str, Any]] = {
    "CodebaseMap": {
        "modules": ["string"],
        "entrypoints": ["string"],
        "languages": ["string"],
        "dependency_graph": {"module_path": ["string"]},
        "config_files": ["string"],
        "architecture_summary": "string",
    },
    "PhasePlan": {
        "objective": "string",
        "summary": "string",
        "phases": [{"name": "string", "tasks": ["string"]}],
        "candidate_files": ["string"],
        "risks": ["string"],
        "test_strategy": "string",
    },
    "ImplementationPatch": {
        "objective": "string",
        "summary": "string",
        "files_to_modify": ["string"],
        "patch_plan": ["string"],
    },
    "QAReport": {
        "verification_ran": "boolean",
        "commands_run": ["string"],
        "tests_ran": "integer",
        "tests_failed": "integer",
        "pass_rate": "float",
        "issues": ["string"],
    },
}

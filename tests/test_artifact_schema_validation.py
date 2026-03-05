import pytest

from agent_harness.core.artifact_schemas import BUILTIN_ARTIFACT_SCHEMAS
from agent_harness.core.artifacts import ArtifactFactory


def test_artifact_factory_validates_schema() -> None:
    artifact = ArtifactFactory.create(
        artifact_type="CodebaseMap",
        schema_version="1.0",
        content={
            "modules": ["a.py"],
            "entrypoints": ["main.py"],
            "languages": ["Python"],
            "dependency_graph": {"a.py": ["os"]},
            "config_files": ["pyproject.toml"],
            "architecture_summary": "summary",
        },
        produced_by_task="repo_analysis",
        schema=BUILTIN_ARTIFACT_SCHEMAS["CodebaseMap"],
    )
    assert artifact.type == "CodebaseMap"


def test_artifact_factory_rejects_invalid_schema() -> None:
    with pytest.raises(ValueError):
        ArtifactFactory.create(
            artifact_type="CodebaseMap",
            schema_version="1.0",
            content={
                "modules": "not-a-list",
            },
            produced_by_task="repo_analysis",
            schema=BUILTIN_ARTIFACT_SCHEMAS["CodebaseMap"],
        )

from pathlib import Path

import pytest

from agent_harness.core.artifact_schemas import BUILTIN_ARTIFACT_SCHEMAS
from agent_harness.dsl.parser import parse_harness_yaml
from agent_harness.dsl.validator import HarnessValidationError, validate_harness


def test_validator_rejects_cycle(tmp_path: Path) -> None:
    harness_file = tmp_path / "cycle.yaml"
    harness_file.write_text(
        "harness: cycle\n"
        "tasks:\n"
        "  a:\n"
        "    skill: analyze_repository\n"
        "    produces: CodebaseMap\n"
        "    evaluate: repo_map_quality\n"
        "  b:\n"
        "    skill: generate_plan\n"
        "    produces: PhasePlan\n"
        "    evaluate: plan_quality\n"
        "flow:\n"
        "  - a -> b\n"
        "  - b -> a\n",
        encoding="utf-8",
    )

    harness = parse_harness_yaml(harness_file)
    with pytest.raises(HarnessValidationError):
        validate_harness(
            harness,
            available_skills={"analyze_repository", "generate_plan"},
            available_evaluations={"repo_map_quality", "plan_quality"},
            available_artifact_schemas=BUILTIN_ARTIFACT_SCHEMAS.keys(),
        )


def test_validator_rejects_unreachable_without_flag(tmp_path: Path) -> None:
    harness_file = tmp_path / "unreachable.yaml"
    harness_file.write_text(
        "harness: unreachable\n"
        "tasks:\n"
        "  a:\n"
        "    skill: analyze_repository\n"
        "    produces: CodebaseMap\n"
        "    evaluate: repo_map_quality\n"
        "  b:\n"
        "    skill: generate_plan\n"
        "    produces: PhasePlan\n"
        "    evaluate: plan_quality\n"
        "  c:\n"
        "    skill: implement_change\n"
        "    produces: ImplementationPatch\n"
        "    evaluate: patch_quality\n"
        "flow:\n"
        "  - a -> b\n",
        encoding="utf-8",
    )

    harness = parse_harness_yaml(harness_file)
    with pytest.raises(HarnessValidationError):
        validate_harness(
            harness,
            available_skills={"analyze_repository", "generate_plan", "implement_change"},
            available_evaluations={"repo_map_quality", "plan_quality", "patch_quality"},
            available_artifact_schemas=BUILTIN_ARTIFACT_SCHEMAS.keys(),
        )


def test_validator_rejects_unknown_artifact_schema(tmp_path: Path) -> None:
    harness_file = tmp_path / "unknown_artifact.yaml"
    harness_file.write_text(
        "harness: unknown_artifact\n"
        "tasks:\n"
        "  a:\n"
        "    skill: analyze_repository\n"
        "    produces: UnknownArtifact\n"
        "    evaluate: repo_map_quality\n"
        "flow: []\n",
        encoding="utf-8",
    )

    harness = parse_harness_yaml(harness_file)
    with pytest.raises(HarnessValidationError):
        validate_harness(
            harness,
            available_skills={"analyze_repository"},
            available_evaluations={"repo_map_quality"},
            available_artifact_schemas=BUILTIN_ARTIFACT_SCHEMAS.keys(),
        )


def test_validator_allows_unreachable_with_flag(tmp_path: Path) -> None:
    harness_file = tmp_path / "allowed_unreachable.yaml"
    harness_file.write_text(
        "harness: allowed_unreachable\n"
        "settings:\n"
        "  allow_unreachable_tasks: true\n"
        "tasks:\n"
        "  a:\n"
        "    skill: analyze_repository\n"
        "    produces: CodebaseMap\n"
        "    evaluate: repo_map_quality\n"
        "  b:\n"
        "    skill: generate_plan\n"
        "    produces: PhasePlan\n"
        "    evaluate: plan_quality\n"
        "  c:\n"
        "    skill: implement_change\n"
        "    produces: ImplementationPatch\n"
        "    evaluate: patch_quality\n"
        "flow:\n"
        "  - a -> b\n",
        encoding="utf-8",
    )

    harness = parse_harness_yaml(harness_file)
    validate_harness(
        harness,
        available_skills={"analyze_repository", "generate_plan", "implement_change"},
        available_evaluations={"repo_map_quality", "plan_quality", "patch_quality"},
        available_artifact_schemas=BUILTIN_ARTIFACT_SCHEMAS.keys(),
    )

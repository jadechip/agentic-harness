from pathlib import Path

from agent_harness.dsl.parser import parse_harness_yaml
from agent_harness.dsl.validator import validate_harness


def test_parse_example_harness() -> None:
    harness_path = Path("examples/coding_agent.harness.yaml")
    harness = parse_harness_yaml(harness_path)

    assert harness.name == "coding_agent"
    assert set(harness.tasks) == {"repo_analysis", "planning", "implementation", "verification"}
    assert ("planning", "implementation") in harness.flow
    assert harness.feedback[0].source_task == "verification"
    assert harness.feedback[0].target_task == "implementation"


def test_validate_harness_with_known_skills_and_evaluations() -> None:
    harness = parse_harness_yaml(Path("examples/coding_agent.harness.yaml"))
    validate_harness(
        harness,
        available_skills={
            "analyze_repository",
            "generate_plan",
            "implement_change",
            "verify_behavior",
            "debug_failure",
        },
        available_evaluations={
            "repo_map_quality",
            "plan_quality",
            "patch_quality",
            "test_pass_rate",
        },
    )

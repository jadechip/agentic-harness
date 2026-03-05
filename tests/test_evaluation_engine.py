from agent_harness.evaluation.engine import EvaluationEngine
from agent_harness.evaluations.repo_map_quality import get_contract


def test_repo_map_quality_passes_for_rich_content() -> None:
    contract = get_contract()
    engine = EvaluationEngine()

    artifact_content = {
        "modules": [f"module_{i}.py" for i in range(20)],
        "entrypoints": ["main.py"],
        "languages": ["Python"],
        "dependency_graph": {"module_1.py": ["os", "json"], "module_2.py": ["typing"]},
        "config_files": ["pyproject.toml"],
        "architecture_summary": "This repository has a layered service architecture with a CLI entrypoint and test modules.",
    }

    result = engine.evaluate(contract, artifact_content)
    assert result.score >= contract.pass_threshold
    assert result.passed is True

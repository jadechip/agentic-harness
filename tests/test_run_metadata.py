from pathlib import Path

from agent_harness.cli.main import _build_scheduler


def test_run_metadata_contains_provider_settings(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "from app import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    harness_file = tmp_path / "harness.yaml"
    harness_file.write_text(
        "harness: metadata_test\n"
        "settings:\n"
        "  evaluation_mode: deterministic\n"
        "tasks:\n"
        "  repo_analysis:\n"
        "    skill: analyze_repository\n"
        "    context: []\n"
        "    produces: CodebaseMap\n"
        "    evaluate: repo_map_quality\n"
        "  planning:\n"
        "    skill: generate_plan\n"
        "    context: [CodebaseMap]\n"
        "    produces: PhasePlan\n"
        "    evaluate: plan_quality\n"
        "  implementation:\n"
        "    skill: implement_change\n"
        "    context: [CodebaseMap, PhasePlan]\n"
        "    produces: ImplementationPatch\n"
        "    evaluate: patch_quality\n"
        "  verification:\n"
        "    skill: verify_behavior\n"
        "    context: [ImplementationPatch]\n"
        "    produces: QAReport\n"
        "    evaluate: test_pass_rate\n"
        "flow:\n"
        "  - repo_analysis -> planning\n"
        "  - planning -> implementation\n"
        "  - implementation -> verification\n",
        encoding="utf-8",
    )

    db = tmp_path / "run.db"
    scheduler = _build_scheduler(
        harness_file=harness_file,
        repo_path=repo,
        db_path=db,
        provider_name="mock",
        model_name="default",
    )
    result = scheduler.run(user_request="x", repo_path=repo)

    run_record = scheduler.run_store.get_run(result.run_id)
    assert run_record is not None
    metadata = run_record["metadata"]
    assert metadata["provider"]["name"] == "mock"
    assert "execution_contract" in metadata

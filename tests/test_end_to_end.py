from pathlib import Path

from agent_harness.cli.main import _build_scheduler
from agent_harness.store.trace_store import TraceStore


def test_end_to_end_run_collects_traces(tmp_path: Path) -> None:
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
        "harness: test_harness\n"
        "settings:\n"
        "  context_size: 3000\n"
        "  max_feedback_cycles: 1\n"
        "evaluations:\n"
        "  repo_map_quality:\n"
        "    criteria: [module_coverage, entrypoint_detection, dependency_accuracy, architecture_summary_quality]\n"
        "    weights:\n"
        "      module_coverage: 3\n"
        "      entrypoint_detection: 2\n"
        "      dependency_accuracy: 3\n"
        "      architecture_summary_quality: 2\n"
        "    pass_threshold: 0.65\n"
        "  plan_quality:\n"
        "    criteria: [phase_completeness, spec_alignment, risk_coverage]\n"
        "    weights:\n"
        "      phase_completeness: 4\n"
        "      spec_alignment: 3\n"
        "      risk_coverage: 3\n"
        "    pass_threshold: 0.65\n"
        "  patch_quality:\n"
        "    criteria: [file_targeting, patch_completeness, spec_alignment]\n"
        "    weights:\n"
        "      file_targeting: 3\n"
        "      patch_completeness: 4\n"
        "      spec_alignment: 3\n"
        "    pass_threshold: 0.65\n"
        "  test_pass_rate:\n"
        "    criteria: [test_pass_rate, issue_reporting]\n"
        "    weights:\n"
        "      test_pass_rate: 8\n"
        "      issue_reporting: 2\n"
        "    pass_threshold: 0.65\n"
        "tasks:\n"
        "  repo_analysis:\n"
        "    skill: analyze_repository\n"
        "    context: []\n"
        "    produces: CodebaseMap\n"
        "    evaluate: repo_map_quality\n"
        "    samples: 1\n"
        "    retry_limit: 1\n"
        "  planning:\n"
        "    skill: generate_plan\n"
        "    context: [CodebaseMap]\n"
        "    produces: PhasePlan\n"
        "    evaluate: plan_quality\n"
        "    samples: 1\n"
        "    retry_limit: 1\n"
        "  implementation:\n"
        "    skill: implement_change\n"
        "    context: [CodebaseMap, PhasePlan]\n"
        "    produces: ImplementationPatch\n"
        "    evaluate: patch_quality\n"
        "    samples: 1\n"
        "    retry_limit: 1\n"
        "  verification:\n"
        "    skill: verify_behavior\n"
        "    context: [ImplementationPatch]\n"
        "    produces: QAReport\n"
        "    evaluate: test_pass_rate\n"
        "    samples: 1\n"
        "    retry_limit: 1\n"
        "flow:\n"
        "  - repo_analysis -> planning\n"
        "  - planning -> implementation\n"
        "  - implementation -> verification\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "harness.db"
    scheduler = _build_scheduler(harness_file=harness_file, repo_path=repo, db_path=db_path)
    result = scheduler.run(user_request="Add OAuth login", repo_path=repo)

    assert result.success is True
    assert len(result.task_outcomes) == 4

    traces = TraceStore(db_path).export_run(result.run_id)
    assert len(traces) >= 4
    assert all("evaluation_score" in trace for trace in traces)

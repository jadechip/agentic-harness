import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_harness.cli.main as cli_main
from agent_harness.cli.main import _command_run


def _write_repo_and_harness(tmp_path: Path) -> tuple[Path, Path, Path]:
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
        "harness: provider_contract\n"
        "settings:\n"
        "  evaluation_mode: deterministic\n"
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
        "    context: [CodebaseMap, PhasePlan, QAReport]\n"
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

    db_path = tmp_path / "run.db"
    return repo, harness_file, db_path


def _args(
    *,
    repo: Path,
    harness_file: Path,
    db_path: Path,
    provider: str = "openrouter",
    model: str | None = None,
    mock: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        harness=str(harness_file),
        repo=str(repo),
        request="Add OAuth login",
        db=str(db_path),
        provider=provider,
        model=model,
        mock=mock,
        temperature=None,
        max_tokens=None,
        max_runtime_seconds=None,
        evaluation_mode="deterministic",
        export_traces=None,
    )


def test_run_requires_model_without_mock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo, harness_file, db_path = _write_repo_and_harness(tmp_path)

    rc = _command_run(_args(repo=repo, harness_file=harness_file, db_path=db_path, model=None, mock=False))
    captured = capsys.readouterr()

    assert rc == 2
    assert "--model is required" in captured.out


def test_run_requires_api_key_without_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, harness_file, db_path = _write_repo_and_harness(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    rc = _command_run(
        _args(
            repo=repo,
            harness_file=harness_file,
            db_path=db_path,
            model="openai/gpt-4o",
            mock=False,
        )
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "OPENROUTER_API_KEY is required" in captured.out


def test_run_mock_mode_allows_execution_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, harness_file, db_path = _write_repo_and_harness(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    rc = _command_run(
        _args(
            repo=repo,
            harness_file=harness_file,
            db_path=db_path,
            model=None,
            mock=True,
        )
    )

    assert rc == 0


def test_parse_env_assignment_supports_export_and_quotes() -> None:
    parsed = cli_main._parse_env_assignment("export OPENROUTER_API_KEY='abc123'")
    assert parsed == ("OPENROUTER_API_KEY", "abc123")

    parsed = cli_main._parse_env_assignment('OPENROUTER_API_KEY="xyz789"')
    assert parsed == ("OPENROUTER_API_KEY", "xyz789")

    assert cli_main._parse_env_assignment("# comment") is None
    assert cli_main._parse_env_assignment("INVALID KEY=value") is None


def test_run_loads_openrouter_key_from_repo_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, harness_file, db_path = _write_repo_and_harness(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (repo / ".env").write_text("OPENROUTER_API_KEY=test-key-from-env\n", encoding="utf-8")

    class _StubScheduler:
        def run(self, user_request: str, repo_path: Path) -> SimpleNamespace:
            return SimpleNamespace(
                run_id="run-1",
                success=True,
                feedback_cycles=0,
                task_outcomes=[],
            )

    def _stub_build_scheduler(**kwargs: object) -> _StubScheduler:
        assert kwargs["provider_name"] == "openrouter"
        assert kwargs["model_name"] == "openai/gpt-4o"
        return _StubScheduler()

    monkeypatch.setattr(cli_main, "_build_scheduler", _stub_build_scheduler)

    rc = _command_run(
        _args(
            repo=repo,
            harness_file=harness_file,
            db_path=db_path,
            model="openai/gpt-4o",
            mock=False,
        )
    )

    assert rc == 0
    assert cli_main.os.getenv("OPENROUTER_API_KEY") == "test-key-from-env"

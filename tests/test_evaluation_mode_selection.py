from pathlib import Path

from agent_harness.cli import main as cli_main
from agent_harness.providers.base_provider import LLMProvider, ProviderResponse


class FakeOpenRouterProvider(LLMProvider):
    name = "openrouter"

    def __init__(self) -> None:
        super().__init__(model="openai/gpt-4o")

    def generate(self, system_prompt, user_prompt, tools, temperature, max_tokens):  # type: ignore[override]
        return ProviderResponse(text="{}", tool_calls=[], token_usage=1, latency=0.0, model=self.model)


def _minimal_harness(path: Path) -> None:
    path.write_text(
        "harness: eval_mode_test\n"
        "tasks:\n"
        "  repo_analysis:\n"
        "    skill: analyze_repository\n"
        "    context: []\n"
        "    produces: CodebaseMap\n"
        "    evaluate: repo_map_quality\n"
        "flow: []\n",
        encoding="utf-8",
    )


def test_default_evaluation_mode_mock_is_deterministic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    harness_file = tmp_path / "harness.yaml"
    _minimal_harness(harness_file)

    scheduler = cli_main._build_scheduler(
        harness_file=harness_file,
        repo_path=repo,
        db_path=tmp_path / "run.db",
        provider_name="mock",
        model_name="default",
    )

    assert scheduler.evaluation_mode == "deterministic"


def test_default_evaluation_mode_openrouter_is_hybrid(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    harness_file = tmp_path / "harness.yaml"
    _minimal_harness(harness_file)

    monkeypatch.setattr(
        cli_main.ProviderFactory,
        "create",
        lambda model_name, temperature, max_tokens: FakeOpenRouterProvider(),
    )

    scheduler = cli_main._build_scheduler(
        harness_file=harness_file,
        repo_path=repo,
        db_path=tmp_path / "run.db",
        provider_name="openrouter",
        model_name="openai/gpt-4o",
    )

    assert scheduler.evaluation_mode == "hybrid"

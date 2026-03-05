from pathlib import Path

from agent_harness.cli import main as cli_main
from agent_harness.providers.base_provider import LLMProvider, ProviderResponse


class FakeOpenRouterProvider(LLMProvider):
    name = "openrouter"

    def __init__(self) -> None:
        super().__init__(model="openai/gpt-4o")

    def generate(self, system_prompt, user_prompt, tools, temperature, max_tokens):  # type: ignore[override]
        return ProviderResponse(text="{}", tool_calls=[], token_usage=1, latency=0.0, model=self.model)


def test_openrouter_run_fails_without_modification_or_verification(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")

    harness_file = tmp_path / "harness.yaml"
    harness_file.write_text(
        "harness: contract_test\n"
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
        "flow: []\n",
        encoding="utf-8",
    )

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

    result = scheduler.run(user_request="do something", repo_path=repo)
    assert result.success is False

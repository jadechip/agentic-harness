import json
from pathlib import Path

from agent_harness.providers.base_provider import LLMProvider, ProviderResponse
from agent_harness.skills.implement_change import run as implement_change_run
from agent_harness.tools.base_tool import ToolSandbox
from agent_harness.tools.file_tool import FileEditTool, FileReadTool
from agent_harness.tools.git_tool import GitTool
from agent_harness.tools.python_tool import PythonTool
from agent_harness.tools.shell_tool import ShellTool


class KnownPatchProvider(LLMProvider):
    name = "mock"

    def __init__(self) -> None:
        super().__init__(model="mock/known")

    def generate(self, system_prompt, user_prompt, tools, temperature, max_tokens):  # type: ignore[override]
        payload = {
            "objective": "test",
            "summary": "apply known patch",
            "files_to_modify": ["app.py"],
            "patch_plan": ["replace function return"],
            "edits": [
                {
                    "path": "app.py",
                    "operation": "replace",
                    "search": "return a + b",
                    "replace": "return a + b + 1",
                }
            ],
            "verification_commands": ["pytest -q"],
            "estimated_complexity": "low",
        }
        return ProviderResponse(
            text=json.dumps(payload),
            tool_calls=[],
            token_usage=50,
            latency=0.1,
            model=self.model,
            raw={},
        )


def _tools(repo: Path) -> ToolSandbox:
    tools = ToolSandbox()
    tools.register(ShellTool(workspace=repo))
    tools.register(FileReadTool(workspace=repo))
    tools.register(FileEditTool(workspace=repo))
    tools.register(GitTool(workspace=repo))
    tools.register(PythonTool(workspace=repo))
    return tools


def test_implement_change_applies_known_patch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "app.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    output = implement_change_run(
        context={"artifacts": {"PhasePlan": {"candidate_files": ["app.py"]}}},
        user_request="modify add",
        repo_path=repo,
        tools=_tools(repo),
        provider=KnownPatchProvider(),
        sample_index=1,
        feedback=[],
    )

    assert "return a + b + 1" in target.read_text(encoding="utf-8")
    assert output.content["files_changed"] == ["app.py"]
    assert output.content["edit_results"][0]["changed"] is True

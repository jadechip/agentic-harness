from pathlib import Path

from agent_harness.providers.provider_factory import MockProvider
from agent_harness.skills.verify_behavior import run as verify_behavior_run
from agent_harness.tools.base_tool import ToolSandbox
from agent_harness.tools.file_tool import FileEditTool, FileReadTool
from agent_harness.tools.git_tool import GitTool
from agent_harness.tools.python_tool import PythonTool
from agent_harness.tools.shell_tool import ShellTool


def _tools(repo: Path) -> ToolSandbox:
    tools = ToolSandbox()
    tools.register(ShellTool(workspace=repo))
    tools.register(FileReadTool(workspace=repo))
    tools.register(FileEditTool(workspace=repo))
    tools.register(GitTool(workspace=repo))
    tools.register(PythonTool(workspace=repo))
    return tools


def test_verify_behavior_detects_test_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo_fail"
    repo.mkdir()
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "from app import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 4\n",
        encoding="utf-8",
    )

    output = verify_behavior_run(
        context={},
        user_request="verify",
        repo_path=repo,
        tools=_tools(repo),
        provider=MockProvider(),
        sample_index=1,
        feedback=[],
    )

    assert output.content["verification_ran"] is True
    assert output.content["tests_failed"] >= 1
    assert output.content["failing_commands"]


def test_verify_behavior_detects_test_success(tmp_path: Path) -> None:
    repo = tmp_path / "repo_pass"
    repo.mkdir()
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "from app import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    output = verify_behavior_run(
        context={},
        user_request="verify",
        repo_path=repo,
        tools=_tools(repo),
        provider=MockProvider(),
        sample_index=1,
        feedback=[],
    )

    assert output.content["verification_ran"] is True
    assert output.content["tests_failed"] == 0
    assert output.content["pass_rate"] >= 1.0

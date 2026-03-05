from pathlib import Path

from agent_harness.core.tasks import Task
from agent_harness.providers.provider_factory import MockProvider
from agent_harness.runtime.executor import AgentExecutor
from agent_harness.skills.common import SkillOutput
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


def _flaky_skill(**kwargs):
    sample_index = int(kwargs["sample_index"])
    if sample_index == 2:
        raise RuntimeError("boom")
    return SkillOutput(
        content={
            "objective": "x",
            "summary": f"sample-{sample_index}",
            "files_to_modify": [],
            "patch_plan": [],
        },
        prompt="prompt",
        tool_calls=[],
        token_usage=10,
        model="mock/default",
        latency=0.01,
    )


def test_executor_sampling_handles_partial_candidate_failures(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    task = Task(
        name="implementation",
        skill="implement_change",
        context=[],
        produces="ImplementationPatch",
        evaluation="patch_quality",
        samples=3,
        retry_limit=1,
    )

    executor = AgentExecutor(
        provider=MockProvider(),
        tools=_tools(repo),
        skill_registry={"implement_change": _flaky_skill},
    )

    candidates = executor.run_task(
        task=task,
        context={},
        user_request="x",
        repo_path=repo,
        feedback=[],
        artifact_schema={
            "objective": "string",
            "summary": "string",
            "files_to_modify": ["string"],
            "patch_plan": ["string"],
        },
    )

    assert [candidate.sample_index for candidate in candidates] == [1, 3]

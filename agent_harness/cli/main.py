from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_harness.dsl.parser import parse_harness_yaml
from agent_harness.dsl.validator import validate_harness
from agent_harness.evaluation.engine import EvaluationEngine
from agent_harness.evaluations.patch_quality import get_contract as patch_quality_contract
from agent_harness.evaluations.plan_quality import get_contract as plan_quality_contract
from agent_harness.evaluations.repo_map_quality import get_contract as repo_map_quality_contract
from agent_harness.evaluations.test_pass_rate import get_contract as test_pass_rate_contract
from agent_harness.rl.optimizer import EvolutionaryOptimizer
from agent_harness.rl.reward import compute_reward
from agent_harness.runtime.context_builder import ContextBuilder
from agent_harness.runtime.executor import AgentExecutor
from agent_harness.runtime.improvement_loop import ImprovementLoop
from agent_harness.runtime.scheduler import HarnessScheduler
from agent_harness.skills.analyze_repository import run as analyze_repository_skill
from agent_harness.skills.debug_failure import run as debug_failure_skill
from agent_harness.skills.generate_plan import run as generate_plan_skill
from agent_harness.skills.implement_change import run as implement_change_skill
from agent_harness.skills.verify_behavior import run as verify_behavior_skill
from agent_harness.store.artifact_store import ArtifactStore
from agent_harness.store.run_store import RunStore
from agent_harness.store.trace_store import TraceStore
from agent_harness.tools.base_tool import ToolSandbox
from agent_harness.tools.file_tool import FileEditTool, FileReadTool
from agent_harness.tools.git_tool import GitTool
from agent_harness.tools.python_tool import PythonTool
from agent_harness.tools.shell_tool import ShellTool


def _builtin_skills() -> dict[str, Any]:
    return {
        "analyze_repository": analyze_repository_skill,
        "generate_plan": generate_plan_skill,
        "implement_change": implement_change_skill,
        "verify_behavior": verify_behavior_skill,
        "debug_failure": debug_failure_skill,
    }


def _builtin_evaluations() -> dict[str, Any]:
    contracts = [
        repo_map_quality_contract(),
        plan_quality_contract(),
        patch_quality_contract(),
        test_pass_rate_contract(),
    ]
    return {contract.name: contract for contract in contracts}


def _build_scheduler(harness_file: Path, repo_path: Path, db_path: Path) -> HarnessScheduler:
    harness = parse_harness_yaml(harness_file)

    skill_registry = _builtin_skills()
    evaluation_contracts = _builtin_evaluations()
    evaluation_contracts.update(harness.evaluations)

    validate_harness(
        harness,
        available_skills=skill_registry.keys(),
        available_evaluations=evaluation_contracts.keys(),
    )

    run_store = RunStore(db_path)
    artifact_store = ArtifactStore(db_path)
    trace_store = TraceStore(db_path)

    tools = ToolSandbox()
    tools.register(ShellTool(workspace=repo_path))
    tools.register(FileReadTool(workspace=repo_path))
    tools.register(FileEditTool(workspace=repo_path))
    tools.register(GitTool(workspace=repo_path))
    tools.register(PythonTool(workspace=repo_path))

    executor = AgentExecutor(tools=tools, skill_registry=skill_registry)
    context_builder = ContextBuilder(
        artifact_store=artifact_store,
        token_budget=int(harness.settings.get("context_size", 4000)),
    )
    improvement_loop = ImprovementLoop(
        executor=executor,
        context_builder=context_builder,
        evaluation_engine=EvaluationEngine(),
        artifact_store=artifact_store,
        trace_store=trace_store,
        evaluation_contracts=evaluation_contracts,
    )

    return HarnessScheduler(
        harness=harness,
        run_store=run_store,
        artifact_store=artifact_store,
        trace_store=trace_store,
        improvement_loop=improvement_loop,
    )


def _command_run(args: argparse.Namespace) -> int:
    harness_file = Path(args.harness).resolve()
    repo_path = Path(args.repo).resolve()
    db_path = Path(args.db).resolve()

    scheduler = _build_scheduler(harness_file=harness_file, repo_path=repo_path, db_path=db_path)
    result = scheduler.run(user_request=args.request, repo_path=repo_path)

    print(json.dumps(
        {
            "run_id": result.run_id,
            "success": result.success,
            "feedback_cycles": result.feedback_cycles,
            "tasks": [
                {
                    "task_name": out.task_name,
                    "passed": out.passed,
                    "score": out.score,
                    "attempts": out.attempts,
                    "artifact_id": out.artifact_id,
                    "feedback": out.feedback,
                }
                for out in result.task_outcomes
            ],
        },
        indent=2,
    ))

    if args.export_traces:
        trace_store = TraceStore(db_path)
        traces = trace_store.export_run(result.run_id)
        output_path = Path(args.export_traces).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
        print(f"Exported traces to {output_path}")

    return 0 if result.success else 2


def _command_export_traces(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    trace_store = TraceStore(db_path)
    traces = trace_store.export_run(args.run_id)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"Exported {len(traces)} traces to {output_path}")
    return 0


def _command_optimize(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    trace_store = TraceStore(db_path)
    traces = trace_store.export_run(args.run_id)

    if not traces:
        print("No traces found for run_id")
        return 1

    base_quality = sum(float(t["evaluation_score"]) for t in traces) / len(traces)
    base_tokens = sum(int(t["token_usage"]) for t in traces)
    base_latency = sum(float(t["latency"]) for t in traces)

    def objective(params: dict[str, Any]) -> float:
        sampling = float(params.get("sampling_count", 1))
        retries = float(params.get("retry_limit", 1))
        context_size = float(params.get("context_size", 4000))
        reasoning_budget = float(params.get("reasoning_budget", 2048))

        scale = (sampling * 0.4) + (retries * 0.2) + (context_size / 4000.0 * 0.2) + (reasoning_budget / 2048.0 * 0.2)
        quality = min(1.0, base_quality * (1.0 + 0.05 * (sampling - 1)))
        tokens = int(base_tokens * max(0.5, scale))
        latency = base_latency * max(0.5, scale * 0.8)
        return compute_reward(artifact_quality_score=quality, token_usage=tokens, latency_seconds=latency)

    optimizer = EvolutionaryOptimizer()
    best_params, history = optimizer.optimize(
        objective=objective,
        generations=args.generations,
        population_size=args.population,
        elite_size=min(args.elite, args.population),
    )

    output = {
        "run_id": args.run_id,
        "best_parameters": best_params,
        "best_reward": max(record.reward for record in history),
        "history": [
            {
                "generation": record.generation,
                "reward": record.reward,
                "parameters": record.parameters,
            }
            for record in history
        ],
    }
    print(json.dumps(output, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness-driven coding agent runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a harness")
    run_parser.add_argument("--harness", required=True, help="Path to harness YAML")
    run_parser.add_argument("--repo", default=".", help="Repository path")
    run_parser.add_argument("--request", required=True, help="User coding request")
    run_parser.add_argument("--db", default=".agent_harness.db", help="SQLite database path")
    run_parser.add_argument("--export-traces", help="Optional path to write trace JSON")
    run_parser.set_defaults(func=_command_run)

    export_parser = subparsers.add_parser("export-traces", help="Export traces for a run")
    export_parser.add_argument("--db", default=".agent_harness.db", help="SQLite database path")
    export_parser.add_argument("--run-id", required=True, help="Run identifier")
    export_parser.add_argument("--output", required=True, help="Output JSON path")
    export_parser.set_defaults(func=_command_export_traces)

    optimize_parser = subparsers.add_parser("optimize", help="Run evolutionary harness optimization")
    optimize_parser.add_argument("--db", default=".agent_harness.db", help="SQLite database path")
    optimize_parser.add_argument("--run-id", required=True, help="Baseline run identifier")
    optimize_parser.add_argument("--generations", type=int, default=5)
    optimize_parser.add_argument("--population", type=int, default=8)
    optimize_parser.add_argument("--elite", type=int, default=2)
    optimize_parser.set_defaults(func=_command_optimize)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

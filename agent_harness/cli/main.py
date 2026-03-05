from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_harness.core.artifact_schemas import BUILTIN_ARTIFACT_SCHEMAS
from agent_harness.dsl.parser import parse_harness_yaml
from agent_harness.dsl.validator import validate_harness
from agent_harness.evaluation.engine import EvaluationEngine
from agent_harness.evaluation.llm_judge import HeuristicLLMJudge, ProviderLLMJudge
from agent_harness.evaluations.patch_quality import get_contract as patch_quality_contract
from agent_harness.evaluations.plan_quality import get_contract as plan_quality_contract
from agent_harness.evaluations.repo_map_quality import get_contract as repo_map_quality_contract
from agent_harness.evaluations.test_pass_rate import get_contract as test_pass_rate_contract
from agent_harness.providers.provider_factory import ProviderFactory
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


_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    if "=" not in text:
        return None

    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not _ENV_KEY_PATTERN.match(key):
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _load_env_file(path: Path, *, only_if_missing: bool = True) -> int:
    if not path.exists() or not path.is_file():
        return 0

    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_assignment(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if only_if_missing and os.getenv(key):
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def _load_env_from_roots(*roots: Path) -> None:
    seen: set[Path] = set()
    for root in roots:
        env_path = (root / ".env").resolve()
        if env_path in seen:
            continue
        seen.add(env_path)
        _load_env_file(env_path)


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


def _build_scheduler(
    harness_file: Path,
    repo_path: Path,
    db_path: Path,
    provider_name: str,
    model_name: str,
    temperature_override: float | None = None,
    max_tokens_override: int | None = None,
    evaluation_mode_override: str | None = None,
    max_runtime_seconds_override: int | None = None,
    task_samples_override: int | None = None,
    task_retry_override: int | None = None,
) -> HarnessScheduler:
    harness = parse_harness_yaml(harness_file)
    if provider_name == "openrouter":
        harness.settings["model"] = f"openrouter/{model_name}"
    elif provider_name == "mock":
        harness.settings["model"] = f"mock/{model_name}"
    if max_runtime_seconds_override is not None:
        harness.settings["max_runtime_seconds"] = int(max_runtime_seconds_override)
    if task_samples_override is not None:
        harness.tasks = {
            name: replace(task, samples=max(1, int(task_samples_override)))
            for name, task in harness.tasks.items()
        }
    if task_retry_override is not None:
        harness.tasks = {
            name: replace(task, retry_limit=max(1, int(task_retry_override)))
            for name, task in harness.tasks.items()
        }

    skill_registry = _builtin_skills()
    evaluation_contracts = _builtin_evaluations()
    evaluation_contracts.update(harness.evaluations)
    artifact_schemas = dict(BUILTIN_ARTIFACT_SCHEMAS)
    artifact_schemas.update(harness.artifact_schemas)

    validate_harness(
        harness,
        available_skills=skill_registry.keys(),
        available_evaluations=evaluation_contracts.keys(),
        available_artifact_schemas=artifact_schemas.keys(),
        require_provider_settings=(provider_name != "mock"),
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

    temperature = (
        float(temperature_override)
        if temperature_override is not None
        else float(harness.settings.get("temperature", 0.2))
    )
    max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(harness.settings.get("max_tokens", 2000))
    )

    if provider_name == "openrouter":
        provider_model = f"openrouter/{model_name}"
    elif provider_name == "mock":
        provider_model = model_name if model_name.startswith("mock/") else f"mock/{model_name}"
    else:
        raise ValueError(f"Unsupported provider '{provider_name}'")

    provider = ProviderFactory.create(
        model_name=provider_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if evaluation_mode_override is not None:
        evaluation_mode = str(evaluation_mode_override).strip().lower()
    elif "evaluation_mode" in harness.settings:
        evaluation_mode = str(harness.settings.get("evaluation_mode", "deterministic")).strip().lower()
    else:
        evaluation_mode = "hybrid" if provider.provider_name == "openrouter" else "deterministic"
    if evaluation_mode not in {"deterministic", "hybrid", "llm"}:
        raise ValueError("evaluation_mode must be one of: deterministic, hybrid, llm")

    llm_judge: object | None = None
    if evaluation_mode in {"hybrid", "llm"}:
        if provider.provider_name == "openrouter":
            llm_judge = ProviderLLMJudge(provider)
        else:
            llm_judge = HeuristicLLMJudge()

    executor = AgentExecutor(provider=provider, tools=tools, skill_registry=skill_registry)
    context_builder = ContextBuilder(
        artifact_store=artifact_store,
        token_budget=int(harness.settings.get("context_size", 4000)),
    )
    improvement_loop = ImprovementLoop(
        executor=executor,
        context_builder=context_builder,
        evaluation_engine=EvaluationEngine(mode=evaluation_mode, llm_judge=llm_judge),
        artifact_store=artifact_store,
        trace_store=trace_store,
        evaluation_contracts=evaluation_contracts,
        artifact_schemas=artifact_schemas,
    )

    return HarnessScheduler(
        harness=harness,
        run_store=run_store,
        artifact_store=artifact_store,
        trace_store=trace_store,
        improvement_loop=improvement_loop,
        provider_name=provider.provider_name,
        model_name=provider.model,
        temperature=temperature,
        max_tokens=max_tokens,
        evaluation_mode=evaluation_mode,
        enforce_real_execution=(provider.provider_name != "mock"),
    )


def _command_run(args: argparse.Namespace) -> int:
    harness_file = Path(args.harness).resolve()
    repo_path = Path(args.repo).resolve()
    db_path = Path(args.db).resolve()
    _load_env_from_roots(harness_file.parent, repo_path)

    if args.mock:
        provider_name = "mock"
        model_name = args.model or "default"
    else:
        provider_name = str(args.provider).strip().lower()
        if provider_name != "openrouter":
            print("Error: only --provider openrouter is currently supported for non-mock runs")
            return 2
        if not args.model:
            print("Error: --model is required unless --mock is set")
            return 2
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            print("Error: OPENROUTER_API_KEY is required for OpenRouter runs")
            return 2
        model_name = args.model

    try:
        scheduler = _build_scheduler(
            harness_file=harness_file,
            repo_path=repo_path,
            db_path=db_path,
            provider_name=provider_name,
            model_name=model_name,
            temperature_override=args.temperature,
            max_tokens_override=args.max_tokens,
            evaluation_mode_override=args.evaluation_mode,
            max_runtime_seconds_override=args.max_runtime_seconds,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    result = scheduler.run(user_request=args.request, repo_path=repo_path)

    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "success": result.success,
                "feedback_cycles": result.feedback_cycles,
                "provider": provider_name,
                "model": model_name,
                "db_path": str(db_path),
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
        )
    )

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

    if args.harness and args.repo and args.request:
        harness_file = Path(args.harness).resolve()
        repo_path = Path(args.repo).resolve()
        _load_env_from_roots(harness_file.parent, repo_path)
        base_provider = "mock" if args.mock else "openrouter"

        if not args.mock and not os.getenv("OPENROUTER_API_KEY", "").strip():
            print("Error: OPENROUTER_API_KEY is required for OpenRouter optimization runs")
            return 2

        def objective(params: dict[str, Any]) -> float:
            provider_name = base_provider
            model_name = (
                str(params.get("model"))
                if provider_name != "mock"
                else (str(params.get("model", "default")).replace("openai/", ""))
            )

            try:
                scheduler = _build_scheduler(
                    harness_file=harness_file,
                    repo_path=repo_path,
                    db_path=db_path,
                    provider_name=provider_name,
                    model_name=model_name if model_name else (args.model or "openai/gpt-4o"),
                    temperature_override=float(params.get("temperature", 0.2)),
                    max_tokens_override=int(params.get("max_tokens", 2000)),
                    evaluation_mode_override=str(params.get("evaluation_mode", "deterministic")),
                    max_runtime_seconds_override=args.max_runtime_seconds,
                    task_samples_override=int(params.get("sampling_count", 1)),
                    task_retry_override=int(params.get("retry_limit", 1)),
                )
            except Exception:
                return -10.0

            result = scheduler.run(user_request=args.request, repo_path=repo_path)
            traces = trace_store.export_run(result.run_id)
            if not traces:
                return -10.0

            quality = sum(float(t["evaluation_score"]) for t in traces) / len(traces)
            token_usage = sum(int(t["token_usage"]) for t in traces)
            latency_seconds = sum(float(t["latency"]) for t in traces)

            verification_traces = [t for t in traces if t.get("skill_name") == "verify_behavior"]
            verification_pass_rate = (
                sum(1.0 for t in verification_traces if t.get("passed")) / len(verification_traces)
                if verification_traces
                else 0.0
            )

            spec_alignment_scores: list[float] = []
            for trace in traces:
                breakdown = trace.get("evaluation_breakdown", {})
                if isinstance(breakdown, dict) and "spec_alignment" in breakdown:
                    try:
                        spec_alignment_scores.append(float(breakdown["spec_alignment"]))
                    except (TypeError, ValueError):
                        continue
            spec_compliance = (
                sum(spec_alignment_scores) / len(spec_alignment_scores) if spec_alignment_scores else 0.0
            )

            retry_attempts = sum(int(out.attempts) for out in result.task_outcomes)
            retry_penalty = max(0, retry_attempts - len(result.task_outcomes)) * 0.03

            score = compute_reward(
                artifact_quality_score=quality,
                token_usage=token_usage,
                latency_seconds=latency_seconds,
                verification_pass_rate=verification_pass_rate,
                spec_compliance=spec_compliance,
            )
            if not result.success:
                score -= 0.5
            score -= retry_penalty
            return score

        optimizer = EvolutionaryOptimizer()
        best_params, history = optimizer.optimize(
            objective=objective,
            generations=args.generations,
            population_size=args.population,
            elite_size=min(args.elite, args.population),
        )
        output = {
            "mode": "run_based",
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

    if not args.run_id:
        print("Error: provide --run-id or provide --harness --repo --request for run-based optimization")
        return 2

    traces = trace_store.export_run(args.run_id)
    if not traces:
        print("No traces found for run_id")
        return 1

    base_quality = sum(float(t["evaluation_score"]) for t in traces) / len(traces)
    base_tokens = sum(int(t["token_usage"]) for t in traces)
    base_latency = sum(float(t["latency"]) for t in traces)

    verification_traces = [t for t in traces if t.get("skill_name") == "verify_behavior"]
    verification_pass_rate = (
        sum(1.0 for t in verification_traces if t.get("passed")) / len(verification_traces)
        if verification_traces
        else 0.0
    )

    spec_alignment_scores: list[float] = []
    for trace in traces:
        breakdown = trace.get("evaluation_breakdown", {})
        if isinstance(breakdown, dict) and "spec_alignment" in breakdown:
            try:
                spec_alignment_scores.append(float(breakdown["spec_alignment"]))
            except (TypeError, ValueError):
                continue
    spec_compliance = (
        sum(spec_alignment_scores) / len(spec_alignment_scores) if spec_alignment_scores else 0.0
    )

    def objective(params: dict[str, Any]) -> float:
        sampling = float(params.get("sampling_count", 1))
        retries = float(params.get("retry_limit", 1))
        context_size = float(params.get("context_size", 4000))
        reasoning_budget = float(params.get("reasoning_budget", 2048))
        scale = (
            (sampling * 0.4)
            + (retries * 0.2)
            + (context_size / 4000.0 * 0.2)
            + (reasoning_budget / 2048.0 * 0.2)
        )
        quality = min(1.0, base_quality * (1.0 + 0.05 * (sampling - 1)))
        tokens = int(base_tokens * max(0.5, scale))
        latency = base_latency * max(0.5, scale * 0.8)
        return compute_reward(
            artifact_quality_score=quality,
            token_usage=tokens,
            latency_seconds=latency,
            verification_pass_rate=verification_pass_rate,
            spec_compliance=spec_compliance,
        )

    optimizer = EvolutionaryOptimizer()
    best_params, history = optimizer.optimize(
        objective=objective,
        generations=args.generations,
        population_size=args.population,
        elite_size=min(args.elite, args.population),
    )

    output = {
        "mode": "trace_replay",
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
    run_parser.add_argument("--provider", default="openrouter", help="Provider name (default: openrouter)")
    run_parser.add_argument("--model", help="Model ID. Required unless --mock is used")
    run_parser.add_argument("--mock", action="store_true", help="Use mock provider explicitly")
    run_parser.add_argument("--temperature", type=float, help="Override sampling temperature")
    run_parser.add_argument("--max-tokens", type=int, help="Override max output tokens")
    run_parser.add_argument("--max-runtime-seconds", type=int, help="Override max runtime seconds")
    run_parser.add_argument(
        "--evaluation-mode",
        choices=["deterministic", "hybrid", "llm"],
        help="Override evaluation mode",
    )
    run_parser.add_argument("--export-traces", help="Optional path to write trace JSON")
    run_parser.set_defaults(func=_command_run)

    export_parser = subparsers.add_parser("export-traces", help="Export traces for a run")
    export_parser.add_argument("--db", default=".agent_harness.db", help="SQLite database path")
    export_parser.add_argument("--run-id", required=True, help="Run identifier")
    export_parser.add_argument("--output", required=True, help="Output JSON path")
    export_parser.set_defaults(func=_command_export_traces)

    optimize_parser = subparsers.add_parser("optimize", help="Run evolutionary harness optimization")
    optimize_parser.add_argument("--db", default=".agent_harness.db", help="SQLite database path")
    optimize_parser.add_argument("--run-id", help="Baseline run identifier (trace replay mode)")
    optimize_parser.add_argument("--harness", help="Harness path for run-based optimization")
    optimize_parser.add_argument("--repo", help="Repo path for run-based optimization")
    optimize_parser.add_argument("--request", help="User request for run-based optimization")
    optimize_parser.add_argument("--mock", action="store_true", help="Use mock provider in run-based mode")
    optimize_parser.add_argument("--model", help="Model override for run-based mode")
    optimize_parser.add_argument("--max-runtime-seconds", type=int, help="Override runtime budget")
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
# agent_harness change: Add OAuth login to repository
from oauthlib.oauth2 import WebApplicationClient
client = WebApplicationClient(client_id)
def oauth_login():
    # Implement OAuth login logic here
    pass
from oauthlib.oauth2 import WebApplicationClient
import requests

# Initialize the OAuth client
client = WebApplicationClient(client_id)

# Function to get the authorization URL
def get_authorization_url():
    auth_url = client.prepare_request_uri(
        'https://provider.com/oauth2/auth',
        redirect_uri='https://yourapp.com/callback',
        scope=['profile', 'email']
    )
    return auth_url

# Function to fetch the token
def fetch_token(authorization_response):
    token_url = 'https://provider.com/oauth2/token'
    token = client.fetch_token(
        token_url,
        authorization_response=authorization_response,
        client_secret='your_client_secret'
    )
    return token
# agent_harness change: Add OAuth login to repository
import oauthlib.oauth2
from requests_oauthlib import OAuth2Session

# Define the OAuth endpoints and client credentials
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
authorization_base_url = 'https://provider.com/oauth2/auth'
token_url = 'https://provider.com/oauth2/token'

# Create an OAuth2 session
oauth = OAuth2Session(client_id)

# Redirect user to the provider's authorization page
authorization_url, state = oauth.authorization_url(authorization_base_url)
print('Please go to %s and authorize access.' % authorization_url)

# Get the authorization verifier code from the callback url
redirect_response = input('Paste the full redirect URL here: ')

# Fetch the access token
oauth.fetch_token(token_url, authorization_response=redirect_response, client_secret=client_secret)

# Now you can use the OAuth session to make requests
response = oauth.get('https://provider.com/api/resource')
print(response.content)

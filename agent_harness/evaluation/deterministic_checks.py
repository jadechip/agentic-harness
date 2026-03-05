from __future__ import annotations

from typing import Any, Callable


def _ratio(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def _non_empty_text(content: dict[str, Any], key: str) -> float:
    value = content.get(key, "")
    if isinstance(value, str) and value.strip():
        return 1.0
    return 0.0


def criterion_module_coverage(content: dict[str, Any]) -> float:
    modules = content.get("modules", [])
    if not isinstance(modules, list):
        return 0.0
    if not modules:
        return 0.0
    # Small repositories should still score reasonably if coverage is complete.
    richness = min(1.0, len(modules) / 20.0)
    return _ratio(0.7 + 0.3 * richness)


def criterion_entrypoint_detection(content: dict[str, Any]) -> float:
    entrypoints = content.get("entrypoints", [])
    if not isinstance(entrypoints, list):
        return 0.0
    return 1.0 if len(entrypoints) > 0 else 0.3


def criterion_dependency_accuracy(content: dict[str, Any]) -> float:
    deps = content.get("dependency_graph", {})
    if not isinstance(deps, dict):
        return 0.0
    if not deps:
        return 0.7
    nodes_with_edges = sum(1 for _, edges in deps.items() if isinstance(edges, list))
    return _ratio(nodes_with_edges / max(1, len(deps)))


def criterion_architecture_summary_quality(content: dict[str, Any]) -> float:
    summary = str(content.get("architecture_summary", ""))
    words = len(summary.split())
    return _ratio(words / 25.0)


def criterion_phase_completeness(content: dict[str, Any]) -> float:
    phases = content.get("phases", [])
    if not isinstance(phases, list):
        return 0.0
    return _ratio(len(phases) / 4.0)


def criterion_spec_alignment(content: dict[str, Any]) -> float:
    objective = str(content.get("objective", ""))
    summary = str(content.get("summary", ""))
    return 1.0 if objective and summary else 0.6 if objective or summary else 0.1


def criterion_risk_coverage(content: dict[str, Any]) -> float:
    risks = content.get("risks", [])
    if not isinstance(risks, list):
        return 0.0
    return _ratio(len(risks) / 3.0)


def criterion_file_targeting(content: dict[str, Any]) -> float:
    files = content.get("files_to_modify", [])
    if not isinstance(files, list):
        return 0.0
    if not files:
        return 0.0
    density = min(1.0, len(files) / 5.0)
    return _ratio(0.65 + 0.35 * density)


def criterion_patch_completeness(content: dict[str, Any]) -> float:
    plan = content.get("patch_plan", [])
    summary = str(content.get("summary", ""))
    if not isinstance(plan, list):
        return 0.0
    score = 0.0
    if summary.strip():
        score += 0.4
    score += min(0.6, len(plan) * 0.2)
    return _ratio(score)


def criterion_test_pass_rate(content: dict[str, Any]) -> float:
    pass_rate = content.get("pass_rate")
    if isinstance(pass_rate, (int, float)):
        return _ratio(float(pass_rate))

    tests_ran = int(content.get("tests_ran", 0))
    tests_failed = int(content.get("tests_failed", 0))
    if tests_ran <= 0:
        return 0.5
    return _ratio((tests_ran - tests_failed) / tests_ran)


def criterion_issue_reporting(content: dict[str, Any]) -> float:
    issues = content.get("issues", [])
    if not isinstance(issues, list):
        return 0.0
    return 1.0 if issues else 0.6


def criterion_non_empty(content: dict[str, Any]) -> float:
    return 1.0 if content else 0.0


CRITERION_FUNCTIONS: dict[str, Callable[[dict[str, Any]], float]] = {
    "module_coverage": criterion_module_coverage,
    "entrypoint_detection": criterion_entrypoint_detection,
    "dependency_accuracy": criterion_dependency_accuracy,
    "architecture_summary_quality": criterion_architecture_summary_quality,
    "phase_completeness": criterion_phase_completeness,
    "spec_alignment": criterion_spec_alignment,
    "risk_coverage": criterion_risk_coverage,
    "file_targeting": criterion_file_targeting,
    "patch_completeness": criterion_patch_completeness,
    "test_pass_rate": criterion_test_pass_rate,
    "issue_reporting": criterion_issue_reporting,
    "non_empty": criterion_non_empty,
}


def run_deterministic_checks(criteria: list[str], content: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for criterion in criteria:
        fn = CRITERION_FUNCTIONS.get(criterion)
        if fn is None:
            scores[criterion] = 0.5
            continue
        scores[criterion] = _ratio(fn(content))
    return scores

from agent_harness.rl.optimizer import EvolutionaryOptimizer


def test_optimizer_returns_history_and_best_vector() -> None:
    optimizer = EvolutionaryOptimizer(seed=42)

    def objective(params: dict) -> float:
        score = 0.0
        score += 1.0 if params.get("sampling_count", 1) >= 3 else 0.0
        score += 0.5 if params.get("retry_limit", 1) >= 3 else 0.0
        score -= 0.1 if params.get("context_size", 4000) > 4000 else 0.0
        return score

    best, history = optimizer.optimize(objective=objective, generations=3, population_size=6, elite_size=2)

    assert isinstance(best, dict)
    assert len(history) == 18

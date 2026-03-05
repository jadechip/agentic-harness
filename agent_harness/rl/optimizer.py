from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from .harness_space import HarnessParameterSpace


@dataclass(slots=True)
class OptimizationRecord:
    generation: int
    parameters: dict[str, Any]
    reward: float


class EvolutionaryOptimizer:
    def __init__(self, search_space: HarnessParameterSpace | None = None, seed: int = 7) -> None:
        self.search_space = search_space or HarnessParameterSpace()
        self.rng = random.Random(seed)

    def optimize(
        self,
        objective: Callable[[dict[str, Any]], float],
        generations: int = 5,
        population_size: int = 8,
        elite_size: int = 2,
    ) -> tuple[dict[str, Any], list[OptimizationRecord]]:
        if elite_size <= 0 or elite_size > population_size:
            raise ValueError("elite_size must be in [1, population_size]")

        population = [self.search_space.random_vector(self.rng) for _ in range(population_size)]
        history: list[OptimizationRecord] = []

        for generation in range(1, generations + 1):
            scored: list[tuple[dict[str, Any], float]] = []
            for vector in population:
                reward = objective(vector)
                scored.append((vector, reward))
                history.append(OptimizationRecord(generation=generation, parameters=dict(vector), reward=reward))

            scored.sort(key=lambda item: item[1], reverse=True)
            elites = [dict(vector) for vector, _ in scored[:elite_size]]

            next_population = list(elites)
            while len(next_population) < population_size:
                parent = self.rng.choice(elites)
                child = self.search_space.mutate(parent, self.rng)
                next_population.append(child)

            population = next_population

        best = max(history, key=lambda record: record.reward)
        return best.parameters, history

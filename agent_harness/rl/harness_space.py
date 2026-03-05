from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HarnessParameterSpace:
    choices: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "planner_model": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"],
            "executor_model": ["gpt-4o-mini", "gpt-4.1", "gpt-5"],
            "sampling_count": [1, 2, 3, 5],
            "retry_limit": [1, 2, 3, 4, 5],
            "context_size": [2000, 4000, 8000],
            "reasoning_budget": [1024, 2048, 4096],
        }
    )

    def random_vector(self, rng: random.Random) -> dict[str, Any]:
        return {name: rng.choice(values) for name, values in self.choices.items()}

    def mutate(self, base: dict[str, Any], rng: random.Random, mutation_rate: float = 0.3) -> dict[str, Any]:
        child = dict(base)
        for name, values in self.choices.items():
            if rng.random() < mutation_rate:
                child[name] = rng.choice(values)
        return child

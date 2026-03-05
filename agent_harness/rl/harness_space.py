from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HarnessParameterSpace:
    choices: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "model": ["openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4.1-mini"],
            "temperature": [0.0, 0.1, 0.2, 0.3],
            "max_tokens": [1000, 2000, 3000],
            "sampling_count": [1, 2, 3, 5],
            "retry_limit": [1, 2, 3, 4, 5],
            "evaluation_mode": ["deterministic", "hybrid"],
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

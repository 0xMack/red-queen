"""Training-seed strategies for simulation jobs.

Why this exists: with a fixed handful of training seeds, every genome is scored on the same few
games (same food sequences) every generation, and evolution learns *those games* rather than the
game. Measured on Snake: the champion's score on unseen games peaked around generation 25 and then
fell while its training fitness kept rising -- it finished at 11.9 on held-out games vs. 17.0 on its
5 training games, losing to a 3-line greedy baseline (18.5). Drawing fresh seeds every generation
removes anything to memorize at the same per-generation cost.

- `fixed:N`    -- seeds 0..N-1 every generation (`fixed:5` is the pre-existing behavior,
                  games.snake.BENCHMARK_SEEDS).
- `resample:N` -- N fresh seeds each generation, drawn from TRAINING_POOL with a seeded rng, so a run
                  is still reproducible.

TRAINING_POOL is disjoint from both evaluation seed ranges (jobs/evaluate.py's leaderboard
HELD_OUT_SEEDS and MONITOR_SEEDS), so neither is ever trained on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TRAINING_POOL = range(100_000, 1_000_000)


@dataclass
class SeedStrategy:
    kind: str  # "fixed" | "resample"
    count: int
    rng_seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "resample"):
            raise ValueError(f"unknown seed strategy {self.kind!r} (expected 'fixed' or 'resample')")
        if self.count < 1:
            raise ValueError("seed count must be >= 1")
        self._rng = random.Random(self.rng_seed)

    @classmethod
    def parse(cls, text: str, rng_seed: int = 0) -> SeedStrategy:
        kind, _, count = text.partition(":")
        if not count.isdigit():
            raise ValueError(f"seed strategy must look like 'fixed:5' or 'resample:5', got {text!r}")
        return cls(kind=kind, count=int(count), rng_seed=rng_seed)

    def __str__(self) -> str:
        return f"{self.kind}:{self.count}"

    @property
    def resamples(self) -> bool:
        return self.kind == "resample"

    def initial(self) -> list[int]:
        return list(range(self.count)) if self.kind == "fixed" else self.draw()

    def draw(self) -> list[int]:
        """A fresh set of training seeds (resample strategy)."""
        return self._rng.sample(TRAINING_POOL, self.count)

    def config(self) -> dict:
        """What a run records about its training seeds."""
        if self.kind == "fixed":
            return {"seed_strategy": str(self), "training_seeds": self.initial()}
        return {
            "seed_strategy": str(self),
            "training_seeds": None,  # different every generation -- no fixed set to report a gap on
            "training_seed_pool": [TRAINING_POOL.start, TRAINING_POOL.stop - 1],
            "seed_rng": self.rng_seed,
        }

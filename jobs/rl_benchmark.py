"""How fast the RL core runs natively (docs/design/0010 Phase 0) -- the numbers `/dev/rl` measures in a browser, so
a live Learn demo's shape can be chosen from what a reader's device can actually do.

Cases (keep in sync with apps/frontend/app/workers/rlBench.worker.ts):
- env steps/s: a random agent training on Snake (features.v1) -- the ceiling for tabular Q-learning's speed;
- act/s: one forward pass of an 11 -> 64 -> 64 -> 3 Q-network -- choosing one action;
- updates/s: forward + backward + Adam on a batch of 32, for three network sizes (features.v1, egocentric.v1 and
  grid-flat.v1 inputs).

Run with: uv run python jobs/rl_benchmark.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import rl
from rl import _native

NETWORKS = {
    "11-64-64-3": ([11, 64, 64, 3], ["relu", "relu", "linear"]),
    "27-128-128-3": ([27, 128, 128, 3], ["relu", "relu", "linear"]),
    "100-256-256-3": ([100, 256, 256, 3], ["relu", "relu", "linear"]),
}
BATCH = 32


def rate(work: Callable[[int], object], unit_count: int, min_seconds: float = 0.5) -> float:
    """Units per second: repeat `work(n)` with growing n until one call takes at least `min_seconds`."""
    n = 1
    while True:
        started = time.perf_counter()
        work(n)
        elapsed = time.perf_counter() - started
        if elapsed >= min_seconds:
            return n * unit_count / elapsed
        n *= 2 if elapsed < min_seconds / 4 else 1
        n = max(n, int(n * min_seconds / max(elapsed, 1e-9)))


def measure() -> dict[str, float]:
    results: dict[str, float] = {}
    trainer = rl.Trainer("random", "snake/features.v1+relative3.v1", seed=0)
    results["env steps/s (Snake, random agent)"] = rate(lambda n: trainer.advance(n * 1000), 1000)
    layers, activations = NETWORKS["11-64-64-3"]
    results["act/s (11-64-64-3)"] = rate(lambda n: _native.bench_forwards(layers, activations, n * 1000, 0), 1000)
    for name, (layers, activations) in NETWORKS.items():
        results[f"updates/s (batch {BATCH}, {name})"] = rate(
            lambda n, layers=layers, activations=activations: _native.bench_updates(layers, activations, BATCH, n, 0), 1
        )
    return results


if __name__ == "__main__":
    for name, value in measure().items():
        print(f"{name:<42} {value:>14,.0f}")

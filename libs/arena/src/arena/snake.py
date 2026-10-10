"""Snake's evaluation protocol (docs/design/0007) and playing a policy under it.

Protocol `snake.score.v2`: HELD_OUT_SEEDS (disjoint from training's games.snake.BENCHMARK_SEEDS and
`arena.seeding.TRAINING_POOL`), a 10x10 board, a MAX_STEPS cap, metric = game score (food eaten), never training
fitness. Changing any of that means a new protocol version, not an edit. v2 is v1's definition unchanged, played by the
Rust game core (docs/design/0009): its PCG32 food placement turns each seed into a different game than v1's Mersenne
Twister did, so v1 and v2 scores are not comparable.

Training imports this too: a run's held-out curve (`monitor_score` on MONITOR_SEEDS) plays by the same rules.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

from games.observation import Interface

PROTOCOL = "snake.score.v2"
HELD_OUT_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_200))
# A second, separate unseen set for *monitoring* training runs (a held-out score every N generations). Kept apart from
# HELD_OUT_SEEDS so the leaderboard's games stay untouched even if a run's monitoring curve is ever used to choose a
# champion (early stopping).
MONITOR_SEEDS: tuple[int, ...] = tuple(range(20_000, 20_100))
BOARD = {"width": 10, "height": 10}
MAX_STEPS = 1000

# A decision policy: observation -> action. Built per episode so stateful/random policies get a
# deterministic, per-seed rng.
Policy = Callable[[list[float]], Any]
PolicyFactory = Callable[[int], Policy]


def play_episode(interface: Interface, policy: Policy, seed: int) -> tuple[int, int]:
    game = interface.make_game(seed=seed, **BOARD)
    observation = game.reset()
    steps = 0
    for _ in range(MAX_STEPS):
        observation, _reward, done = game.step(policy(observation))
        steps += 1
        if done:
            break
    return game.score, steps


def monitor_score(interface: Interface, policy: Policy, seeds: Sequence[int] = MONITOR_SEEDS) -> float:
    """Mean game score over `seeds` -- the cheap held-out check a training job records per N
    generations (GenerationStats.held_out_score). Same rules as the leaderboard protocol."""
    return statistics.fmean(play_episode(interface, policy, seed)[0] for seed in seeds)


def score_stats(scores: Sequence[int]) -> dict[str, Any]:
    n = len(scores)
    mean = statistics.fmean(scores)
    stdev = statistics.stdev(scores) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "ci95": round(1.96 * stdev / n**0.5, 4) if n > 1 else 0.0,
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "zero_rate": round(sum(1 for s in scores if s == 0) / n, 4),
    }


def measure_quality(interface: Interface, factory: PolicyFactory, training_seeds: Sequence[int]) -> dict[str, Any]:
    held_out = [play_episode(interface, factory(seed), seed) for seed in HELD_OUT_SEEDS]
    scores = [score for score, _ in held_out]
    quality = score_stats(scores)
    # Every held-out game's score, in seed order -- small (one int per game), and what lets a UI
    # say "you beat this model in X% of its games" rather than only comparing to its mean.
    quality["scores"] = scores
    quality["mean_steps"] = round(statistics.fmean(steps for _, steps in held_out), 2)
    train_scores = [play_episode(interface, factory(seed), seed)[0] for seed in training_seeds]
    quality["train_mean"] = round(statistics.fmean(train_scores), 4) if train_scores else None
    quality["generalization_gap"] = (
        round(quality["train_mean"] - quality["mean"], 4) if quality["train_mean"] is not None else None
    )
    return quality

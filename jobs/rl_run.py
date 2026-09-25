"""A reinforcement-learning run, recorded to telemetry (docs/design/0010).

The agent learns in the Rust core (`libs/rl`); this script only drives it one *iteration* at a time -- a budget of
environment steps -- and records each iteration the way every other run is recorded (Decision 3):

- best/mean/worst fitness: the max/mean/min *return* of the episodes that ended during the iteration;
- diversity: the entropy of the policy the agent acts by;
- champion: a snapshot of the current policy; held-out score: the greedy policy's mean game score on the monitor
  seeds, every --held-out-every iterations and on the last one;
- extras: env_steps, episodes, mean episode length, plus whatever the algorithm reports (epsilon, td_loss, ...).

Training games are drawn from jobs/seeding.py's TRAINING_POOL, disjoint from the leaderboard's held-out games and
the monitor's. Algorithms: `random` (learns nothing: the pipeline's smoke test), `q_learning` and `sarsa` (tabular,
Phase 1; `--param n_step=3`, `--param epsilon_decay_steps=200000`, ... -- libs/rl/rust/core/src/tabular.rs), `dqn`
(Phase 2; `--param double=1 --param dueling=1`, ... -- dqn.rs).

Run with:
  uv run python jobs/rl_run.py [--algo q_learning] [--env snake/features.v1+relative3.v1 | reach1d]
                               [--iterations 20] [--steps-per-iteration 10000] [--held-out-every 5] [--rng-seed 0]
                               [--param NAME=VALUE ...] [--reward shaped|sparse] [--snapshot-every N]
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import rl
from costs import TrainingCostMeter
from evaluate import BOARD, MAX_STEPS, MONITOR_SEEDS
from run_context import recorded_run
from seeding import TRAINING_POOL
from telemetry import GenerationStats


def parse_params(pairs: list[str]) -> dict[str, float]:
    """`["alpha=0.2", "n_step=3"]` -> `{"alpha": 0.2, "n_step": 3.0}`."""
    params = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--param wants NAME=VALUE, got {pair!r}")
        params[name.strip()] = float(value)
    return params


class _Counters:
    """What TrainingCostMeter reads from a fitness evaluator (`episodes`, `steps`), kept by the RL loop."""

    def __init__(self) -> None:
        self.episodes = 0
        self.steps = 0


def main(
    algorithm: str = "random",
    env_id: str = "snake/features.v1+relative3.v1",
    iterations: int = 20,
    steps_per_iteration: int = 10_000,
    held_out_every: int = 5,
    rng_seed: int = 0,
    params: dict[str, float] | None = None,
    tags: dict[str, Any] | None = None,
    reward: str = "shaped",
    snapshot_every: int = 1,
) -> str:
    """Trains one run, records it to telemetry, and returns its run_id."""
    game = env_id.split("/")[0]
    trainer = rl.Trainer(
        algorithm,
        env_id,
        rng_seed,
        params=params or {},
        seed_pool=(TRAINING_POOL.start, TRAINING_POOL.stop),
        max_episode_steps=MAX_STEPS,
        reward=reward,
        **BOARD,
    )
    observation_size, actions, action_kind = rl.env_info(env_id, **BOARD)
    config = {
        "representation": algorithm,
        "game": game,
        "interface": env_id if game == "snake" else None,
        "paradigm": "reinforcement_learning",
        "params": params or {},
        "reward": reward,
        "observation_size": observation_size,
        "action_space": {"kind": action_kind, "values": actions},
        "iterations": iterations,
        "steps_per_iteration": steps_per_iteration,
        "max_episode_steps": MAX_STEPS,
        "board": BOARD,
        "training_seed_pool": [TRAINING_POOL.start, TRAINING_POOL.stop - 1],
        "held_out_every": held_out_every,
        "snapshot_every": snapshot_every,
        "monitor_seeds": [MONITOR_SEEDS[0], MONITOR_SEEDS[-1]],
        "rng_seed": rng_seed,
        **(tags or {}),
    }
    counters = _Counters()
    cost = TrainingCostMeter(population_size=1, fitness=counters)
    last_returns = (0.0, 0.0, 0.0)
    champion_ref = ""

    with recorded_run(config) as run:
        control = run.control_callback(cost)
        for iteration in range(iterations):
            stats = trainer.train(steps_per_iteration)
            episodes = stats["episodes"]
            counters.episodes, counters.steps = stats["total_episodes"], stats["total_steps"]
            if episodes:  # an episode can outlast an iteration: then the previous returns stand
                returns = [e["total_reward"] for e in episodes]
                last_returns = (max(returns), statistics.fmean(returns), min(returns))

            # A table snapshot is ~150 KB: `snapshot_every` > 1 stores one every N iterations (and the last), and the
            # iterations between name the latest one stored -- the policy as of that iteration's start or earlier.
            if iteration % snapshot_every == 0 or iteration == iterations - 1:
                champion_ref = f"{run.run_id}-gen{iteration}"
                run.artifacts.put_program(champion_ref, trainer.snapshot().encode("utf-8"))
            held_out_score = None
            if game == "snake" and (iteration % held_out_every == 0 or iteration == iterations - 1):
                held_out = trainer.evaluate(list(MONITOR_SEEDS), MAX_STEPS)
                held_out_score = statistics.fmean(e["score"] for e in held_out)

            best, mean, worst = last_returns
            run.metrics.record_generation(
                GenerationStats(
                    run_id=run.run_id,
                    island_id=None,
                    generation=iteration,
                    timestamp=time.time(),
                    best_fitness=best,
                    mean_fitness=mean,
                    worst_fitness=worst,
                    diversity=max(stats["entropy"], 0.0),
                    champion_ref=champion_ref,
                    held_out_score=held_out_score,
                    extras={
                        "env_steps": float(stats["total_steps"]),
                        "episodes": float(len(episodes)),
                        "mean_episode_length": statistics.fmean(e["steps"] for e in episodes) if episodes else 0.0,
                        **stats["extras"],
                    },
                )
            )
            cost.on_generation(None)
            control(None)
        history = run.set_training_summary(cost)

    last = history[-1]
    print(
        f"status=completed  {len(history)} iterations, {counters.steps:,} env steps, {counters.episodes:,} episodes; "
        f"last mean return {last.mean_fitness:.3f}, held-out score {last.held_out_score}",
        flush=True,
    )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A reinforcement-learning run (libs/rl), recorded to telemetry.")
    parser.add_argument("--algo", default="random", choices=rl.ALGORITHMS)
    parser.add_argument("--env", default="snake/features.v1+relative3.v1", help="an interface id, or reach1d")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps-per-iteration", type=int, default=10_000)
    parser.add_argument("--held-out-every", type=int, default=5)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="an algorithm parameter")
    parser.add_argument("--reward", default="shaped", choices=["shaped", "sparse"], help="Snake's reward signal")
    parser.add_argument("--experiment", default=None, help="tag recorded in the run config (kept off the leaderboard)")
    parser.add_argument(
        "--snapshot-every", type=int, default=1, help="store the policy every N iterations (and the last)"
    )
    args = parser.parse_args()
    main(
        algorithm=args.algo,
        env_id=args.env,
        iterations=args.iterations,
        steps_per_iteration=args.steps_per_iteration,
        held_out_every=args.held_out_every,
        rng_seed=args.rng_seed,
        params=parse_params(args.param),
        tags={"experiment": args.experiment} if args.experiment else None,
        reward=args.reward,
        snapshot_every=args.snapshot_every,
    )

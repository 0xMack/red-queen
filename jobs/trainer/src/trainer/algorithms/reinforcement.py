"""A single-agent reinforcement-learning run (docs/design/0010), one adapter for every `rl.Trainer` algorithm.

The agent learns in the Rust core (`libs/rl`); this drives it one *iteration* at a time -- a budget of environment
steps -- and records each iteration the way every other run is recorded (Decision 3):

- best/mean/worst fitness: the max/mean/min *return* of the episodes that ended during the iteration;
- diversity: the entropy of the policy the agent acts by;
- champion: a snapshot of the current policy; held-out score: the greedy policy's mean game score on the monitor
  seeds, every `held_out_every` iterations and on the last one (Snake only);
- extras: env_steps, episodes, mean episode length, plus whatever the algorithm reports (epsilon, td_loss, ...).

Training games are drawn from arena.seeding's TRAINING_POOL, disjoint from the leaderboard's held-out games and
the monitor's. `rl.random` learns nothing (the pipeline's smoke test); `rl.q_learning` / `rl.sarsa` are tabular
(Phase 1: `agent: {n_step: 3, epsilon_decay_steps: 200000}`, ... -- libs/rl/rust/core/src/tabular.rs); `rl.dqn` is
Phase 2 (`agent: {double: 1, dueling: 1}`, ... -- dqn.rs); `rl.reinforce` / `rl.a2c` / `rl.ppo` are Phase 3
(`agent: {baseline: 1}`, `agent: {clip: 0.1}`, ... -- pg.rs).
"""

from __future__ import annotations

import math
import statistics
import time

import rl
from arena.costs import TrainingCostMeter
from arena.seeding import TRAINING_POOL
from arena.snake import BOARD, MAX_STEPS, MONITOR_SEEDS
from jobcore import Sink, recorded_run
from jobcore.algorithms.reinforcement import RL_ALGORITHMS, RlParams
from jobcore.specs import TrainSpec
from telemetry import GenerationStats

from trainer.registry import adapter, held_out_every
from trainer.snake import DEFAULT_INTERFACE


class _Counters:
    """What TrainingCostMeter reads from a fitness evaluator (`episodes`, `steps`), kept by the RL loop."""

    def __init__(self) -> None:
        self.episodes = 0
        self.steps = 0


def train_rl(spec: TrainSpec, params: RlParams, sink: Sink) -> str:
    algorithm = spec.algorithm.removeprefix("rl.")
    env_id = (spec.interface or DEFAULT_INTERFACE) if spec.game == "snake" else spec.game
    iterations, every, steps_per_iteration = spec.budget_amount, held_out_every(spec, 5), params.steps_per_iteration
    agent = dict(params.agent)
    trainer = rl.Trainer(
        algorithm,
        env_id,
        spec.seed,
        params=agent,
        seed_pool=(TRAINING_POOL.start, TRAINING_POOL.stop),
        max_episode_steps=MAX_STEPS,
        reward=params.reward,
        **BOARD,
    )
    observation_size, actions, action_kind = rl.env_info(env_id, **BOARD)
    config = {
        "representation": algorithm,
        "game": spec.game,
        "interface": env_id if spec.game == "snake" else None,
        "paradigm": "reinforcement_learning",
        "params": agent,
        "reward": params.reward,
        "observation_size": observation_size,
        "action_space": {"kind": action_kind, "values": actions},
        "iterations": iterations,
        "steps_per_iteration": steps_per_iteration,
        "max_episode_steps": MAX_STEPS,
        "board": BOARD,
        "training_seed_pool": [TRAINING_POOL.start, TRAINING_POOL.stop - 1],
        "held_out_every": every,
        "snapshot_every": params.snapshot_every,
        "monitor_seeds": [MONITOR_SEEDS[0], MONITOR_SEEDS[-1]],
        "rng_seed": spec.seed,
        **spec.tags,
    }
    counters = _Counters()
    cost = TrainingCostMeter(population_size=1, fitness=counters)
    last_returns = (0.0, 0.0, 0.0)
    champion_ref = ""

    with recorded_run(config, sink) as run:
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
            if iteration % params.snapshot_every == 0 or iteration == iterations - 1:
                champion_ref = f"{run.run_id}-gen{iteration}"
                run.artifacts.put_program(champion_ref, trainer.snapshot().encode("utf-8"))
            held_out_score = None
            if spec.game == "snake" and (iteration % every == 0 or iteration == iterations - 1):
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
                    # A policy-gradient agent reports NaN until its first update (an iteration shorter than its first
                    # batch of episodes); that crashed the run. No update yet: no entropy to report.
                    diversity=max(stats["entropy"], 0.0) if math.isfinite(stats["entropy"]) else 0.0,
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
        run.set_training_summary(cost)
    return run.run_id


for _name in RL_ALGORITHMS:
    adapter(f"rl.{_name}", "snake", "reach1d")(train_rl)

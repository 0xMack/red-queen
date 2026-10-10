"""What the Snake training algorithms share (neuroevolution and NEAT, docs/design/0007/0008): playing a genome through
an interface, the game core's fast rollout, and the telemetry callback with its held-out curve."""

from __future__ import annotations

import time

from arena.seeding import SeedStrategy
from arena.snake import monitor_score
from evolve import GenerationSummary, SimulationFitnessEvaluator
from evolve.networks import compiled
from games.nets import native_policy
from telemetry import ArtifactStore, GenerationStats, MetricsSink

DEFAULT_INTERFACE = "snake/features.v1+relative3.v1"
BOARD = {"width": 10, "height": 10}


def make_act(interface):
    """(genome, observation) -> action through the interface's action adapter -- for
    relative3.v1, 3 outputs -> {-1, 0, 1} via argmax (games.snake.RelativeTurn3, the game core's)."""

    def act(genome, observation):
        return interface.action.decode(genome.forward(observation))

    return act


def make_rollout(interface):
    """SimulationFitnessEvaluator's `rollout`: each training episode played entirely by the game core
    (games.snake.Snake.play), the genome compiled once per evaluation. The same fitness as `make_act`'s
    Python loop, bit for bit (tests/test_snake_rollout.py), and many times faster -- the Python
    forward pass was ~87% of a generation's time."""
    if interface.action.id != "relative3.v1":
        raise ValueError(f"the game core plays relative3.v1 policies, not {interface.action.id}")

    def rollout(genome, envs, max_steps):
        policy = native_policy(compiled(genome))
        return [env.play(policy, max_steps) for env in envs]

    return rollout


def make_telemetry_callback(
    metrics: MetricsSink,
    artifacts: ArtifactStore,
    run_id: str,
    held_out: tuple[object, int, int] | None = None,
):
    """held_out = (interface, every, last_generation): also record the champion's game score on
    arena.snake's MONITOR_SEEDS every `every` generations and on the last one."""

    def on_generation(summary: GenerationSummary) -> None:
        champion_ref = f"{run_id}-gen{summary.generation}"
        artifacts.put_program(champion_ref, summary.champion.to_json().encode("utf-8"))
        held_out_score = None
        if held_out is not None:
            interface, every, last = held_out
            if summary.generation % every == 0 or summary.generation == last:
                champion = native_policy(compiled(summary.champion))
                held_out_score = monitor_score(interface, lambda o: interface.action.decode(champion.forward(o)))
        metrics.record_generation(
            GenerationStats(
                run_id=run_id,
                island_id=None,
                generation=summary.generation,
                timestamp=time.time(),
                best_fitness=summary.best_fitness,
                mean_fitness=summary.mean_fitness,
                worst_fitness=summary.worst_fitness,
                diversity=summary.diversity,
                champion_ref=champion_ref,
                held_out_score=held_out_score,
                extras=summary.extras or None,  # algorithm-specific (NEAT's species count, ...)
            )
        )

    return on_generation


def make_resample_callback(fitness: SimulationFitnessEvaluator, seeds: SeedStrategy, interface):
    """After each generation, give the next one fresh training games (resample strategies only)."""

    def on_generation(_summary: GenerationSummary) -> None:
        fitness.set_environments([interface.make_game(seed=seed, **BOARD) for seed in seeds.draw()])

    return on_generation

"""Linear GP on the fixed symbolic-regression benchmark (docs/design/0001, 0003 phase 1): the first end-to-end proof
that evolve -> metrics -> replay works, and still the reference example of wiring an algorithm to telemetry."""

from __future__ import annotations

import random
import time

from evolve import (
    GenerationSummary,
    LinearCrossoverMutation,
    LinearProgram,
    SymbolicRegressionFitness,
    TournamentSelection,
    evolve,
    random_program,
)
from jobcore import Sink, recorded_run
from jobcore.algorithms.evolution import GpParams
from jobcore.specs import TrainSpec
from telemetry import ArtifactStore, GenerationStats, MetricsSink

from trainer.registry import adapter

# The fixed benchmark problem (docs/design/0003 "fixed benchmark problems as an anchor") -- reused as-is by later
# algorithm-comparison experiments so results stay comparable. Degree 4 so a random individual can't stumble into a
# near-perfect answer by luck (x**2 could, with these ops) -- best_fitness should visibly improve, not start solved.
TARGET = lambda x: x**4 - 3 * x**2 + 2
INPUTS = [i / 5 for i in range(-5, 6)]


def serialize_program(program: LinearProgram) -> bytes:
    """Prototype serialization -- human-readable, good enough to prove the artifact round-trip works. Not meant to be
    the eventual wire format (its repr includes function addresses, so the bytes differ run to run)."""
    return repr(program).encode("utf-8")


def make_telemetry_callback(metrics: MetricsSink, artifacts: ArtifactStore, run_id: str):
    def on_generation(summary: GenerationSummary[LinearProgram]) -> None:
        champion_ref = f"{run_id}-gen{summary.generation}"
        artifacts.put_program(champion_ref, serialize_program(summary.champion))
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
            )
        )

    return on_generation


@adapter("gp", "symbolic-regression")
def train_gp(spec: TrainSpec, params: GpParams, sink: Sink) -> str:
    generations = spec.budget_amount
    config = {
        "representation": "linear_gp",
        "population_size": params.population_size,
        "num_instructions": params.num_instructions,
        "num_registers": params.num_registers,
        "generations": generations,
        "selection": "tournament(k=3)",
        "variation": "linear_crossover_mutation(rate=0.1)",
        "benchmark": "x**2",  # sic: the label every GP run has carried (the target is the quartic above)
        **spec.tags,
    }
    rng = random.Random(spec.seed)
    population = [
        random_program(params.num_instructions, params.num_registers, num_inputs=1, rng=rng)
        for _ in range(params.population_size)
    ]
    with recorded_run(config, sink) as run:
        evolve(
            population,
            fitness=SymbolicRegressionFitness(target=TARGET, inputs=INPUTS),
            selection=TournamentSelection(k=3),
            variation=LinearCrossoverMutation(mutation_rate=0.1),
            generations=generations,
            on_generation=[make_telemetry_callback(run.metrics, run.artifacts, run.run_id), run.control_callback()],
            rng=rng,
        )
        run.set_summary({"best_fitness": run.history()[-1].best_fitness})
    return run.run_id

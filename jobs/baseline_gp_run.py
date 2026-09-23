"""Baseline GA run, wired to telemetry.

This is the "integration glue" doc 0001 describes: the only place that imports both `evolve`
(zero telemetry dependency) and `telemetry` (zero algorithm dependency), adapting evolve's
telemetry-agnostic GenerationSummary into telemetry's GenerationStats. See docs/design/0001
"Decoupling from telemetry" and docs/design/0003 phase 1 ("first end-to-end validation that
evolve -> metrics -> replay actually works").

Run with: uv run python jobs/baseline_gp_run.py
"""

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
from run_context import recorded_run
from telemetry import (
    FileArtifactStore,
    FileMetricsStore,
    GenerationStats,
    SqliteRunRegistry,
)

# The fixed benchmark problem (docs/design/0003 "fixed benchmark problems as an anchor") --
# reused as-is by later algorithm-comparison experiments so results stay comparable. Degree 4 so a
# random individual can't stumble into a near-perfect answer by luck (x**2 could, with these ops)
# -- best_fitness should visibly improve across generations, not start solved.
TARGET = lambda x: x**4 - 3 * x**2 + 2
INPUTS = [i / 5 for i in range(-5, 6)]

POPULATION_SIZE = 60
NUM_INSTRUCTIONS = 12
NUM_REGISTERS = 4
GENERATIONS = 60


def serialize_program(program: LinearProgram) -> bytes:
    """Prototype serialization -- human-readable, good enough to prove the artifact round-trip
    works. Not meant to be the eventual wire format."""
    return repr(program).encode("utf-8")


def make_telemetry_callback(
    registry: SqliteRunRegistry,
    metrics: FileMetricsStore,
    artifacts: FileArtifactStore,
    run_id: str,
):
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


def main() -> None:
    config = {
        "representation": "linear_gp",
        "population_size": POPULATION_SIZE,
        "num_instructions": NUM_INSTRUCTIONS,
        "num_registers": NUM_REGISTERS,
        "generations": GENERATIONS,
        "selection": "tournament(k=3)",
        "variation": "linear_crossover_mutation(rate=0.1)",
        "benchmark": "x**2",
    }

    rng = random.Random(0)
    population = [
        random_program(NUM_INSTRUCTIONS, NUM_REGISTERS, num_inputs=1, rng=rng)
        for _ in range(POPULATION_SIZE)
    ]

    with recorded_run(config) as run:
        evolve(
            population,
            fitness=SymbolicRegressionFitness(target=TARGET, inputs=INPUTS),
            selection=TournamentSelection(k=3),
            variation=LinearCrossoverMutation(mutation_rate=0.1),
            generations=GENERATIONS,
            on_generation=[
                make_telemetry_callback(run.registry, run.metrics, run.artifacts, run.run_id),
                run.control_callback(),
            ],
            rng=rng,
        )
        final_history = run.history()
        run.set_summary({"best_fitness": final_history[-1].best_fitness})

    # Prove replay works, not just that writing worked: read everything back from storage.
    run_info = run.registry.get_run(run.run_id)
    print(f"status={run_info.status} summary={run_info.summary}")
    print(f"recorded {len(final_history)} generations")
    print(f"gen 0   best_fitness={final_history[0].best_fitness:.4f}")
    print(f"gen {final_history[-1].generation:<3} best_fitness={final_history[-1].best_fitness:.4f}")

    champion_bytes = run.artifacts.get_program(final_history[-1].champion_ref)
    print(f"final champion ({len(champion_bytes)} bytes stored):")
    print(champion_bytes.decode("utf-8"))


if __name__ == "__main__":
    main()

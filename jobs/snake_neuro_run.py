"""Neuroevolution-against-Snake run, wired to telemetry.

Same shape as baseline_gp_run.py (the "integration glue" doc 0001 describes), same benchmark setup
already validated in notebooks/0005-neuroevolution-snake.ipynb -- this is that setup, wired to
telemetry end to end instead of collecting summaries in a plain list, so its champions become real
ArtifactStore entries a frontend can load (docs/design/0005 step 6, interaction modes 2/3).

Unlike baseline_gp_run.py's serialize_program() (explicitly a prototype, never read back),
WeightVector.to_json()/from_json() is the real wire format here: apps/frontend's Pyodide bridge
loads a champion back with the exact same code, since libs/evolve is pure stdlib Python.

Run with: uv run python jobs/snake_neuro_run.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from control import make_control_callback
from evolve import (
    GaussianMutation,
    GenerationSummary,
    SimulationFitnessEvaluator,
    TournamentSelection,
    evolve,
    random_weight_vector,
)
from games.snake import benchmark_environments
from telemetry import (
    FileArtifactStore,
    FileMetricsStore,
    GenerationStats,
    SqliteRunRegistry,
)

RUN_DATA_DIR = Path(__file__).parent / "run-data"

# Matches notebooks/0005-neuroevolution-snake.ipynb exactly, so a run here is directly comparable
# to that notebook's results -- not a new experiment, just this same one wired to telemetry.
WIDTH, HEIGHT = 10, 10
LAYER_SIZES = (WIDTH * HEIGHT, 24, 3)
POPULATION_SIZE = 80
GENERATIONS = 150
MAX_STEPS = 150


def act(genome, observation) -> int:
    """3 outputs -> {-1, 0, 1} (left, straight, right) via argmax. Also the exact convention
    apps/frontend's Pyodide watch-mode bridge uses to drive a loaded champion -- see
    app/workers/snakeGame.worker.ts's snake_policy_action, which must stay in sync with this."""
    outputs = genome.forward(observation)
    best_index = max(range(len(outputs)), key=lambda i: outputs[i])
    return best_index - 1


def make_telemetry_callback(
    registry: SqliteRunRegistry,
    metrics: FileMetricsStore,
    artifacts: FileArtifactStore,
    run_id: str,
):
    def on_generation(summary: GenerationSummary) -> None:
        champion_ref = f"{run_id}-gen{summary.generation}"
        artifacts.put_program(champion_ref, summary.champion.to_json().encode("utf-8"))
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
    registry = SqliteRunRegistry(RUN_DATA_DIR / "runs.db")
    metrics = FileMetricsStore(RUN_DATA_DIR / "metrics")
    artifacts = FileArtifactStore(RUN_DATA_DIR / "artifacts")

    run_id = registry.create_run(
        config={
            "representation": "neuroevolution",
            "game": "snake",  # apps/frontend's watch page reads this to know what to render
            "layer_sizes": list(LAYER_SIZES),
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "max_steps": MAX_STEPS,
            "selection": "tournament(k=4)",
            "variation": "gaussian_mutation(sigma=0.2)",
            "benchmark": "games.snake.benchmark_environments",
        }
    )
    print(f"run_id={run_id}")

    rng = random.Random(0)
    population = [
        random_weight_vector(LAYER_SIZES, rng, scale=0.5)
        for _ in range(POPULATION_SIZE)
    ]

    evolve(
        population,
        fitness=SimulationFitnessEvaluator(
            envs=benchmark_environments(), act=act, max_steps=MAX_STEPS
        ),
        selection=TournamentSelection(k=4),
        variation=GaussianMutation(sigma=0.2),
        generations=GENERATIONS,
        on_generation=[
            make_telemetry_callback(registry, metrics, artifacts, run_id),
            make_control_callback(registry, run_id),
        ],
        rng=rng,
    )

    final_history = metrics.history(run_id)
    registry.set_summary(run_id, {"best_fitness": final_history[-1].best_fitness})
    registry.update_status(run_id, "completed")

    # Prove replay works, not just that writing worked: read everything back from storage.
    run_info = registry.get_run(run_id)
    print(f"status={run_info.status} summary={run_info.summary}")
    print(f"recorded {len(final_history)} generations")
    print(f"gen 0   best_fitness={final_history[0].best_fitness:.4f}")
    print(
        f"gen {final_history[-1].generation:<3} best_fitness={final_history[-1].best_fitness:.4f}"
    )

    champion_bytes = artifacts.get_program(final_history[-1].champion_ref)
    print(f"final champion ({len(champion_bytes)} bytes stored):")
    print(champion_bytes.decode("utf-8")[:200] + "...")


if __name__ == "__main__":
    main()

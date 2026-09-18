"""Neuroevolution-against-Snake run, wired to telemetry.

Same shape as baseline_gp_run.py (the "integration glue" doc 0001 describes). Diverges from
notebooks/0005-neuroevolution-snake.ipynb's setup in two ways, both principled, not arbitrary:

1. games.snake.Snake's observation changed from a 100-float flattened grid to 11 hand-engineered
   features (danger/heading/food-direction) -- see games/snake.py's module docstring for why. That
   notebook's own conclusion ("a representation ceiling, not a compute shortage") is exactly what
   motivated this.
2. Selection is LexicaseSelection, not TournamentSelection -- Snake's fitness is 5 genuinely
   different benchmark scenarios (games.snake.benchmark_environments()), and this repo's own
   notebooks/0001 finding is that tournament/aggregate selection can trade away per-scenario
   performance for a better average, exactly the tension lexicase exists to address
   (docs/design/0003). Worth using here, not just knowing about.

Champions become real ArtifactStore entries (WeightVector.to_json(), the real -- round-trippable --
wire format, unlike baseline_gp_run.py's prototype serialize_program()) a frontend can load
(docs/design/0005 step 6, interaction modes 2/3): apps/frontend's Pyodide bridge loads one back
with the exact same code, since libs/evolve is pure stdlib Python.

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
    LexicaseSelection,
    SimulationFitnessEvaluator,
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

# games.snake._observation() is fixed at 11 features regardless of board size (see its docstring).
# ~10x fewer weights than the old (100, 24, 3) network for the same hidden width -- faster to train
# and to run -- so this run affords a bigger population/generation budget than before for similar
# wall-clock cost.
LAYER_SIZES = (11, 16, 3)
POPULATION_SIZE = 100
GENERATIONS = 250
MAX_STEPS = 200


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
            "selection": "lexicase",
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
        selection=LexicaseSelection(),
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

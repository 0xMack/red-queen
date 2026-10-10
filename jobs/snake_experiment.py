"""A tracked comparison of neuroevolution and NEAT on Snake (docs/design/0008).

One training run is an anecdote: a different rng seed can swing a run's held-out score by several
points. So an *experiment* is a set of **arms** (algorithm variants) each trained from the same list of
rng seeds under an identical budget, every run recorded to telemetry like any other run but tagged
`config.experiment` / `config.arm` / `config.rng_seed`, then aggregated per arm.

Arms (all: `snake/features.v1+relative3.v1`, resample:5 training games, population 100, 250 generations,
so every arm gets the same number of fitness evaluations and never sees a held-out game):
- `neuro-lexicase`    fixed 11-16-3 network, Gaussian mutation, lexicase selection (the existing default)
- `neuro-tournament`  same, tournament selection -- NEAT's selection is also mean-fitness based, so this is
                      the control that separates "structure evolution" from "not using lexicase"
- `neat`              NEAT with speciation
- `neat-no-speciation` NEAT ablation: one species, everything else identical

Runs are resumable (a finished (arm, seed) is skipped), and independent arms can be run in parallel as
separate processes -- they only share the SQLite run registry, which serializes its own writes.

  uv run python jobs/snake_experiment.py run    --name NAME --arms neat,neuro-lexicase --seeds 0-4
  uv run python jobs/snake_experiment.py report --name NAME     # aggregates + writes data/experiments/NAME.json

The report scores each run's *final* champion on the 200 leaderboard games (arena.snake's
HELD_OUT_SEEDS, protocol arena.snake.PROTOCOL) -- never the best-looking generation, which would be
selecting on the test set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from typing import Any

from arena.snake import PROTOCOL, measure_quality
from evolve import network_from_json
from evolve.networks import parameter_count
from games import interfaces
from jobcore import experiments_dir, open_sink
from telemetry import FileArtifactStore, FileMetricsStore, RunInfo, SqliteRunRegistry
from trainer import train

INTERFACE = "snake/features.v1+relative3.v1"
SEED_STRATEGY = "resample:5"
HELD_OUT_EVERY = 10
GENERATIONS = 250
POPULATION_SIZE = 100  # the trainer's default for both algorithms (jobcore.algorithms.evolution)
# Where the report samples each arm's held-out monitor curve.
CURVE_GENERATIONS = (0, 50, 100, 150, 200, 249, 500, 599, 999)

# (rng_seed, generations, tags) -> run_id
Trainer = Callable[[int, int, dict[str, Any]], str]


def _arm(algorithm: str, *, generations: int | None = None, interface: str = INTERFACE, **params: Any) -> Trainer:
    """An arm as a partial TrainSpec (docs/design/0018: experiments become spec files the scheduler fans out).
    `generations` (when given) overrides the experiment-wide budget: the long-run arms *are* a bigger budget."""

    def run(rng_seed: int, default_generations: int, tags: dict[str, Any]) -> str:
        return train(
            {
                "game": "snake",
                "algorithm": algorithm,
                "interface": interface,
                "budget": {"generations": generations or default_generations},
                "seed": rng_seed,
                "held_out_every": HELD_OUT_EVERY,
                "params": {"seeds": SEED_STRATEGY, **params},
                "tags": tags,
            }
        )

    return run


ARMS: dict[str, Trainer] = {
    "neuro-lexicase": _arm("neuroevolution", selection="lexicase"),
    "neuro-tournament": _arm("neuroevolution", selection="tournament"),
    "neat": _arm("neat"),
    "neat-no-speciation": _arm("neat", neat={"speciation": False}),
    # docs/design/0008 "Longer runs": one change at a time from `neat`, then everything together. Training
    # games were capped at 200 steps while the leaderboard scores 1000-step games, so `h1000` trains on the
    # horizon it is judged on.
    "neat-long": _arm("neat", generations=1000),
    "neat-pop400": _arm("neat", population_size=400),
    "neat-h1000": _arm("neat", max_steps=1000),
    "neat-max": _arm("neat", generations=600, population_size=300, max_steps=1000, seeds="resample:10"),
    # docs/design/0007's L2 observer: the same budget as `neat-max`, only the observation differs, so a seed-for-seed
    # comparison with `neat-max` (snake-long-v1, same rng seeds) isolates the representation.
    "neat-max-ego": _arm(
        "neat",
        generations=600,
        population_size=300,
        max_steps=1000,
        seeds="resample:10",
        interface="snake/egocentric.v1+relative3.v1",
    ),
    # egocentric.v2 = egocentric.v1 + reachable space per move (docs/design/0010): what the NEAT deaths analysis said
    # the rays were missing. Same budget again, paired with `neat-max-ego` (snake-ego-v1, same rng seeds).
    "neat-max-ego2": _arm(
        "neat",
        generations=600,
        population_size=300,
        max_steps=1000,
        seeds="resample:10",
        interface="snake/egocentric.v2+relative3.v1",
    ),
}


def parse_seeds(text: str) -> list[int]:
    """`0-4` -> [0..4]; `1,3,5` -> [1, 3, 5]."""
    if "-" in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(s) for s in text.split(",")]


def experiment_runs(registry: SqliteRunRegistry, name: str) -> list[RunInfo]:
    return [r for r in registry.list_runs() if r.config.get("experiment") == name]


def run_experiment(name: str, arms: list[str], seeds: list[int], generations: int) -> None:
    registry = open_sink().registry
    for seed in seeds:
        for arm in arms:
            done = [
                r
                for r in experiment_runs(registry, name)
                if r.config.get("arm") == arm and r.config.get("rng_seed") == seed and r.status == "completed"
            ]
            if done:
                print(
                    f"skip {arm} seed {seed}: already completed as {done[0].run_id}",
                    flush=True,
                )
                continue
            print(f"=== {name} · {arm} · seed {seed}", flush=True)
            ARMS[arm](seed, generations, {"experiment": name, "arm": arm})


# --- Report ------------------------------------------------------------------------------------------


def summarize_run(run: RunInfo, metrics: FileMetricsStore, artifacts: FileArtifactStore) -> dict[str, Any]:
    history = metrics.history(run.run_id)
    champion = network_from_json(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    interface = interfaces.get(run.config["interface"])
    quality = measure_quality(
        interface,
        lambda _seed: lambda obs: interface.action.decode(champion.forward(obs)),
        training_seeds=[],
    )
    cost = (run.summary or {}).get("cost", {})
    curve = {g.generation: g.held_out_score for g in history if g.held_out_score is not None}
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "arm": run.config["arm"],
        "rng_seed": run.config["rng_seed"],
        "held_out_mean": quality["mean"],
        "held_out_ci95": quality["ci95"],
        "zero_rate": quality["zero_rate"],
        "training_best_fitness": history[-1].best_fitness,
        "parameters": parameter_count(champion),
        "curve": {str(g): curve.get(g) for g in CURVE_GENERATIONS},
        "active_s": cost.get("active_s"),
        "env_steps": cost.get("env_steps"),
        "generations": len(history),
    }
    if hasattr(champion, "complexity"):
        row["hidden_nodes"], row["connections"] = champion.complexity()
        row["species_final"] = (history[-1].extras or {}).get("species")
    return row


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def build_report(name: str) -> dict[str, Any]:
    sink = open_sink()
    registry, metrics, artifacts = sink.registry, sink.metrics, sink.artifacts
    runs = [r for r in experiment_runs(registry, name) if r.status == "completed"]
    rows = sorted(
        (summarize_run(r, metrics, artifacts) for r in runs),
        key=lambda row: (row["arm"], row["rng_seed"]),
    )
    arms: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        mine = [row for row in rows if row["arm"] == arm]
        entry: dict[str, Any] = {
            "n": len(mine),
            "seeds": [row["rng_seed"] for row in mine],
            "held_out_mean": _stats([row["held_out_mean"] for row in mine]),
            "held_out_per_seed": [row["held_out_mean"] for row in mine],
            "parameters": _stats([float(row["parameters"]) for row in mine]),
            "active_s": _stats([row["active_s"] for row in mine if row["active_s"] is not None]),
            "curve_mean": {
                str(g): round(
                    statistics.fmean(row["curve"][str(g)] for row in mine if row["curve"][str(g)] is not None),
                    2,
                )
                for g in CURVE_GENERATIONS
                if any(row["curve"][str(g)] is not None for row in mine)
            },
        }
        if "hidden_nodes" in mine[0]:
            entry["hidden_nodes"] = _stats([float(row["hidden_nodes"]) for row in mine])
            entry["connections"] = _stats([float(row["connections"]) for row in mine])
        arms[arm] = entry
    return {
        "name": name,
        "protocol": f"{PROTOCOL} (200 held-out games per run, final champion)",
        "setup": {
            "interface": INTERFACE,
            "training": SEED_STRATEGY,
            "population": POPULATION_SIZE,
        },
        "arms": arms,
        "runs": rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\n## {report['name']} -- {report['protocol']}\n")
    print("| arm | n | held-out score (mean ± sd) | min-max | params | hidden nodes | train time (s) |")
    print("|---|---|---|---|---|---|---|")
    for arm, a in report["arms"].items():
        h = a["held_out_mean"]
        hidden = f"{a['hidden_nodes']['mean']:.1f}" if "hidden_nodes" in a else "0 (fixed 16)"
        print(
            f"| {arm} | {a['n']} | {h['mean']:.2f} ± {h['sd']:.2f} | {h['min']:.1f}-{h['max']:.1f} | "
            f"{a['parameters']['mean']:.0f} | {hidden} | {a['active_s']['mean']:.0f} |"
        )
    print("\nMonitor curve (mean held-out score at generation):")
    print("| arm | " + " | ".join(f"g{g}" for g in CURVE_GENERATIONS) + " |")
    print("|---|" + "---|" * len(CURVE_GENERATIONS))
    for arm, a in report["arms"].items():
        cells = [f"{a['curve_mean'].get(str(g), float('nan')):.1f}" for g in CURVE_GENERATIONS]
        print(f"| {arm} | " + " | ".join(cells) + " |")
    print("\nPer-seed final held-out score:")
    for arm, a in report["arms"].items():
        print(f"  {arm:<20} " + "  ".join(f"{s:>6.2f}" for s in a["held_out_per_seed"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tracked NEAT vs. neuroevolution comparison on Snake.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="train the (arm, seed) runs not done yet")
    run.add_argument("--name", required=True)
    run.add_argument("--arms", default=",".join(ARMS), help=f"comma list of {sorted(ARMS)}")
    run.add_argument("--seeds", default="0-4")
    run.add_argument("--generations", type=int, default=GENERATIONS)
    report = sub.add_parser("report", help="aggregate finished runs")
    report.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        arms = args.arms.split(",")
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            sys.exit(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")
        run_experiment(args.name, arms, parse_seeds(args.seeds), args.generations)
    else:
        result = build_report(args.name)
        experiments_dir().mkdir(parents=True, exist_ok=True)
        (experiments_dir() / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print_report(result)
        print(f"\nwrote {experiments_dir() / (args.name + '.json')}")


if __name__ == "__main__":
    main()

"""Experiment reports (docs/design/0008, 0010, 0013): what an experiment's arms x seeds add up to.

Every report reads the experiment's stored spec (POST /experiments) for its arms' baselines and setup, scores each
completed run tagged with the experiment by its *final* champion -- never the best-looking generation, which would be
selecting on the test set -- and writes `<data dir>/experiments/<name>[-h2h[-full]].json` (what GET
/experiments/{name}/report serves). Ported from the old experiment scripts (snake_experiment.py, rl_experiment.py,
checkers_selfplay_experiment.py); `jobs/tests/test_golden_eval.py` holds each to what its script produced.

- `experiment.snake-report` -- each run's final champion on the 200 leaderboard games, aggregated per arm.
- `experiment.rl-report` -- the same, plus each arm against its baseline arm: an exact paired permutation test over
  the seeds (runs with the same seed are paired), so each comparison isolates the one thing the arm changes.
- `experiment.checkers-field` -- a two-player game has no held-out *score*: each run's final network, searching
  `depth` plies, plays a fixed field (material search to 2, 3 and 4 plies, and the strongest evolved evaluator if this
  machine has its run); points per game, paired against the baseline arm.
- `experiment.checkers-h2h` -- the comparison that doesn't run out of opponents (docs/design/0013): each arm's network
  plays its baseline arm's network from the same seed, head to head, in game pairs over the ballot, into one SPRT
  (`full`: the whole ballot regardless). Per arm: the verdict, the pooled Elo with its interval, each seed's.
"""

from __future__ import annotations

import itertools
import json
import statistics
from typing import Any

from arena.checkers import REPORT_SEED_BASE, play_pairing
from arena.snake import PROTOCOL, measure_quality
from arena.versus_stats import Sprt, pair_elo
from evolve import WeightVector, network_from_json
from evolve.networks import parameter_count
from games import interfaces
from games.checkers_openings import ballot
from games.checkers_strategies import STRATEGIES, evaluator
from jobcore import Sink, experiments_dir
from jobcore.specs import EvalSpec, ExperimentSpec
from modelpack import QTable, champion_parameters, load_champion
from telemetry import ArtifactStore, MetricsSource, RunInfo, RunRegistry, open_job_store

from evaluator.protocols.checkers_sprt import run_factory, sequential_match
from evaluator.registry import protocol

CURVE_GENERATIONS = (0, 50, 100, 150, 200, 249, 500, 599, 999)  # where the Snake report samples the monitor curve
FIELD = ("material-2", "material-3", "material-4")
# The versus leaderboard's best evolved evaluator (0.757 points per game there): the benchmark from the other paradigm.
EVOLVED_RUN = "802e1c7624ae402595d5384aaa1da2e8"


# --- shared ----------------------------------------------------------------------------------------------------


def stored_experiment(name: str) -> ExperimentSpec:
    try:
        return ExperimentSpec.model_validate(open_job_store().get_experiment(name).spec)
    except KeyError:
        raise ValueError(f"no stored experiment {name!r}: submit its spec first (POST /experiments)") from None


def experiment_runs(registry: RunRegistry, name: str) -> list[RunInfo]:
    return [r for r in registry.list_runs() if r.config.get("experiment") == name]


def completed_by_seed(registry: RunRegistry, experiment: str, arm: str) -> dict[int, RunInfo]:
    """An arm's completed runs, by rng seed (list_runs is newest first: the oldest wins, as it always has)."""
    return {
        r.config["rng_seed"]: r
        for r in experiment_runs(registry, experiment)
        if r.config.get("arm") == arm and r.status == "completed"
    }


def resolve_baseline(spec: ExperimentSpec, arm: str) -> tuple[str, str] | None:
    """An arm's baseline as (experiment, arm): `arm` is in this experiment, `experiment/arm` in another."""
    baseline = spec.arms[arm].baseline if arm in spec.arms else None
    if baseline is None:
        return None
    experiment, _, other = baseline.rpartition("/")
    return (experiment or spec.name, other)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def paired_permutation_p(a: list[float], b: list[float]) -> float:
    """Exact two-sided p-value for "arm a and arm b score the same", pairing runs by seed: under the null, each pair's
    difference is equally likely to have either sign, so enumerate all 2^n sign flips."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(sum(diffs))
    flips = list(itertools.product((1, -1), repeat=len(diffs)))
    extreme = sum(
        1 for signs in flips if abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-12
    )
    return extreme / len(flips)


def _versus_baseline(
    mine: list[tuple[int, float]], reference: dict[int, float], baseline: str
) -> dict[str, Any] | None:
    paired = [(score, reference[seed]) for seed, score in mine if seed in reference]
    if len(paired) < 2:
        return None
    a, b = zip(*paired, strict=True)
    return {
        "arm": baseline,
        "mean_difference": round(statistics.fmean(x - y for x, y in paired), 3),
        "paired_p": round(paired_permutation_p(list(a), list(b)), 4),
        "pairs": len(paired),
    }


def _write(name: str, report: dict[str, Any]) -> dict[str, Any]:
    experiments_dir().mkdir(parents=True, exist_ok=True)
    (experiments_dir() / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _held_out_row(run: RunInfo, champion: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    interface = interfaces.get(run.config["interface"])
    quality = measure_quality(
        interface, lambda _seed: lambda obs: interface.action.decode(champion.forward(obs)), training_seeds=[]
    )
    cost = (run.summary or {}).get("cost", {})
    return {
        "run_id": run.run_id,
        "arm": run.config["arm"],
        "rng_seed": run.config["rng_seed"],
        "held_out_mean": quality["mean"],
        "held_out_ci95": quality["ci95"],
        "zero_rate": quality["zero_rate"],
    }, cost


# --- Snake: neuroevolution vs. NEAT ------------------------------------------------------------------------------


def snake_run_row(run: RunInfo, metrics: MetricsSource, artifacts: ArtifactStore) -> dict[str, Any]:
    history = metrics.history(run.run_id)
    champion = network_from_json(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    row, cost = _held_out_row(run, champion)
    curve = {g.generation: g.held_out_score for g in history if g.held_out_score is not None}
    row |= {
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


@protocol("experiment.snake-report")
def snake_report(spec: EvalSpec, params, sink: Sink) -> dict[str, Any]:
    experiment = stored_experiment(params.experiment)
    runs = [r for r in experiment_runs(sink.registry, experiment.name) if r.status == "completed"]
    rows = sorted(
        (snake_run_row(r, sink.metrics, sink.artifacts) for r in runs), key=lambda row: (row["arm"], row["rng_seed"])
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
                    statistics.fmean(row["curve"][str(g)] for row in mine if row["curve"][str(g)] is not None), 2
                )
                for g in CURVE_GENERATIONS
                if any(row["curve"][str(g)] is not None for row in mine)
            },
        }
        if "hidden_nodes" in mine[0]:
            entry["hidden_nodes"] = _stats([float(row["hidden_nodes"]) for row in mine])
            entry["connections"] = _stats([float(row["connections"]) for row in mine])
        arms[arm] = entry
    defaults = experiment.defaults
    return _write(
        experiment.name,
        {
            "name": experiment.name,
            "protocol": f"{PROTOCOL} (200 held-out games per run, final champion)",
            "setup": {
                "interface": defaults.get("interface", "snake/features.v1+relative3.v1"),
                "training": defaults.get("params", {}).get("seeds", "fixed:5"),
                "population": defaults.get("params", {}).get("population_size", 100),
            },
            "arms": arms,
            "runs": rows,
        },
    )


# --- Snake: reinforcement learning ---------------------------------------------------------------------------------


def rl_run_row(run: RunInfo, metrics: MetricsSource, artifacts: ArtifactStore) -> dict[str, Any]:
    history = metrics.history(run.run_id)
    champion = load_champion(artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    row, cost = _held_out_row(run, champion)
    return row | {
        "env_steps": cost.get("env_steps"),
        "active_s": cost.get("active_s"),
        "parameters": champion_parameters(champion),
        "visited_states": champion.visited_states() if isinstance(champion, QTable) else None,
        # how high the network's value estimates ended up (overestimation shows here), and its loss
        "final_q_mean": (history[-1].extras or {}).get("q_mean"),
        "final_td_loss": (history[-1].extras or {}).get("td_loss"),
        # the monitor curve: held-out score (100 other unseen games) against env steps
        "curve": [
            [int((h.extras or {}).get("env_steps", 0)), h.held_out_score]
            for h in history
            if h.held_out_score is not None
        ],
    }


@protocol("experiment.rl-report")
def rl_report(spec: EvalSpec, params, sink: Sink) -> dict[str, Any]:
    experiment = stored_experiment(params.experiment)
    runs = [r for r in experiment_runs(sink.registry, experiment.name) if r.status == "completed"]
    rows = sorted(
        (rl_run_row(r, sink.metrics, sink.artifacts) for r in runs), key=lambda row: (row["arm"], row["rng_seed"])
    )
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in sorted({row["arm"] for row in rows})}
    by_seed = {arm: {row["rng_seed"]: row["held_out_mean"] for row in mine} for arm, mine in by_arm.items()}
    arms: dict[str, Any] = {}
    for arm, mine in by_arm.items():
        scores = [row["held_out_mean"] for row in mine]
        entry: dict[str, Any] = {
            "n": len(mine),
            "seeds": [row["rng_seed"] for row in mine],
            "held_out_mean": _stats(scores),
            "held_out_per_seed": scores,
            "env_steps": _stats([float(row["env_steps"]) for row in mine if row["env_steps"] is not None]),
            "active_s": _stats([row["active_s"] for row in mine if row["active_s"] is not None]),
        }
        if all(row["visited_states"] is not None for row in mine):
            entry["visited_states"] = _stats([float(row["visited_states"]) for row in mine])
        if all(row["final_q_mean"] is not None for row in mine):
            entry["final_q_mean"] = _stats([row["final_q_mean"] for row in mine])
        resolved = resolve_baseline(experiment, arm)
        baseline = resolved[1] if resolved and resolved[0] == experiment.name else None
        if baseline is not None:
            versus = _versus_baseline(
                [(r["rng_seed"], r["held_out_mean"]) for r in mine], by_seed.get(baseline, {}), baseline
            )
            if versus is not None:
                entry["vs_baseline"] = versus
        arms[arm] = entry
    trains = {arm: train for arm, _seed, train in reversed(experiment.expand())}  # each arm's first seed
    steps_per_iteration = experiment.defaults.get("params", {}).get("steps_per_iteration", 10_000)
    return _write(
        experiment.name,
        {
            "name": experiment.name,
            "protocol": f"{PROTOCOL} (200 held-out games per run, final champion)",
            "setup": {
                "steps_per_iteration": steps_per_iteration,
                "arms": {
                    arm: {
                        "interface": trains[arm].interface,
                        "baseline": experiment.arms[arm].baseline,
                        "steps": trains[arm].budget_amount * trains[arm].params.get("steps_per_iteration", 10_000),
                        "params": trains[arm].params.get("agent", {}),
                    }
                    for arm in experiment.arms
                    if arm in arms
                },
            },
            "arms": arms,
            "runs": rows,
        },
    )


# --- Checkers self-play: the fixed field, and head to head ------------------------------------------------------


def field_factories(sink: Sink) -> dict[str, Any]:
    """The fixed opponents: material search, and the best evolved evaluator (if this machine has its run)."""
    factories = {name: STRATEGIES[name] for name in FIELD}
    if EVOLVED_RUN not in {r.run_id for r in sink.registry.list_runs()}:
        return factories
    history = sink.metrics.history(EVOLVED_RUN)
    champion = network_from_json(sink.artifacts.get_program(history[-1].champion_ref).decode("utf-8"))
    depth = int(sink.registry.get_run(EVOLVED_RUN).config.get("search_depth", 1))
    factories["evolved-3ply"] = evaluator(champion.weights, champion.layer_sizes, depth)
    return factories


def score_run(snapshot: str, field_: dict[str, Any], games: int, depth: int) -> dict[str, float]:
    """Points per game against each field member, and overall."""
    network = WeightVector.from_json(snapshot)
    me = evaluator(network.weights, network.layer_sizes, depth)
    by_opponent = {}
    openings = list(ballot())
    for i, (name, opponent) in enumerate(field_.items()):
        chosen = [openings[(i * 37 + k) % len(openings)] for k in range(max(1, games // 2))]
        played = play_pairing(me, opponent, chosen, REPORT_SEED_BASE + 1000 * i)
        by_opponent[name] = statistics.fmean(points for points, _ in played)
    return {"overall": statistics.fmean(by_opponent.values()), **by_opponent}


@protocol("experiment.checkers-field")
def checkers_field(spec: EvalSpec, params, sink: Sink) -> dict[str, Any]:
    experiment = stored_experiment(params.experiment)
    name, games, depth = experiment.name, params.games, params.depth
    field_ = field_factories(sink)
    rows = []
    for run in sorted(experiment_runs(sink.registry, name), key=lambda r: (r.config["arm"], r.config["rng_seed"])):
        if run.status != "completed":
            continue
        history = sink.metrics.history(run.run_id)
        snapshot = sink.artifacts.get_program(history[-1].champion_ref).decode("utf-8")
        cost = (run.summary or {}).get("cost", {})
        rows.append(
            {
                "run_id": run.run_id,
                "arm": run.config["arm"],
                "rng_seed": run.config["rng_seed"],
                "points": score_run(snapshot, field_, games, depth),
                "games": cost.get("episodes"),
                "active_s": cost.get("active_s"),
                "curve": [[h.generation, h.held_out_score] for h in history if h.held_out_score is not None],
            }
        )
        print(f"scored {run.config['arm']} seed {run.config['rng_seed']}: {rows[-1]['points']}", flush=True)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    arms = {}
    for arm, mine in by_arm.items():
        overall = [row["points"]["overall"] for row in mine]
        entry: dict[str, Any] = {
            "n": len(mine),
            "points_per_seed": overall,
            "points": _stats(overall),
            "by_opponent": {o: round(statistics.fmean(row["points"][o] for row in mine), 3) for o in field_},
            "active_s": _stats([row["active_s"] for row in mine if row["active_s"] is not None]),
        }
        resolved = resolve_baseline(experiment, arm)
        # the field score is only computed for this experiment's runs
        baseline = resolved[1] if resolved and resolved[0] == name else None
        if baseline is not None:
            reference = {row["rng_seed"]: row["points"]["overall"] for row in by_arm.get(baseline, [])}
            versus = _versus_baseline([(r["rng_seed"], r["points"]["overall"]) for r in mine], reference, baseline)
            if versus is not None:
                entry["vs_baseline"] = versus
        arms[arm] = entry
    return _write(
        name,
        {
            "name": name,
            "protocol": f"final network at {depth}-ply vs {list(field_)}, {games} games each",
            "arms": arms,
            "runs": rows,
        },
    )


def across_seeds(per_seed: dict[int, dict[str, float]]) -> dict[str, Any]:
    """The regime-level view of a head-to-head. The pooled interval counts every game pair, so it answers "are these
    networks stronger than those" -- but a regime is a distribution over training seeds, and seeds differ. Here each
    seed's Elo difference is one sample: their mean and an exact sign-flip permutation test against 0 (with 5 seeds the
    smallest possible p is 0.0625, whatever the games say)."""
    elos = [e["elo"] for e in per_seed.values()]
    return {
        "mean": round(statistics.fmean(elos), 1),
        "sd": round(statistics.stdev(elos), 1) if len(elos) > 1 else 0.0,
        "positive": sum(1 for e in elos if e > 0),
        "seeds": len(elos),
        "sign_flip_p": round(paired_permutation_p(elos, [0.0] * len(elos)), 4) if len(elos) > 1 else None,
    }


@protocol("experiment.checkers-h2h")
def checkers_h2h(spec: EvalSpec, params, sink: Sink) -> dict[str, Any]:
    experiment = stored_experiment(params.experiment)
    name, depth, elo1, full = experiment.name, params.depth, params.elo1, params.full
    registry, metrics, artifacts = sink.registry, sink.metrics, sink.artifacts
    by_arm: dict[str, dict[int, Any]] = {}
    for run in experiment_runs(registry, name):
        if run.status == "completed":
            by_arm.setdefault(run.config["arm"], {})[run.config["rng_seed"]] = run
    out = {}
    for arm in params.arms or sorted(by_arm):
        resolved = resolve_baseline(experiment, arm)
        if resolved is None or arm not in by_arm:
            continue
        theirs = by_arm.get(resolved[1], {}) if resolved[0] == name else completed_by_seed(registry, *resolved)
        baseline = resolved[1] if resolved[0] == name else f"{resolved[0]}/{resolved[1]}"
        seeds = sorted(set(by_arm[arm]) & set(theirs))
        if not seeds:
            continue
        couples = [
            (run_factory(by_arm[arm][s], metrics, artifacts, depth), run_factory(theirs[s], metrics, artifacts, depth))
            for s in seeds
        ]
        # full: bounds no finite LLR reaches, so the whole ballot is played
        test = Sprt(elo0=0.0, elo1=elo1, alpha=1e-300, beta=1e-300) if full else Sprt(elo0=0.0, elo1=elo1)
        print(f"=== {arm} vs {baseline}: {len(seeds)} seeds, {len(ballot())} openings available", flush=True)
        result = sequential_match(couples, test)
        per_seed = {seed: pair_elo(pairs) for seed, pairs in zip(seeds, result.pop("per_couple"), strict=True)}
        if full:
            result["verdict"] = "full ballot"
            result["alpha"] = result["beta"] = None
        out[arm] = {
            "baseline": baseline,
            "seeds": seeds,
            **result,
            "per_seed": per_seed,
            "across_seeds": across_seeds(per_seed),
        }
    return _write(
        f"{name}-h2h-full" if full else f"{name}-h2h",
        {
            "name": name,
            "protocol": f"head to head at {depth}-ply, game pairs over the {len(ballot())}-opening ballot, "
            + ("the whole ballot" if full else f"SPRT elo0 0, elo1 {elo1}, alpha = beta = 0.05"),
            "arms": out,
        },
    )

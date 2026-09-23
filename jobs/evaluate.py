"""Leaderboard evaluation (docs/design/0007): run every entrant under a fixed, versioned protocol and
record a vector of measurements -- quality, inference cost, training cost -- not one score.

Entrants for Snake today:
- every Snake run's final champion, under the interface recorded in its config (see
  jobs/backfill_interfaces.py for pre-0007 runs). Once published (jobs/publish_models.py), a champion
  is evaluated *as its model package*, run by ONNX Runtime -- the exact bytes a visitor's browser
  downloads (docs/design/0009) -- and each package variant that plays differently from the champion
  is an entrant of its own (`run:<id>@fp32`);
- fixed baselines (random, greedy) -- always included, since a ranking says nothing without them.

Protocol `snake.score.v2`: HELD_OUT_SEEDS (disjoint from training's games.snake.BENCHMARK_SEEDS),
10x10 board, MAX_STEPS cap, metric = game score (food eaten), never training fitness. Changing any of
that means a new protocol version, not an edit. v2 is v1's definition unchanged, played by the Rust
game core (docs/design/0009): its PCG32 food placement turns each seed into a different game than
v1's Mersenne Twister did, so v1 and v2 scores are not comparable.

Run with: uv run python jobs/evaluate.py
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

from costs import hardware_fingerprint
from evolve import network_from_json
from evolve.networks import describe, parameter_count
from games import baselines, interfaces
from games.observation import Interface
from games.snake import BENCHMARK_SEEDS
from modelpack import LocalModelStore, ModelStore, PackagedModel
from run_context import RUN_DATA_DIR, TelemetryStores
from telemetry import (
    EvaluationRecord,
    FileArtifactStore,
    FileMetricsStore,
    RunInfo,
    SqliteEvaluationStore,
    SqliteRunRegistry,
)

PROTOCOL = "snake.score.v2"
HELD_OUT_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_200))
# A second, separate unseen set for *monitoring* training runs (snake_neuro_run.py's held-out score
# every N generations). Kept apart from HELD_OUT_SEEDS so the leaderboard's games stay untouched even
# if a run's monitoring curve is ever used to choose a champion (early stopping).
MONITOR_SEEDS: tuple[int, ...] = tuple(range(20_000, 20_100))
BOARD = {"width": 10, "height": 10}
MAX_STEPS = 1000

# A decision policy: observation -> action. Built per episode so stateful/random policies get a
# deterministic, per-seed rng.
Policy = Callable[[list[float]], Any]
PolicyFactory = Callable[[int], Policy]


# --- Baselines ---------------------------------------------------------------------------------------

# Defined in games.baselines (not here) so the browser can run the same code to let visitors watch
# them play. The two factories are re-exported for tests.
FEATURES = "snake/features.v1+relative3.v1"
random_policy_factory = baselines.get("snake", "random").factory
greedy_policy_factory = baselines.get("snake", "greedy").factory


def baseline_entrants(game: str) -> list[dict[str, Any]]:
    return [
        {
            "entrant_id": b.entrant_id,
            "label": b.label,
            "interface": b.interface,
            "factory": b.factory,
            "model": b.description,
        }
        for b in baselines.for_game(game)
    ]


# --- Measurement -------------------------------------------------------------------------------------


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


def measure_inference(
    interface: Interface, factory: PolicyFactory, repeats: int = 5, decisions: int = 400
) -> dict[str, Any]:
    """Median-of-repeats per-decision latency, split into encoding the game (observer) and deciding
    (policy), after a warm-up. Uses real game states from one episode, replayed."""
    game = interface.make_game(seed=HELD_OUT_SEEDS[0], **BOARD)
    game.reset()
    policy = factory(HELD_OUT_SEEDS[0])
    observations: list[list[float]] = []
    for _ in range(decisions):
        observation = interface.observer.encode(game)
        observations.append(observation)
        _, _, done = game.step(policy(observation))
        if done:
            game.reset()

    def per_call_us(fn: Callable[[], None]) -> float:
        fn()  # warm-up
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) / len(observations) * 1e6)
        return statistics.median(samples)

    encode_us = per_call_us(lambda: [interface.observer.encode(game) for _ in observations])
    decide_us = per_call_us(lambda: [policy(o) for o in observations])
    return {
        "encode_us": round(encode_us, 3),
        "decide_us": round(decide_us, 3),
        "total_us": round(encode_us + decide_us, 3),
    }


# --- Entrants from runs -----------------------------------------------------------------------------


def training_cost(run: RunInfo, metrics: FileMetricsStore) -> dict[str, Any]:
    """The run's measured cost block (jobs/costs.py) if it has one; otherwise an estimate from its
    config + metrics timestamps, labelled as such (docs/design/0007: legacy runs)."""
    measured = (run.summary or {}).get("cost")
    if measured:
        return dict(measured)
    history = metrics.history(run.run_id)
    population = run.config.get("population_size")
    seeds = run.config.get("training_seeds") or list(BENCHMARK_SEEDS)  # size of a generation's evaluation
    evaluations = population * len(history) if population else None
    return {
        "measured": False,
        "generations": len(history),
        "fitness_evaluations": evaluations,
        "episodes": evaluations * len(seeds) if evaluations else None,
        "env_steps": None,
        # First-to-last generation timestamps: may include paused time, and nothing recorded CPU,
        # memory, or hardware for these runs.
        "active_s": round(history[-1].timestamp - history[0].timestamp, 3) if len(history) > 1 else None,
        "cpu_s": None,
        "peak_rss_bytes": None,
        "hardware": None,
    }


def model_shape(network, algorithm: str, selection: str | None) -> dict[str, Any]:
    """algorithm + size facts: hidden nodes/connections for an evolved graph, layer sizes for a fixed MLP."""
    from evolve.neat import NeatGenome

    if isinstance(network, NeatGenome):
        hidden, connections = network.complexity()
        return {
            "algorithm": algorithm,
            "selection": selection,
            "hidden_nodes": hidden,
            "connections": connections,
            "inputs": network.num_inputs,
            "outputs": network.num_outputs,
        }
    return {"algorithm": algorithm, "selection": selection, "layer_sizes": list(network.layer_sizes)}


def champion_entrants(
    registry: SqliteRunRegistry, metrics: FileMetricsStore, artifacts: FileArtifactStore, game: str
) -> list[dict[str, Any]]:
    entrants = []
    for run in registry.list_runs():
        interface_id = run.config.get("interface")
        if run.config.get("game") != game or not interface_id:
            continue
        # Only finished runs: a still-training run's champion isn't its final one (re-run this job
        # once it completes).
        if run.status in ("running", "paused"):
            continue
        # Repeated runs of a comparison (jobs/snake_experiment.py) are aggregated there, not ranked
        # one by one: twenty seeds of four arms would bury every other entrant.
        if run.config.get("experiment"):
            continue
        # RL champions (docs/design/0010) need modelpack's loaders for their formats, which arrive with the first
        # learning algorithm (Phase 1); until then an RL run is never an entrant.
        if run.config.get("paradigm") == "reinforcement_learning":
            continue
        history = metrics.history(run.run_id)
        if not history:
            continue
        champion_ref = history[-1].champion_ref
        raw = artifacts.get_program(champion_ref)
        champion = network_from_json(raw.decode("utf-8"))  # WeightVector or NeatGenome
        interface = interfaces.get(interface_id)

        def factory(_seed: int, champion=champion, interface=interface) -> Policy:
            return lambda observation: interface.action.decode(champion.forward(observation))

        selection = run.config.get("selection")
        network = describe(champion)
        representation = run.config.get("representation", "unknown")
        kind = "NEAT" if representation == "neat" else "Neuroevolution"
        label = f"{kind} {network}" + (f" · {selection}" if selection else "")
        entrants.append(
            {
                "entrant_id": f"run:{run.run_id}",
                "label": label,
                "interface": interface_id,
                "factory": factory,
                "run": run,
                "champion_ref": champion_ref,
                "model": (
                    f"evolved graph {network}, tanh (neat)"
                    if representation == "neat"
                    else f"MLP {network}, tanh ({representation})"
                ),
                "parameters": parameter_count(champion),
                # Structured, so every UI names a model the same way (apps/frontend utils/modelLabel.ts)
                # instead of parsing `label`.
                "shape": model_shape(champion, kind, selection),
                "artifact_bytes": len(raw),
                # None for resampled runs (no fixed training set, so no train-vs-held-out gap to show).
                "training_seeds": run.config.get("training_seeds", list(BENCHMARK_SEEDS)) or [],
                "note": run.config.get("note"),
            }
        )
    return entrants


def packaged_entrants(entrants: list[dict[str, Any]], store: ModelStore, game: str) -> list[dict[str, Any]]:
    """Swap each published champion for its model package (docs/design/0009): the entrant keeps its
    id but decides through ONNX Runtime running the package's first exact variant, and every other
    catalog entry for the same run (a variant that plays differently) becomes an extra entrant. Champions
    without a package -- or with no variant that plays exactly like them -- also stay ranked as
    themselves (pure-Python forward pass)."""
    catalog = store.catalog(game)
    result = []
    for entrant in entrants:
        run_id = entrant["run"].run_id
        entries = [e for e in catalog.entries if e.run_id == run_id and e.champion_ref == entrant["champion_ref"]]
        if not any(e.entrant_id == entrant["entrant_id"] for e in entries):
            # Unpublished, or no variant plays exactly like the champion: rank the champion itself.
            result.append({**entrant, "runtime": "python"})
        interface = interfaces.get(entrant["interface"])
        for entry in entries:
            manifest = store.manifest(entry.package_id)
            variant = entry.variants[0]
            model = PackagedModel(manifest, variant, store.read_blob)

            def factory(_seed: int, model=model, interface=interface) -> Policy:
                return lambda observation: interface.action.decode(model.forward(observation))

            result.append(
                {
                    **entrant,
                    "entrant_id": entry.entrant_id,
                    "label": entry.label,
                    "factory": factory,
                    "artifact_bytes": manifest.variant(variant).requirements.download_bytes,
                    "runtime": f"onnxruntime-cpu {variant}",
                    "package_id": entry.package_id,
                    "variant": variant,
                    "variants": entry.variants,
                }
            )
    return result


def evaluate_entrant(
    entrant: dict[str, Any], metrics: FileMetricsStore | None, hardware: dict[str, Any]
) -> EvaluationRecord:
    interface = interfaces.get(entrant["interface"])
    run: RunInfo | None = entrant.get("run")
    quality = measure_quality(interface, entrant["factory"], entrant.get("training_seeds", BENCHMARK_SEEDS))
    inference = measure_inference(interface, entrant["factory"])
    inference["parameters"] = entrant.get("parameters", 0)
    inference["artifact_bytes"] = entrant.get("artifact_bytes", 0)
    inference["runtime"] = entrant.get("runtime", "python")
    training = training_cost(run, metrics) if run and metrics else {"measured": True, "none": True}
    level = interface.observer.level
    return EvaluationRecord(
        game=interface.game,
        protocol=PROTOCOL,
        entrant_id=entrant["entrant_id"],
        entrant_kind="champion" if run else "baseline",
        label=entrant["label"],
        interface=interface.id,
        run_id=run.run_id if run else None,
        champion_ref=entrant.get("champion_ref"),
        created_at=time.time(),
        metrics={
            "quality": quality,
            "inference": inference,
            "training": training,
            "model": {
                "description": entrant["model"],
                "shape": entrant.get("shape"),
                "observer_level": level,
                "note": entrant.get("note"),
                "package_id": entrant.get("package_id"),
                "variant": entrant.get("variant"),
                "variants": entrant.get("variants"),
            },
            "protocol": {
                "held_out_seeds": [HELD_OUT_SEEDS[0], HELD_OUT_SEEDS[-1]],
                "episodes": len(HELD_OUT_SEEDS),
                "max_steps": MAX_STEPS,
                "board": BOARD,
                "metric": "score (food eaten)",
            },
        },
        hardware=hardware,
    )


def main() -> None:
    registry, metrics, artifacts = TelemetryStores.open()
    store = SqliteEvaluationStore(RUN_DATA_DIR / "evaluations.db")
    hardware = hardware_fingerprint()

    models = LocalModelStore(RUN_DATA_DIR / "models")
    champions = packaged_entrants(champion_entrants(registry, metrics, artifacts, "snake"), models, "snake")
    entrants = [*baseline_entrants("snake"), *champions]
    for entrant in entrants:
        record = evaluate_entrant(entrant, metrics, hardware)
        store.put(record)
        q, inf = record.metrics["quality"], record.metrics["inference"]
        print(
            f"{record.label:<40} {record.interface:<34} mean {q['mean']:>6.2f} ±{q['ci95']:<5} "
            f"train {q['train_mean']!s:>6}  {inf['total_us']:>7.2f} µs/decision"
        )


if __name__ == "__main__":
    main()

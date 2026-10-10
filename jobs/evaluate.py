"""Leaderboard evaluation (docs/design/0007): run every entrant under a fixed, versioned protocol and
record a vector of measurements -- quality, inference cost, training cost -- not one score.

Entrants for Snake today:
- every Snake run's final champion, under the interface recorded in its config (see
  pre-0007 runs, which had it backfilled once). Once published (jobs/publish_models.py), a champion
  is evaluated *as its model package*, run by ONNX Runtime -- the exact bytes a visitor's browser
  downloads (docs/design/0009) -- and each package variant that plays differently from the champion
  is an entrant of its own (`run:<id>@fp32`);
- fixed baselines (random, greedy) -- always included, since a ranking says nothing without them.

The protocol itself (`snake.score.v2`: seeds, board, step cap, scoring) is defined in `arena.snake`; this job
enumerates the entrants, measures them under it, and writes the records.

Run with: uv run python jobs/evaluate.py
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

from arena.costs import hardware_fingerprint
from arena.snake import BOARD, HELD_OUT_SEEDS, MAX_STEPS, PROTOCOL, Policy, PolicyFactory, measure_quality
from games import baselines, interfaces
from games.observation import Interface
from games.snake import BENCHMARK_SEEDS
from jobcore import open_sink
from modelpack import (
    LocalModelStore,
    MlpPolicy,
    ModelStore,
    PackagedModel,
    QTable,
    UnsupportedChampion,
    champion_parameters,
    describe_champion,
    load_champion,
)
from telemetry import (
    EvaluationRecord,
    FileArtifactStore,
    FileMetricsStore,
    RunInfo,
    SqliteRunRegistry,
    models_dir,
)

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
    """The run's measured cost block (arena.costs) if it has one; otherwise an estimate from its
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


# How a run's `representation` is named on a leaderboard (the frontend's runMeta REPRESENTATION_LABELS agree).
ALGORITHM_LABELS = {
    "neat": "NEAT",
    "neuroevolution": "Neuroevolution",
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "a2c": "A2C",
    "ppo": "PPO",
}


def model_shape(network, algorithm: str, selection: str | None) -> dict[str, Any]:
    """algorithm + size facts: hidden nodes/connections for an evolved graph, layer sizes for a fixed MLP, states
    (visited of all) for a table."""
    from evolve.neat import NeatGenome

    if isinstance(network, QTable):
        return {
            "algorithm": algorithm,
            "selection": selection,
            "table_states": network.states,
            "visited_states": network.visited_states(),
            "actions": network.num_actions,
        }
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
        # Only runs that completed: a still-training run's champion isn't its final one (re-run this job once it
        # completes), and a failed or stopped run's is whatever it had when it stopped.
        if run.status != "completed":
            continue
        # Repeated runs of a comparison (jobs/snake_experiment.py) are aggregated there, not ranked
        # one by one: twenty seeds of four arms would bury every other entrant.
        if run.config.get("experiment"):
            continue
        history = metrics.history(run.run_id)
        if not history:
            continue
        champion_ref = history[-1].champion_ref
        raw = artifacts.get_program(champion_ref)
        try:
            champion = load_champion(raw.decode("utf-8"))  # an evolved network or an RL table (modelpack.champions)
        except UnsupportedChampion:
            continue  # nothing to rank: the RL pipeline's random agent (the random baseline already is)
        interface = interfaces.get(interface_id)

        def factory(_seed: int, champion=champion, interface=interface) -> Policy:
            return lambda observation: interface.action.decode(champion.forward(observation))

        selection = run.config.get("selection")
        network = describe_champion(champion)
        representation = run.config.get("representation", "unknown")
        kind = ALGORITHM_LABELS.get(representation, representation.replace("_", " ").capitalize())
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
                    else f"{network} ({representation})"
                    if isinstance(champion, QTable)
                    else f"MLP {network}, {'/'.join(sorted(set(champion.activations[:-1])))} ({representation})"
                    if isinstance(champion, MlpPolicy)
                    else f"MLP {network}, tanh ({representation})"
                ),
                "parameters": champion_parameters(champion),
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
    sink = open_sink()
    registry, metrics, artifacts = sink.registry, sink.metrics, sink.artifacts
    store = sink.evaluations
    hardware = hardware_fingerprint()

    models = LocalModelStore(models_dir())
    champions = packaged_entrants(champion_entrants(registry, metrics, artifacts, "snake"), models, "snake")
    entrants = [*baseline_entrants("snake"), *champions]
    evaluated = set()
    for entrant in entrants:
        record = evaluate_entrant(entrant, metrics, hardware)
        store.put(record)
        evaluated.add(record.entrant_id)
        q, inf = record.metrics["quality"], record.metrics["inference"]
        print(
            f"{record.label:<40} {record.interface:<34} mean {q['mean']:>6.2f} ±{q['ci95']:<5} "
            f"train {q['train_mean']!s:>6}  {inf['total_us']:>7.2f} µs/decision"
        )
    # An entrant that no longer qualifies (its run failed, or was deleted) leaves the leaderboard, not a stale row.
    for entrant_id in store.prune("snake", PROTOCOL, evaluated):
        print(f"removed {entrant_id}: no longer an entrant")


if __name__ == "__main__":
    main()

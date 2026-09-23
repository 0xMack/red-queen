"""Search distillation for Checkers: evolve a position evaluator to predict what a *deeper* search says, then play it.

Why not win/loss fitness (jobs/checkers_neuro_run.py)? Against deterministic opponents a genome's fitness is a
handful of games, so selection has almost no signal -- measured: 300 generations of best fitness that never rose
while held-out play drifted *down* (docs/design/0006 follow-up, docs/CODING_GUIDELINES.md). Here the fitness is dense
and noise-free instead: a pool of positions from real (partly random) games, each labelled with the value a
`material-K` alpha-beta search gives it (`--label-depth`, default 6), and a genome is scored by how well its
network -- one forward pass, no search -- predicts that value. A player is then that evaluator searched `--depth`
plies (default 3): a good evaluator makes a shallow search behave like a deeper one, so a depth-3 player can beat a
plain `material-4`. The champion is scored, as in the other Checkers jobs, on games it never trained on.

Every generation fits a fresh sample from the labelled pool (`--sample`), split into groups so lexicase selection
still has cases to work with; the pool is cached under run-data/ so runs are comparable.

  uv run python jobs/checkers_distill_run.py [--generations 300] [--population 100] [--hidden 16] [--depth 3]
                                             [--label-depth 6] [--workers 8] [--experiment NAME --arm ARM]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from checkers_training import INPUTS, MAX_MOVES, MONITOR_GAMES, make_telemetry_callback
from control import make_control_callback
from costs import TrainingCostMeter
from evolve import GaussianMutation, TournamentSelection, WeightVector, evolve, random_weight_vector
from games import _native
from games.checkers import Checkers
from parallel import ProcessPoolEvaluator
from telemetry import FileArtifactStore, FileMetricsStore, SqliteRunRegistry

RUN_DATA_DIR = Path(__file__).parent / "run-data"
POOL_DIR = RUN_DATA_DIR / "distill"
LABEL_SCALE = 4.0  # material units at which a label reaches ~76% of the network's range (tanh)
GROUPS = 10  # fitness cases per genome: the sample split into this many groups (negative mean squared error each)
DEFAULT_OPPONENTS = ("material-3", "material-4")


def _strategy(name: str, seed: int):
    return _native.CheckersStrategy(name, seed, None, None, 1, None)


def _label(value: float) -> float:
    return math.tanh(max(-50.0, min(50.0, value)) / LABEL_SCALE)  # a decided game reads as +-1


ROLLOUTS = 3  # playouts per position for the `rollout` / `blend` labels
ROLLOUT_PLAYER = "material-3"  # both sides of every playout


def _playout(env: Checkers, seed: int) -> float:
    """The result of playing `env` out with `ROLLOUT_PLAYER` on both sides (random tie-breaks, seeded), from the
    view of the side to move: +1 win, -1 loss, 0 a draw (the ply cap, or nothing decided)."""
    me = env.current_player()
    players = {0: _strategy(ROLLOUT_PLAYER, seed), 1: _strategy(ROLLOUT_PLAYER, seed + 1)}
    for _ in range(MAX_MOVES):
        if env.winner() is not None or not env.legal_moves():
            break
        env.step(env.choose(players[env.current_player()]))
    winner = env.winner()
    return 0.0 if winner is None else 1.0 if winner == me else -1.0


def _games_chunk(args: tuple[int, int, int, str]) -> list[tuple[list[float], float]]:
    """`games` games' positions with their labels. Play is a mix of material-1 and random moves (so the pool covers
    tactics a careful player would avoid as well as ones it would reach). Labels (`kind`):
    `search` -- the side to move's value under a `label_depth` material search (nothing beyond material);
    `rollout` -- what actually happens when strong (material-3) players finish the game from there, averaged over
    ROLLOUTS playouts: information material search doesn't have (position, tempo, structure), noisy per position
    but unbiased; `blend` -- their mean."""
    import copy

    seed, games, label_depth, kind = args
    rng = random.Random(seed)
    labeller = _strategy(f"material-{label_depth}", seed)
    mover = _strategy("material-1", seed + 1)
    out: list[tuple[list[float], float]] = []
    for _ in range(games):
        env = Checkers()
        for _ply in range(MAX_MOVES):
            moves = env.legal_moves()
            if not moves or env.winner() is not None:
                break
            label = 0.0
            if kind in ("search", "blend"):
                label += _label(max(labeller.scores(env._core)))
            if kind in ("rollout", "blend"):
                label += sum(_playout(copy.copy(env), rng.getrandbits(40)) for _ in range(ROLLOUTS)) / ROLLOUTS
            if kind == "blend":
                label /= 2
            out.append((env._observation(), label))
            env.step(rng.choice(moves) if rng.random() < 0.3 else env.choose(mover))
    return out


def build_pool(label_depth: int, games: int, workers: int, seed: int = 0, kind: str = "search") -> tuple[list[list[float]], list[float]]:
    """The labelled positions, generated once and cached (the labels cost a deep search each)."""
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    path = POOL_DIR / f"pool-{kind}-k{label_depth}-g{games}-s{seed}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data["observations"], data["labels"]
    per = max(1, games // (workers * 4))
    tasks = [(seed * 1_000_003 + i, per, label_depth, kind) for i in range(0, games, per)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for chunk in pool.map(_games_chunk, tasks) for row in chunk]
    observations, labels = [r[0] for r in rows], [r[1] for r in rows]
    path.write_text(json.dumps({"observations": observations, "labels": labels}))
    return observations, labels


class DistillFitness:
    """`evaluate(genome)` -> `GROUPS` negative mean squared errors of the network against the labels of this
    generation's sample. `on_generation` moves on to a fresh sample; within a generation every genome is scored on
    the same positions, so they compare fairly."""

    def __init__(self, observations: list[list[float]], labels: list[float], sample: int, seed: int = 0):
        self._pool = (observations, labels)
        self._sample = sample
        self._seed = seed
        self._draw(0)

    def _draw(self, generation: int) -> None:
        observations, labels = self._pool
        rng = random.Random(self._seed * 7_919 + generation)
        picked = rng.sample(range(len(labels)), self._sample)
        self._xs = [observations[i] for i in picked]
        self._ys = [labels[i] for i in picked]

    def __getstate__(self) -> dict[str, Any]:
        # Workers get this generation's sample, not the whole pool (it is ~100k positions).
        return {"_xs": self._xs, "_ys": self._ys}

    def on_generation(self, summary: Any) -> None:
        self._draw(summary.generation + 1)

    def evaluate(self, genome: WeightVector) -> list[float]:
        group = len(self._ys) // GROUPS
        errors = []
        for g in range(GROUPS):
            total = 0.0
            for x, y in zip(self._xs[g * group : (g + 1) * group], self._ys[g * group : (g + 1) * group], strict=True):
                total += (genome.forward(x)[0] - y) ** 2
            errors.append(-total / group)
        return errors


def main(
    generations: int = 300,
    population_size: int = 100,
    hidden: int = 16,
    depth: int = 3,
    label_depth: int = 6,
    pool_games: int = 2000,
    sample: int = 600,
    sigma: float = 0.1,
    mutation_rate: float = 0.1,
    opponents: tuple[str, ...] = DEFAULT_OPPONENTS,
    rng_seed: int = 0,
    held_out_every: int = 10,
    workers: int = 1,
    tags: dict[str, Any] | None = None,
    label_kind: str = "search",
) -> str:
    layer_sizes = (INPUTS, hidden, 1)
    registry = SqliteRunRegistry(RUN_DATA_DIR / "runs.db")
    metrics = FileMetricsStore(RUN_DATA_DIR / "metrics")
    artifacts = FileArtifactStore(RUN_DATA_DIR / "artifacts")
    observations, labels = build_pool(label_depth, pool_games, max(workers, 1), kind=label_kind)

    run_id = registry.create_run(
        config={
            "representation": "neuroevolution",
            "game": "checkers",
            "interface": "checkers/board32.v1+evaluate1ply.v1",
            "search_depth": depth,
            "layer_sizes": list(layer_sizes),
            "population_size": population_size,
            "generations": generations,
            "max_moves": MAX_MOVES,
            "opponents": list(opponents),  # what the held-out monitor plays; nothing here trains against them
            "selection": "tournament(k=4)",
            "variation": f"gaussian_mutation(sigma={sigma}, rate={mutation_rate})",
            "fitness": {
                "search": f"distillation: predict a material-{label_depth} search's value of sampled positions",
                "rollout": f"regression on playout outcomes ({ROLLOUTS} x {ROLLOUT_PLAYER} self-play from each sampled position)",
                "blend": f"half material-{label_depth} search value, half {ROLLOUTS}-playout outcome",
            }[label_kind],
            "label_kind": label_kind,
            "label_depth": label_depth,
            "pool_positions": len(labels),
            "sample_per_generation": sample,
            "held_out_every": held_out_every,
            "rng_seed": rng_seed,
            **(tags or {}),
        }
    )
    print(f"run_id={run_id}  pool={len(labels)} positions", flush=True)

    rng = random.Random(rng_seed)
    population = [random_weight_vector(layer_sizes, rng, scale=0.5) for _ in range(population_size)]
    fitness_inner = DistillFitness(observations, labels, sample, rng_seed)
    fitness = ProcessPoolEvaluator(fitness_inner, workers) if workers > 1 else fitness_inner
    cost = TrainingCostMeter(population_size=population_size, fitness=fitness)
    try:
        evolve(
            population,
            fitness=fitness,
            selection=TournamentSelection(k=4),
            variation=GaussianMutation(sigma=sigma, rate=mutation_rate),
            generations=generations,
            on_generation=[
                make_telemetry_callback(metrics, artifacts, run_id, opponents, depth, held_out_every, generations - 1, MONITOR_GAMES),
                fitness_inner.on_generation,
                cost.on_generation,
                cost.excluding_pauses(make_control_callback(registry, run_id)),
            ],
            elitism=2,
            rng=rng,
        )
    except BaseException:
        registry.update_status(run_id, "failed")
        raise
    history = metrics.history(run_id)
    registry.set_summary(
        run_id,
        {"best_fitness": history[-1].best_fitness, "held_out_score": history[-1].held_out_score, "cost": cost.summary()},
    )
    registry.update_status(run_id, "completed")
    print(f"status=completed  recorded {len(history)} generations", flush=True)
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search distillation vs. Checkers, recorded to telemetry.")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3, help="plies the finished player searches")
    parser.add_argument("--label-depth", type=int, default=6, help="plies of material search that label positions")
    parser.add_argument("--label-kind", choices=("search", "rollout", "blend"), default="search")
    parser.add_argument("--pool-games", type=int, default=2000)
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--held-out-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--arm", default=None)
    a = parser.parse_args()
    main(
        a.generations, a.population, a.hidden, a.depth, a.label_depth, a.pool_games, a.sample, a.sigma,
        a.mutation_rate, tuple(a.opponents.split(",")), a.rng_seed, a.held_out_every, a.workers,
        {"experiment": a.experiment, "arm": a.arm} if a.experiment else None,
        a.label_kind,
    )

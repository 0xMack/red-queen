"""What the Checkers training jobs share (jobs/checkers_neuro_run.py: fixed-topology networks;
jobs/checkers_neat_run.py: NEAT graphs): docs/design/0006, 0008.

A genome is a *position evaluator* -- a network scoring a board from the mover's side -- and a player is that
evaluator plus a search (`depth` plies of alpha-beta, in the Rust core; depth 1 is one ply of lookahead).
Fitness is match outcomes, not a simulation score:

- **fixed opponents** (`random`, `material-N`, ...): a benchmark that never changes, so progress against it
  is real progress;
- a **hall of fame** of the run's own past champions: an opponent that keeps getting harder, so a population
  that has beaten every fixed opponent still has something to learn from. Without it, "beat the strongest
  fixed opponent" is a cliff with no slope on the way up; with only it, a population can chase itself in a
  circle -- hence both;
- each opponent is played from both seats, one fitness value per (opponent, seat), so lexicase selection can
  prefer a genome that beats the hard one from one seat over one that only beats the easy ones.

`material_seed_*` start part of a population near a plain material evaluator (a stand-in for "count the
pieces", which any Checkers player begins with) so early generations spend their time on what material
misses -- advancement, back row, tempo -- instead of rediscovering that pieces matter. The honest bar for a
seeded run is the *same-depth material* opponent (`material-N`), not a shallower one.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

from evolve import (
    MatchFitnessEvaluator,
    NeatGenome,
    WeightVector,
    initial_genome,
    play_match,
)
from evolve.neat import InnovationTracker
from games.checkers import Checkers
from games.checkers_strategies import STRATEGIES, evaluator, graph_evaluator
from telemetry import FileArtifactStore, FileMetricsStore, GenerationStats

INPUTS = 32  # games.checkers' observation: the 32 playable squares from the mover's perspective
MAX_MOVES = 200  # plies per match (a draw beyond that), the same cap jobs/checkers_round_robin.py uses
MONITOR_GAMES = 10  # per opponent, alternating seats
MONITOR_SEED_BASE = 20_000  # opponent rng seeds; training seeds by opponent index, never these

Genome = WeightVector | NeatGenome

# A draw is worth at most this much: a win (+1) must always beat the best draw, however big the material edge.
MARGIN_WEIGHT = 0.5
MARGIN_SCALE = 3.0  # material difference (men 1, kings 2) at which the margin is ~76% of its maximum


def material_balance(env: Checkers, seat: int) -> float:
    """`seat`'s material minus its opponent's (men 1, kings 2) in `env`'s current position."""
    return sum((1 if owner == seat else -1) * (2 if king else 1) for owner, king in env.board.values())


def margin_scorer(env: Checkers, seat: int, result) -> float:
    """Win +1, loss -1; a game nobody won is worth the material edge held at the end, squashed into
    (-MARGIN_WEIGHT, MARGIN_WEIGHT). Building an advantage is rewarded before it becomes a win, which is
    exactly the gradient a population needs when most games against a strong opponent are draws."""
    if result.winner is not None:
        return 1.0 if result.winner == seat else -1.0
    return MARGIN_WEIGHT * math.tanh(material_balance(env, seat) / MARGIN_SCALE)


def strategy_factory(genome: Genome, depth: int):
    """`(env, rng) -> Strategy` for a genome of either kind, searching `depth` plies."""
    if isinstance(genome, NeatGenome):
        return graph_evaluator(genome.graph_encoding(), depth)
    return evaluator(genome.weights, genome.layer_sizes, depth)


def make_act(depth: int, tie_seed: int = 0):
    """`(genome, env) -> Strategy`, the env-aware `act` MatchFitnessEvaluator expects. Ties between equally
    scored moves are broken by a fixed rng, so a genome's fitness is deterministic."""

    def act(genome: Genome, env: Checkers):
        return strategy_factory(genome, depth)(env, random.Random(tie_seed))

    return act


def opponent_pool(names: Sequence[str], seed_offset: int = 0, repeats: int = 1):
    """Env-aware fixed opponents for MatchFitnessEvaluator: each name `repeats` times, every copy with its own rng
    seed (`seed_offset` + its index), so they play different games (random tie-breaks, a random player's moves).
    Fixed *within* a generation -- every genome faces the same games, which is what selection needs to compare
    them fairly."""

    def bind(name: str, seed: int):
        return lambda env: STRATEGIES[name](env, random.Random(seed))

    return [bind(name, seed_offset + rep * 7919 + i) for rep in range(repeats) for i, name in enumerate(names)]


class OpponentPool:
    """The fitness evaluator: fixed opponents plus a hall of fame of past champions (see the module docs).
    Register `on_generation` as an `on_generation` callback so the hall fills as the run goes.

    With `resample` the opponents' rng seeds change every generation. Otherwise every generation replays
    the very same games, and a population happily learns to win *those games* (training fitness reaches +1
    while games it never saw get worse -- measured here) instead of learning to play. Fresh games each
    generation leave nothing to memorize; `games_per_opponent` plays each opponent that many times per
    generation, for a less noisy signal."""

    def __init__(
        self,
        names: Sequence[str],
        depth: int,
        hall_size: int = 0,
        hall_every: int = 10,
        max_moves: int = MAX_MOVES,
        margin: bool = False,
        resample: bool = False,
        games_per_opponent: int = 1,
        opening_plies: int = 0,
    ):
        self._opening_plies = opening_plies
        self._names = list(names)
        self._resample = resample
        self._repeats = games_per_opponent
        self._generation = 0
        self._depth = depth
        self._hall_size = hall_size
        self._hall_every = hall_every
        self._max_moves = max_moves
        self._margin = margin  # score draws by the material edge held (see `margin_scorer`)
        self.hall: list[Genome] = []

    def _fixed(self):
        # Well clear of the monitor's seeds (MONITOR_SEED_BASE + ...), and different every generation if resampling.
        offset = 100_000 + (self._generation * 1_009 if self._resample else 0)
        return opponent_pool(self._names, offset, self._repeats)

    def _env_factory(self):
        """Fresh games, each opened by `opening_plies` random moves. Deterministic players otherwise replay the very
        same game from the standard start, so a genome's fitness would be a handful of games however many opponents
        it plays. Both seats of an opponent get the same opening (matches are created in order, seat 0 then 1), so
        an opening's luck cancels; every genome in a generation faces the same openings, so they compare fairly."""
        if not self._opening_plies:
            return Checkers
        count = [0]
        base = 500_000 + (self._generation * 1_009 if self._resample else 0)

        def make() -> Checkers:
            rng = random.Random(base + count[0] // 2)
            count[0] += 1
            env = Checkers()
            for _ in range(self._opening_plies):
                moves = env.legal_moves()
                if not moves or env.winner() is not None:
                    break
                env.step(rng.choice(moves))
            return env

        return make

    def evaluate(self, genome: Genome) -> list[float]:
        depth = self._depth
        hall = [
            lambda env, g=g, i=i: strategy_factory(g, depth)(env, random.Random(1000 + i))
            for i, g in enumerate(self.hall)
        ]
        inner = MatchFitnessEvaluator(
            env_factory=self._env_factory(),
            opponents=[*self._fixed(), *hall],
            act=make_act(depth),
            max_moves=self._max_moves,
            env_aware=True,
            scorer=margin_scorer if self._margin else None,
        )
        return inner.evaluate(genome)

    def on_generation(self, summary: Any) -> None:
        """After each generation: move on to the next one's games, and every `hall_every` generations let the
        champion join the hall (the oldest leaves when it's full)."""
        self._generation = summary.generation + 1
        if (
            self._hall_size
            and summary.generation % self._hall_every == 0
            and (not self.hall or summary.champion != self.hall[-1])
        ):
            self.hall = [*self.hall, summary.champion][-self._hall_size :]


def monitor_score(
    genome: Genome,
    names: Sequence[str],
    games: int = MONITOR_GAMES,
    seed_base: int = MONITOR_SEED_BASE,
    depth: int = 1,
    max_moves: int = MAX_MOVES,
) -> float:
    """Mean of win=1 / draw=0 / loss=-1 over `games` games per opponent, seats alternating, with opponent rng
    seeds training never used. (`seed_base` lets a later, separate measurement use games that champion
    *selection* never saw either.)"""
    results = []
    for i, name in enumerate(names):
        for game in range(games):
            env = Checkers()
            seat = game % 2
            seed = seed_base + 1000 * i + game
            strategies = {
                seat: strategy_factory(genome, depth)(env, random.Random(seed)),
                1 - seat: STRATEGIES[name](env, random.Random(seed + 50)),
            }
            winner = play_match(env, strategies, max_moves=max_moves).winner
            results.append(0.0 if winner is None else 1.0 if winner == seat else -1.0)
    return sum(results) / len(results)


def make_telemetry_callback(
    metrics: FileMetricsStore,
    artifacts: FileArtifactStore,
    run_id: str,
    opponents: Sequence[str],
    depth: int,
    held_out_every: int,
    last_generation: int,
    monitor_games: int = MONITOR_GAMES,
) -> Callable[[Any], None]:
    """An `on_generation` callback: store the champion, and every `held_out_every` generations score it on
    games it never trained on (the overfitting curve, same idea as snake_neuro_run.py's)."""

    def on_generation(summary: Any) -> None:
        champion_ref = f"{run_id}-gen{summary.generation}"
        artifacts.put_program(champion_ref, summary.champion.to_json().encode("utf-8"))
        held_out = None
        if summary.generation % held_out_every == 0 or summary.generation == last_generation:
            held_out = monitor_score(summary.champion, opponents, games=monitor_games, depth=depth)
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
                held_out_score=held_out,
                extras=summary.extras or None,  # NEAT: species count, champion size
            )
        )
        print(
            f"gen {summary.generation:>3}  best {summary.best_fitness:+.3f}  mean {summary.mean_fitness:+.3f}"
            + (f"  held-out {held_out:+.3f}" if held_out is not None else ""),
            flush=True,
        )

    return on_generation


# --- Seeding (see the module docs) -------------------------------------------------------------------


def material_seed_weights(layer_sizes: tuple[int, ...], rng: random.Random, noise: float = 0.15) -> WeightVector:
    """A `WeightVector` that starts as (a noisy) material evaluator: its first hidden unit sums the board
    (every square's own-man +1 / own-king +2 / opponent negative, so the sum is the material balance), and the
    output reads that unit. Everything else starts small and random for evolution to build on."""
    n_in, n_hidden = layer_sizes[0], layer_sizes[1]
    # Everything but the material unit starts small: a seed should play like material, not like noise.
    weights = [rng.uniform(-noise / 3, noise / 3) for _ in range(sum(o * (i + 1) for i, o in pairwise(layer_sizes)))]
    for k in range(n_in):
        weights[k] = 0.12 + rng.uniform(-noise / 4, noise / 4)  # hidden unit 0: tanh(0.12 * material)
    weights[n_hidden * n_in] = 0.0  # hidden unit 0's bias: an even position must read as even
    second = n_hidden * n_in + n_hidden  # start of the next layer's weights
    weights[second] = 2.0 + rng.uniform(-noise, noise)  # ... which reads hidden unit 0 strongly
    weights[-1] = 0.0  # the output's bias, for the same reason
    return WeightVector(weights=tuple(weights), layer_sizes=layer_sizes)


def material_seed_neat(
    tracker: InnovationTracker, rng: random.Random, weight_scale: float, noise: float = 0.1
) -> NeatGenome:
    """A NEAT genome that starts as (a noisy) material evaluator: every square wired to the output with the
    same small positive weight (the bias near zero)."""
    genome = initial_genome(INPUTS, 1, tracker, rng, weight_scale)
    from dataclasses import replace

    # Every square reads with the same small positive weight; the bias (source == INPUTS) starts near zero, so an
    # even position reads as even.
    connections = tuple(
        replace(c, weight=0.15 + rng.uniform(-noise, noise) if c.source < INPUTS else rng.uniform(-noise, noise) / 4)
        for c in genome.connections
    )
    return NeatGenome(genome.num_inputs, genome.num_outputs, connections)

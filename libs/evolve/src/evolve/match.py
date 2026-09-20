"""Multi-agent match running and match-outcome fitness -- the two-player (and, later, N-player)
analogue of simulation.py's single-agent Environment/SimulationFitnessEvaluator (docs/design/0006).

`MultiAgentEnvironment` is the interface this module depends on; a concrete game (e.g.
games.checkers in libs/games) implements it but never imports from here -- same dependency
direction as Environment/games (docs/design/0001 "Decoupling from telemetry", applied here to
decoupling from evolve instead).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

Genome = TypeVar("Genome")
Observation = Any
Move = Any


class MultiAgentEnvironment(Protocol):
    def reset(self) -> Observation: ...
    def legal_moves(self) -> list[Move]: ...
    def current_player(self) -> int: ...
    def step(self, move: Move) -> tuple[Observation, dict[int, float], bool]:
        """Returns (observation, {player_index: reward}, done). Only players who should receive a
        signal this step need a key -- a non-terminal step can return {} if the game has no
        natural per-move reward; a terminal step sets the winner's reward to 1.0 and the loser's
        to -1.0 (0.0 for both on a draw)."""
        ...

    def winner(self) -> int | None:
        """Player index, or None if the game is still ongoing or ended in a draw -- a caller that
        needs to tell those two apart uses `done` (from the last step()) alongside this."""
        ...


# (observation, legal_moves) -> move. No separate class hierarchy -- every backend (static
# heuristic, evolved genome, classifier) satisfies this same plain-function shape; see
# docs/design/0006 for why this is just a wider version of the act(genome, observation) -> action
# pattern already used for Snake, not a new concept.
Strategy = Callable[[Observation, list[Move]], Move]

# A strategy that needs the environment itself -- to look ahead (`env.simulate(move)`, a copy + step) or
# evaluate the position each legal move leads to. The plain `Strategy` shape can't do that: it only
# sees the observation and the move list. Binding to the env is a separate step so a fitness
# evaluator, which builds a fresh env per match, can hand each match's own env to the strategy.
StrategyFactory = Callable[[MultiAgentEnvironment], Strategy]


@dataclass(frozen=True, slots=True)
class MatchResult:
    winner: int | None
    moves_played: int


def play_match(
    env: MultiAgentEnvironment,
    strategies: dict[int, Strategy],
    max_moves: int = 200,
) -> MatchResult:
    """Runs one match to completion (or max_moves), alternating turns per env.current_player().

    The "pit any strategy against any strategy" mechanic requested as reusable infrastructure
    (docs/design/0006), not a checkers-specific feature -- every future multi-agent game gets this
    for free by implementing MultiAgentEnvironment.
    """
    observation = env.reset()
    moves_played = 0
    done = False
    while not done and moves_played < max_moves:
        player = env.current_player()
        move = strategies[player](observation, env.legal_moves())
        observation, _rewards, done = env.step(move)
        moves_played += 1

    return MatchResult(winner=env.winner(), moves_played=moves_played)


class MatchFitnessEvaluator:
    """FitnessEvaluator via match outcomes against a fixed pool of reference opponents -- the
    multi-agent analogue of SimulationFitnessEvaluator's fixed benchmark environments. One fitness
    value per (opponent, seat) pair, matching the per-test-case contract every other
    FitnessEvaluator in this package follows, so LexicaseSelection works on match outcomes exactly
    as it does on dataset/simulation fitness -- a genome that beats a strong opponent and loses to
    a weak one shouldn't look the same in aggregate as the reverse, the same tension lexicase
    already exists to address (docs/design/0003).

    Plays each opponent from both seat 0 and seat 1 (two fitness values per opponent) so first-move
    advantage is cancelled out in the fitness signal itself, not just left to average out by luck.

    `act` is injected `(genome, observation, legal_moves) -> move`, the multi-agent-aware sibling
    of SimulationFitnessEvaluator's `act` -- this evaluator never touches genome internals.

    `env_aware=True` is for strategies that need the match's environment (lookahead opponents, a
    genome that scores the position each legal move leads to -- see `StrategyFactory`): `opponents`
    are then `StrategyFactory`s and `act` is `(genome, env) -> Strategy`, both called once per match
    with that match's fresh env. Without it, both are plain strategies, exactly as before.

    `scorer(env, genome_seat, result) -> float` replaces the default win 1 / draw 0 / loss -1 with a finer
    score, given the finished environment (a draw can be worth the material edge held, so a genome is rewarded
    for building an advantage before it learns to convert it; keep it inside (-1, 1) so a win still counts
    most). Many matches against a strong opponent end as draws or losses, and a fitness that can't tell those
    apart gives selection nothing to climb.
    """

    def __init__(
        self,
        env_factory: Callable[[], MultiAgentEnvironment],
        opponents: Sequence[Strategy],
        act: Callable[[Genome, Observation, list[Move]], Move],
        max_moves: int = 200,
        env_aware: bool = False,
        scorer: Callable[[MultiAgentEnvironment, int, MatchResult], float] | None = None,
    ):
        self._env_factory = env_factory
        self._opponents = list(opponents)
        self._act = act
        self._max_moves = max_moves
        self._env_aware = env_aware
        self._scorer = scorer

    def evaluate(self, genome: Genome) -> list[float]:
        def genome_strategy(observation: Observation, legal_moves: list[Move]) -> Move:
            return self._act(genome, observation, legal_moves)

        fitnesses = []
        for opponent in self._opponents:
            for genome_seat in (0, 1):
                env = self._env_factory()
                if self._env_aware:
                    strategies = {genome_seat: self._act(genome, env), 1 - genome_seat: opponent(env)}
                else:
                    strategies = {genome_seat: genome_strategy, 1 - genome_seat: opponent}
                result = play_match(env, strategies, max_moves=self._max_moves)
                if self._scorer is not None:
                    fitnesses.append(self._scorer(env, genome_seat, result))
                elif result.winner is None:
                    fitnesses.append(0.0)
                elif result.winner == genome_seat:
                    fitnesses.append(1.0)
                else:
                    fitnesses.append(-1.0)
        return fitnesses

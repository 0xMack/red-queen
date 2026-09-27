"""Multi-armed bandits (docs/design/0011): K slot machines with hidden payouts, a fixed budget of pulls.

Pull an arm, get its payout; the game ends when the budget is spent. A *scenario* decides how the arms are drawn
from the seed -- each is built to expose a particular strategy's weakness (see `SCENARIOS`). The rules run in the
Rust game core (rust/core/src/bandit.rs); this is their Python face.

Two things make it a good teaching game, and both come from the core's streams:

- the n-th pull of an arm pays the same whoever pulls it, so players racing on one seed face the same luck;
- regret is *expected* regret (the best arm's mean minus the pulled arm's, per pull): it scores the choices, not
  the dice.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

from games import _native


@dataclass(frozen=True)
class ScenarioInfo:
    id: str
    title: str
    arms: int
    budget: int
    contexts: int
    binary: bool
    lesson: str
    sequential: bool = False


SCENARIOS: dict[str, ScenarioInfo] = {
    s.id: s
    for s in [
        ScenarioInfo("classic", "Classic", 5, 100, 1, True, "Five win/lose machines, one clearly best."),
        ScenarioInfo("close-call", "Close call", 5, 100, 1, True, "The best machine is only a little better."),
        ScenarioInfo("lucky-start", "Lucky start", 5, 100, 1, False, "Noisy payouts: an early lucky win misleads."),
        ScenarioInfo("jackpot", "Jackpot", 5, 100, 1, False, "The best machine rarely pays -- but pays big."),
        ScenarioInfo("drifting", "Drifting", 5, 200, 1, True, "Early on, the best machine breaks."),
        ScenarioInfo("too-many-arms", "Too many arms", 16, 100, 1, True, "Too many machines to try them all."),
        ScenarioInfo("two-lamps", "Two lamps", 5, 100, 2, True, "The best machine depends on the lamp's colour."),
        ScenarioInfo(
            "detour", "Detour", 5, 100, 2, False, "A machine that pays nothing opens a better room.", sequential=True
        ),
    ]
}


class Bandit:
    def __init__(self, scenario: str = "classic", seed: int = 0):
        self.scenario = scenario
        self.seed = seed
        self._core = _native.BanditCore(scenario, seed)

    def __copy__(self):
        return copy.deepcopy(self)

    @property
    def info(self) -> ScenarioInfo:
        return SCENARIOS[self.scenario]

    def reset(self) -> None:
        """Back to the first pull of the same game."""
        self._core.reset()

    def pull(self, arm: int) -> tuple[float, bool]:
        """Pull `arm`: (payout, the game is over)."""
        return self._core.pull(arm)

    @property
    def lamp(self) -> int:
        """The lamp lit for the next pull: 0 (red) or 1 (blue); always 0 in a scenario without lamps. In `detour` it's
        the room you're in (0 red, 1 gold), and your pull decides the next one."""
        return self._core.lamp

    def means(self) -> list[float]:
        """Each arm's true mean right now (under this lamp, after any drift) -- hidden from players."""
        return self._core.means()

    def payouts(self, lamp: int = 0) -> list[tuple[str, dict[str, float]]]:
        """Each arm's payout under `lamp` before any drift: its kind and parameters (the end-of-game reveal)."""
        return [(kind, dict(params)) for kind, params in self._core.payouts(lamp)]

    def drift(self) -> tuple[int, list[float]] | None:
        """A drifting game's switch: the pull it happens at and each arm's mean after it."""
        return self._core.drift()

    arms = property(lambda self: self._core.arms)
    budget = property(lambda self: self._core.budget)
    pulls = property(lambda self: self._core.pulls)
    done = property(lambda self: self._core.done)
    total = property(lambda self: self._core.total)
    regret = property(lambda self: self._core.regret)
    efficiency = property(lambda self: self._core.efficiency)
    skill = property(lambda self: self._core.skill)
    counts = property(lambda self: self._core.counts)
    best_pulls = property(lambda self: self._core.best_pulls)
    best_arm = property(lambda self: self._core.best_arm)
    detour = property(lambda self: self._core.detour)


class BanditNone:
    """`none.v1` -- sees nothing: every pull is the same situation, so a strategy keeps one row of estimates. Right
    for a plain bandit; blind to the lamp in `two-lamps`, where the arms pay differently under each colour."""

    id = "none.v1"
    level = 1
    description = "Nothing: every pull looks the same, so a strategy keeps one value per machine -- a one-row table."

    def encode(self, game: Bandit) -> list[float]:
        return []

    def feature_names(self, game: Bandit) -> list[str]:
        return []


class BanditLamp:
    """`lamp.v1` -- the lamp's colour (0 red, 1 blue): two situations, so a strategy keeps a row per colour (the
    contextual bandit, docs/design/0011 Level 2). The lamp is always red outside `two-lamps`."""

    id = "lamp.v1"
    level = 1
    description = "The lamp's colour, red or blue: two situations, so a strategy keeps a row of values per colour."

    def encode(self, game: Bandit) -> list[float]:
        return [float(game.lamp)]

    def feature_names(self, game: Bandit) -> list[str]:
        return ["lamp is blue"]


class PullArm:
    """`arm.v1` -- one output per machine; the strategy pulls the machine it scores highest."""

    id = "arm.v1"
    description = "One score per machine; pull the machine that scores highest."
    num_outputs = 5

    def decode(self, outputs: Sequence[float]) -> int:
        return max(range(len(outputs)), key=lambda i: outputs[i])

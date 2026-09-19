"""Observers, action adapters, and interfaces -- how a model sees and acts on a game
(docs/design/0007).

A game owns its rules and full state. An **observer** turns that state into the flat
`list[float]` a model consumes; an **action adapter** turns a model's raw outputs into a game
action. An **interface** is the named, versioned combination `(game, observer, action adapter)`,
e.g. `snake/features.v1+relative3.v1` -- every run, champion, and leaderboard entry records one, so
a model is only ever run under the interface it was trained for. Changing what an observer encodes
means a new version, never an in-place edit (docs/design/0007: that's exactly what broke the
pre-0007 Snake champions).

Like the rest of this package, nothing here depends on `evolve`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

# Representation levels (docs/design/0007), from rawest to most pre-chewed.
LEVELS: dict[int, str] = {
    0: "visual",
    1: "full state",
    2: "local / egocentric",
    3: "engineered",
}


class Observer(Protocol):
    id: str  # e.g. "features.v1" -- unique within a game, versioned
    level: int  # a LEVELS key
    description: str

    def encode(self, game: Any) -> list[float]:
        """The observation a model sees. Called every step of training -- keep it cheap."""
        ...

    def feature_names(self, game: Any) -> list[str]:
        """One human-readable name per encoded value, in order."""
        ...


class ActionAdapter(Protocol):
    id: str  # e.g. "relative3.v1"
    description: str
    num_outputs: int  # how many raw outputs a model must produce

    def decode(self, outputs: Sequence[float]) -> Any:
        """Raw model outputs -> a game action."""
        ...


@dataclass(frozen=True)
class Interface:
    game: str
    observer: Observer
    action: ActionAdapter
    # Builds a game instance wired to this interface's observer; kwargs are the game's own
    # constructor arguments (e.g. seed, board size).
    make_game: Callable[..., Any]

    @property
    def id(self) -> str:
        return f"{self.game}/{self.observer.id}+{self.action.id}"

    def describe(self, **game_kwargs: Any) -> dict[str, Any]:
        """Plain-data description (for apis/backend and UIs). Sizes and feature names are computed
        against a real game instance, since some depend on board size."""
        game = self.make_game(**game_kwargs)
        names = self.observer.feature_names(game)
        return {
            "id": self.id,
            "game": self.game,
            "observer": {
                "id": self.observer.id,
                "level": self.observer.level,
                "level_name": LEVELS[self.observer.level],
                "description": self.observer.description,
                "size": len(names),
                "feature_names": names,
            },
            "action": {
                "id": self.action.id,
                "description": self.action.description,
                "num_outputs": self.action.num_outputs,
            },
        }

"""In-memory game session state.

No telemetry/persistence layer for this (docs/design/0005): a watch/play session exists only for
the life of the backend process, which is the right scope -- it's not a training run, so it has no
reason to go through `libs/telemetry`. A session doesn't survive a backend restart.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from games.snake import Snake


class RenderableEnvironment(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool]: ...
    def render_state(self) -> dict: ...


# Only games implementing Renderable belong here -- e.g. games.reach1d doesn't have a
# render_state() yet, so it isn't watchable/playable through this API until it does.
_GAME_FACTORIES: dict[str, type[RenderableEnvironment]] = {
    "snake": Snake,
}


def known_games() -> list[str]:
    return list(_GAME_FACTORIES)


def json_safe_render_state(render_state: dict) -> dict:
    """render_state()'s `cells` (grid games, e.g. games.snake) is `dict[(x, y): label]` -- tuple
    keys aren't valid JSON object keys. Flatten to a list of cell records instead; every other
    field passes through unchanged."""
    safe = dict(render_state)
    cells = safe.get("cells")
    if isinstance(cells, dict):
        safe["cells"] = [
            {"x": x, "y": y, "label": label} for (x, y), label in cells.items()
        ]
    return safe


@dataclass
class GameSession:
    game: str
    seed: int
    env: RenderableEnvironment
    step_count: int = 0
    done: bool = False
    states: list[dict] = field(default_factory=list)
    actions: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


class GameSessionStore:
    """Local-dev-only in-memory session registry -- single process, no persistence, no locking
    (docs/design/0005 "local single-user tool" scope)."""

    def __init__(self) -> None:
        self._sessions: dict[str, GameSession] = {}

    def create(self, game: str, seed: int | None) -> tuple[str, GameSession]:
        if game not in _GAME_FACTORIES:
            raise KeyError(game)
        resolved_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        env = _GAME_FACTORIES[game](seed=resolved_seed)
        env.reset()
        session = GameSession(game=game, seed=resolved_seed, env=env)
        session.states.append(json_safe_render_state(env.render_state()))
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = session
        return session_id, session

    def get(self, game: str, session_id: str) -> GameSession:
        session = self._sessions.get(session_id)
        if session is None or session.game != game:
            raise KeyError(session_id)
        return session

    def step(
        self, game: str, session_id: str, action: float
    ) -> tuple[GameSession, float]:
        session = self.get(game, session_id)
        if session.done:
            raise ValueError(f"session already finished: {session_id}")
        _, reward, done = session.env.step(action)
        session.step_count += 1
        session.done = done
        session.actions.append(action)
        session.rewards.append(reward)
        session.states.append(json_safe_render_state(session.env.render_state()))
        return session, reward

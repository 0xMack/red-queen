"""Registry of every named interface (game + observer + action adapter), docs/design/0007.

The one place that maps an interface id -- as recorded in a run's config, a champion, or a
leaderboard entry -- back to something that can build a correctly-observed game and decode a
model's outputs. Used by jobs (training, evaluation), apis/backend, and the Pyodide worker alike.
"""

from __future__ import annotations

from games.checkers import Checkers, CheckersBoard32, Evaluate1Ply
from games.observation import Interface
from games.snake import RelativeTurn3, Snake, SnakeFeatures, SnakeGridFlat


def _snake_interface(observer) -> Interface:
    return Interface(
        game="snake",
        observer=observer,
        action=RelativeTurn3(),
        make_game=lambda **kwargs: Snake(observer=observer, **kwargs),
    )


_ALL: list[Interface] = [
    _snake_interface(SnakeFeatures()),
    _snake_interface(SnakeGridFlat()),
    Interface(game="checkers", observer=CheckersBoard32(), action=Evaluate1Ply(), make_game=lambda **kwargs: Checkers(**kwargs)),
]
_BY_ID: dict[str, Interface] = {i.id: i for i in _ALL}

# What a Snake run recorded before interfaces existed was trained against -- see
# jobs/backfill_interfaces.py, which resolves each legacy run properly from its champion's layer
# sizes rather than assuming this.
SNAKE_DEFAULT = "snake/features.v1+relative3.v1"


def get(interface_id: str) -> Interface:
    try:
        return _BY_ID[interface_id]
    except KeyError:
        raise KeyError(f"unknown interface: {interface_id} (known: {sorted(_BY_ID)})") from None


def for_game(game: str) -> list[Interface]:
    return [i for i in _ALL if i.game == game]


def find(game: str, input_size: int, num_outputs: int, **game_kwargs) -> Interface | None:
    """The interface of `game` whose observation/action shapes match a model's -- how legacy
    champions (recorded without an interface id) are matched to one. None if nothing matches."""
    for interface in for_game(game):
        observed = interface.make_game(**game_kwargs)
        if (
            len(interface.observer.feature_names(observed)) == input_size
            and interface.action.num_outputs == num_outputs
        ):
            return interface
    return None

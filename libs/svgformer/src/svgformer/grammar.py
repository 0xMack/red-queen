"""The token grammar (docs/design/0024 Decision 2): which tokens may come next, so constrained decoding only ever writes
sequences that decode to an SVG in the subset.

A `State` is immutable and small. `allowed(state)` is the set of token ids that may follow; `mask(state)` the same as a
boolean vector over the vocabulary (cached by the state's `key`, of which there are a handful, so a decoder can turn
them into tensors once); `advance(state, token)` the state after it.

The grammar is the tokenizer's, made strict where a looser one would decode to nothing:
- a polygon has at least three points before `POLY_END`;
- a path starts with `M`; each subpath has at least one `L`, `Q` or `C` before a `Z`, a new `M` or `PATH_END`; a new `M`
  comes only after a `Z` or a drawn segment.
A sequence cut off by a length cap is still fine: `tokens.decode` drops the shape it ended in.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np

from svgformer import tokens as T

NUMBERS = frozenset(range(T.NUM0, T.NUM0 + T.GRID))
COLOURS = frozenset(range(T.COLOUR0, T.COLOUR0 + T.COLOURS))
OPACITIES = frozenset(range(T.OPACITY0, T.OPACITY0 + T.OPACITIES))
TYPES = frozenset(T.TYPE_ID.values())
M, L, Q, C, Z = (T.COMMAND_ID[c] for c in T.COMMANDS)


@dataclasses.dataclass(frozen=True)
class State:
    """Where a sequence is.

    - `kind`: `start` (before BOS), `shape` (a type or END next), `numbers` (`left` numbers to go, then back to
      `after`), `polygon` (`points` so far: a number or POLY_END next), `path` (a command or PATH_END next), `colour`
      (`left` colour tokens to go: the paint follows the geometry), `style` (an opacity, or the next type or END),
      `done`.
    - `shape`: the shape being written.
    - `drawn`: in a path, whether the current subpath has a segment; `closed`: whether it was just closed with Z;
      `started`: whether the path has had its first M.
    """

    kind: str
    shape: str = ""
    left: int = 0
    after: str = ""
    points: int = 0
    started: bool = False
    drawn: bool = False
    closed: bool = False

    @property
    def key(self) -> tuple:
        """What `allowed` depends on."""
        if self.kind == "polygon":
            return ("polygon", self.points >= 3)
        if self.kind == "path":
            return ("path", self.started, self.drawn, self.closed)
        return (self.kind,)


START = State("start")


def allowed(state: State) -> frozenset[int]:
    return _allowed(state.key)


@cache
def _allowed(key: tuple) -> frozenset[int]:
    kind = key[0]
    if kind == "start":
        return frozenset({T.BOS})
    if kind == "shape":
        return TYPES | {T.END}
    if kind == "colour":
        return COLOURS
    if kind == "style":
        return OPACITIES | TYPES | {T.END}
    if kind == "numbers":
        return NUMBERS
    if kind == "polygon":
        return NUMBERS | ({T.POLY_END} if key[1] else frozenset())
    if kind == "path":
        _, started, drawn, closed = key
        if not started:
            return frozenset({M})
        if closed:
            return frozenset({M, T.PATH_END})
        if drawn:
            return frozenset({L, Q, C, Z, M, T.PATH_END})
        return frozenset({L, Q, C})
    return frozenset()


@cache
def _mask(key: tuple) -> np.ndarray:
    out = np.zeros(T.VOCAB, dtype=bool)
    out[sorted(_allowed(key))] = True
    out.setflags(write=False)
    return out


def mask(state: State) -> np.ndarray:
    return _mask(state.key)


def advance(state: State, token: int) -> State:
    if token not in allowed(state):
        raise ValueError(f"{T.NAMES[token]} can't follow {state}")
    kind = state.kind
    if kind == "start":
        return State("shape")
    if kind == "shape":
        if token == T.END:
            return State("done")
        return _geometry(T.NAMES[token])
    if kind == "colour":
        if state.left > 1:
            return dataclasses.replace(state, left=state.left - 1)
        return State("style", shape=state.shape)
    if kind == "style":
        if token in OPACITIES:
            return State("shape")
        return advance(State("shape"), token)  # opaque: this is already the next shape's type, or END
    if kind == "numbers":
        if state.left > 1:
            return dataclasses.replace(state, left=state.left - 1)
        return _after_numbers(state)
    if kind == "polygon":
        if token == T.POLY_END:
            return State("colour", shape="polygon", left=3)
        return State("numbers", shape="polygon", left=1, after="polygon", points=state.points)
    if kind == "path":
        if token == T.PATH_END:
            return State("colour", shape="path", left=3)
        return _command(state, token)
    raise ValueError(f"nothing follows {state}")


def _geometry(shape: str) -> State:
    """The state at the start of `shape`'s geometry."""
    if shape in T.PRIMITIVE_NUMBERS:
        return State("numbers", shape=shape, left=T.PRIMITIVE_NUMBERS[shape], after="paint")
    if shape == "polygon":
        return State("polygon", shape="polygon", points=0)
    return State("path", shape="path")


def _after_numbers(state: State) -> State:
    if state.after == "paint":
        return State("colour", shape=state.shape, left=3)
    if state.after == "polygon":
        return State("polygon", shape="polygon", points=state.points + 1)
    # a path command's numbers are done: the subpath has been started or drawn on
    return State("path", shape="path", started=True, drawn=state.drawn, closed=False)


def _command(state: State, token: int) -> State:
    command = T.NAMES[token]
    if command == "Z":
        return State("path", shape="path", started=True, drawn=False, closed=True)
    n = T.COMMAND_NUMBERS[command]
    # after M the subpath has nothing drawn yet; after L/Q/C it has
    return State("numbers", shape="path", left=n, after="path", drawn=command != "M")


def accepts(tokens: list[int]) -> bool:
    """Whether `tokens` is a complete sequence (BOS ... END) of the grammar."""
    state = START
    try:
        for t in tokens:
            state = advance(state, t)
    except ValueError:
        return False
    return state.kind == "done"

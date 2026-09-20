"""Checkers (American/English draughts): the first two-player game (docs/design/0006).

Implements evolve.match.MultiAgentEnvironment's shape -- doesn't depend on evolve. Also implements
games.rendering.Renderable's render_state(), same decoupling as every other game in this package. The
rules run in the Rust game core (rust/core/src/checkers.rs, docs/design/0009); this is its Python face.

A Move is a full turn: `(from, *landing_squares)`. Length 2 is a non-capturing step; length 3+ is a
capture (possibly a multi-jump chain, each consecutive pair one jump). Representing a whole turn as
one Move -- rather than one jump per step() call -- keeps current_player() alternating strictly
after every step(), so MultiAgentEnvironment doesn't need a "the same player moves again" concept
just for this one game.

Rules: men move/capture diagonally forward only; kings (reaching the far row) move/capture in all
four diagonal directions; captures are mandatory when available, and a piece that can capture again
immediately must continue the chain (both enforced by legal_moves() only ever returning capture
moves when any exist, as *maximal* chains). A player with no legal move on their turn has lost.
Draws are a move-count-without-capture limit, the same shape as games.snake's
max_steps_without_food starvation timeout, rather than full threefold-repetition detection.

Observation: each of the 32 playable squares encoded from the *current player's* perspective (own
man=1, own king=2, empty=0, opponent man=-1, opponent king=-2) -- a fixed-size vector regardless of
how many pieces remain. Deliberately not per-candidate-move features (docs/design/0006 left the exact
scheme open): simulate() lets a strategy evaluate "what would the resulting position look like" for
any legal move using this same encoding, which supports a classic 1-ply position-evaluator strategy
without needing move-specific feature engineering.

`board` reads and assigns as a `{(x, y): (owner, is_king)}` dict whose *order matters*: legal moves
are generated in board order, the way the original dict-based implementation did (a moved piece goes
last), and strategies pick by position in that list.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from games import _native

Square = tuple[int, int]
Move = tuple[Square, ...]
_Piece = tuple[int, bool]  # (owner, is_king)

_PLAYABLE_SQUARES: tuple[Square, ...] = tuple((x, y) for y in range(8) for x in range(8) if (x + y) % 2 == 1)


def _square_name(x: int, y: int) -> str:
    return f"{'abcdefgh'[x]}{y + 1}"


class CheckersBoard32:
    """`board32.v1` -- the observer every Checkers model sees today (docs/design/0007, level 1: the whole
    board, symbolically): the 32 playable squares, each from the *player to move's* side (own man 1, own
    king 2, opponent -1 / -2, empty 0). Same values `Checkers.reset()`/`step()` return."""

    id = "board32.v1"
    level = 1
    description = (
        "The 32 playable squares, signed from the mover's perspective: own man +1, own king +2, "
        "opponent man -1, opponent king -2, empty 0. Row by row from Red's side."
    )

    def encode(self, game: Checkers) -> list[float]:
        return game._observation()

    def feature_names(self, game: Checkers) -> list[str]:
        return [_square_name(x, y) for x, y in _PLAYABLE_SQUARES]


class Evaluate1Ply:
    """`evaluate1ply.v1` -- how a model's output becomes a move. Not a fixed action space: a Checkers
    model is a *position evaluator* with one output, and the player scores the position each legal move
    leads to (`Checkers.simulate()`) and plays the best -- one ply of lookahead, the reason the observer
    can be a plain board (docs/design/0006)."""

    id = "evaluate1ply.v1"
    description = (
        "One output: how good a position is for the player to move. The player scores the position "
        "every legal move leads to and plays the move whose position is worst for the opponent."
    )
    num_outputs = 1

    def decode(self, outputs: Sequence[float]) -> float:
        """The evaluation itself; choosing the move is `games.checkers_strategies.evaluator`'s job."""
        return outputs[0]


class Checkers:
    def __init__(self, max_moves_without_capture: int = 40):
        self.max_moves_without_capture = max_moves_without_capture
        self._core = _native.CheckersCore(max_moves_without_capture)

    def __copy__(self):
        # The state lives in the native core, so even a "shallow" copy must not share it.
        return copy.deepcopy(self)

    @property
    def board(self) -> dict[Square, _Piece]:
        return {(x, y): (owner, king) for x, y, owner, king in self._core.board}

    @board.setter
    def board(self, board: dict[Square, _Piece]) -> None:
        self._core.board = [(x, y, owner, bool(king)) for (x, y), (owner, king) in board.items()]

    @property
    def current_player_index(self) -> int:
        return self._core.current_player

    @current_player_index.setter
    def current_player_index(self, player: int) -> None:
        self._core.current_player = player

    def reset(self) -> list[float]:
        self._core.reset()
        return self._core.observation()

    def legal_moves(self) -> list[Move]:
        return [tuple(move) for move in self._core.legal_moves()]

    def current_player(self) -> int:
        return self._core.current_player

    def winner(self) -> int | None:
        return self._core.winner

    def step(self, move: Move) -> tuple[list[float], dict[int, float], bool]:
        done = self._core.step([tuple(square) for square in move])
        rewards: dict[int, float] = {}
        if done:
            winner = self._core.winner
            rewards = {winner: 1.0, 1 - winner: -1.0} if winner is not None else {0: 0.0, 1: 0.0}
        return self._core.observation(), rewards, done

    def choose(self, strategy: _native.CheckersStrategy) -> Move:
        """The move a native strategy (games.checkers_strategies) picks in the current position."""
        return self.legal_moves()[strategy.pick(self._core)]

    def simulate(self, move: Move) -> list[float]:
        """The observation that *would* result from playing `move` (from the next mover's
        perspective), without mutating this environment -- lets a position-evaluator strategy score
        every legal move by resulting position without a full environment copy per candidate."""
        return self._core.simulate([tuple(square) for square in move])

    def _observation(self) -> list[float]:
        return self._core.observation()

    def render_state(self) -> dict:
        return {
            "width": 8,
            "height": 8,
            "cells": {(x, y): label for x, y, label in self._core.cells()},
            "current_player": self._core.current_player,
            "winner": self._core.winner,
        }

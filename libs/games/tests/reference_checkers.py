"""The oracle for the Rust port of Checkers (docs/design/0009 Decision 2): `games/checkers.py` exactly as it
was in pure Python before the game core moved to Rust, unchanged. `test_native_parity.py` holds the
Rust to it. Test-only on purpose -- the executable spec, not a second implementation to run.

The original module docstring follows.
"""

"""Checkers (American/English draughts): the first two-player game (docs/design/0006).

Implements evolve.match.MultiAgentEnvironment's shape -- doesn't depend on evolve. Also implements
games.rendering.Renderable's render_state(), same decoupling as every other game in this package.

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
how many pieces remain, unlike a raw "whatever's on the board" dump. Deliberately not per-candidate-
move features (docs/design/0006 left the exact scheme open): simulate() lets a strategy evaluate
"what would the resulting position look like" for any legal move using this same encoding, which
supports a classic 1-ply position-evaluator strategy (score every legal move's resulting position,
play the best) without needing move-specific feature engineering.
"""


from itertools import pairwise

Square = tuple[int, int]
Move = tuple[Square, ...]
_Piece = tuple[int, bool]  # (owner, is_king)

_PLAYABLE_SQUARES: tuple[Square, ...] = tuple((x, y) for y in range(8) for x in range(8) if (x + y) % 2 == 1)


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < 8 and 0 <= y < 8


def _king_row(player: int) -> int:
    return 7 if player == 0 else 0


def _directions(player: int, is_king: bool) -> tuple[tuple[int, int], ...]:
    if is_king:
        return ((1, 1), (-1, 1), (1, -1), (-1, -1))
    return ((1, 1), (-1, 1)) if player == 0 else ((1, -1), (-1, -1))


def _capture_chains(pos: Square, player: int, is_king: bool, board: dict[Square, _Piece]) -> list[list[Square]]:
    """All maximal capture-chain landing-square sequences starting from `pos` (excluding `pos`
    itself); [] if no capture is available from here. `board` must already have the moving piece's
    original square cleared and every piece captured earlier in this same chain removed, so
    occupancy/capture checks never see the mover's own trail or a piece it's already jumped."""
    chains: list[list[Square]] = []
    for dx, dy in _directions(player, is_king):
        over = (pos[0] + dx, pos[1] + dy)
        landing = (pos[0] + 2 * dx, pos[1] + 2 * dy)
        if not _in_bounds(*landing) or landing in board:
            continue
        occupant = board.get(over)
        if occupant is None or occupant[0] == player:
            continue
        next_board = dict(board)
        del next_board[over]
        continuing_is_king = is_king or landing[1] == _king_row(player)
        further = _capture_chains(landing, player, continuing_is_king, next_board)
        if further:
            chains.extend([landing, *rest] for rest in further)
        else:
            chains.append([landing])
    return chains


def _apply(board: dict[Square, _Piece], move: Move) -> tuple[dict[Square, _Piece], int]:
    """Returns (resulting board, number of pieces captured) -- shared by step() and simulate() so
    they can never disagree about what a move does."""
    new_board = dict(board)
    owner, is_king = new_board.pop(move[0])
    captured = 0
    for a, b in pairwise(move):
        if abs(b[0] - a[0]) == 2:
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            del new_board[mid]
            captured += 1
    end = move[-1]
    if end[1] == _king_row(owner):
        is_king = True
    new_board[end] = (owner, is_king)
    return new_board, captured


def _encode(board: dict[Square, _Piece], perspective_player: int) -> list[float]:
    values = []
    for square in _PLAYABLE_SQUARES:
        occupant = board.get(square)
        if occupant is None:
            values.append(0.0)
        else:
            owner, is_king = occupant
            sign = 1.0 if owner == perspective_player else -1.0
            values.append(sign * (2.0 if is_king else 1.0))
    return values


class Checkers:
    def __init__(self, max_moves_without_capture: int = 40):
        self.max_moves_without_capture = max_moves_without_capture
        self.reset()

    def reset(self) -> list[float]:
        self.board: dict[Square, _Piece] = {}
        for y in range(3):
            for x in range(8):
                if (x + y) % 2 == 1:
                    self.board[(x, y)] = (0, False)
        for y in range(5, 8):
            for x in range(8):
                if (x + y) % 2 == 1:
                    self.board[(x, y)] = (1, False)
        self.current_player_index = 0
        self._moves_without_capture = 0
        self._winner: int | None = None
        return self._observation()

    def legal_moves(self) -> list[Move]:
        player = self.current_player_index
        capture_moves: list[Move] = []
        simple_moves: list[Move] = []
        for pos, (owner, is_king) in self.board.items():
            if owner != player:
                continue
            in_transit = dict(self.board)
            del in_transit[pos]
            chains = _capture_chains(pos, player, is_king, in_transit)
            for chain in chains:
                capture_moves.append((pos, *chain))
            if not chains:
                for dx, dy in _directions(player, is_king):
                    dest = (pos[0] + dx, pos[1] + dy)
                    if _in_bounds(*dest) and dest not in self.board:
                        simple_moves.append((pos, dest))
        # Mandatory capture: if any piece has a capture available, simple moves aren't legal at all.
        return capture_moves if capture_moves else simple_moves

    def current_player(self) -> int:
        return self.current_player_index

    def winner(self) -> int | None:
        return self._winner

    def step(self, move: Move) -> tuple[list[float], dict[int, float], bool]:
        self.board, captured = _apply(self.board, move)
        self._moves_without_capture = 0 if captured else self._moves_without_capture + 1
        self.current_player_index = 1 - self.current_player_index

        if self._moves_without_capture >= self.max_moves_without_capture:
            done, self._winner = True, None
        elif not self.legal_moves():
            done, self._winner = True, 1 - self.current_player_index
        else:
            done, self._winner = False, None

        rewards: dict[int, float] = {}
        if done:
            rewards = {self._winner: 1.0, 1 - self._winner: -1.0} if self._winner is not None else {0: 0.0, 1: 0.0}

        return self._observation(), rewards, done

    def simulate(self, move: Move) -> list[float]:
        """The observation that *would* result from playing `move`, without mutating this
        environment -- lets a position-evaluator strategy score every legal move by resulting
        position (see the module docstring) without needing a full environment copy per candidate."""
        board, _captured = _apply(self.board, move)
        return _encode(board, 1 - self.current_player_index)

    def _observation(self) -> list[float]:
        return _encode(self.board, self.current_player_index)

    def render_state(self) -> dict:
        cells = {}
        for square, (owner, is_king) in self.board.items():
            color = "red" if owner == 0 else "black"
            kind = "king" if is_king else "man"
            cells[square] = f"{color}_{kind}"
        return {
            "width": 8,
            "height": 8,
            "cells": cells,
            "current_player": self.current_player_index,
            "winner": self._winner,
        }

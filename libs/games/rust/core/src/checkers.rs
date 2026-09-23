//! Checkers (American/English draughts), ported from `libs/games/src/games/checkers.py` -- see that
//! module's docstring for the rules and representation choices (a Move is a whole turn; captures are
//! mandatory and always maximal chains; draw by a moves-without-capture limit; observation = the 32
//! playable squares from the mover's perspective).
//!
//! One thing beyond the rules has to match too: **move order**. Strategies index into
//! `legal_moves()` (a seeded random one picks by position; a greedy one breaks ties by it), and the
//! Python original generated moves by walking a `dict` in insertion order -- which changes as pieces
//! move (a moved piece is re-inserted at the end). `Board` is therefore an insertion-ordered map with
//! exactly Python dict semantics, not a plain 8x8 array. `tests/test_native_parity.py` checks move
//! lists, observations, outcomes and render order against the original, game after game.

pub type Square = (i32, i32);
pub type Move = Vec<Square>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub owner: u8,
    pub king: bool,
}

/// An insertion-ordered map `Square -> Piece` with Python `dict` semantics: assigning an existing key
/// keeps its position, a new key goes last, deleting preserves the others' order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Board {
    cells: Vec<(Square, Piece)>,
}

impl Board {
    pub fn from_cells(cells: Vec<(Square, Piece)>) -> Board {
        let mut board = Board::default();
        for (square, piece) in cells {
            board.insert(square, piece);
        }
        board
    }

    pub fn get(&self, square: Square) -> Option<Piece> {
        self.cells.iter().find(|(s, _)| *s == square).map(|&(_, p)| p)
    }

    pub fn contains(&self, square: Square) -> bool {
        self.cells.iter().any(|(s, _)| *s == square)
    }

    pub fn insert(&mut self, square: Square, piece: Piece) {
        match self.cells.iter_mut().find(|(s, _)| *s == square) {
            Some(entry) => entry.1 = piece,
            None => self.cells.push((square, piece)),
        }
    }

    pub fn remove(&mut self, square: Square) -> Option<Piece> {
        let i = self.cells.iter().position(|(s, _)| *s == square)?;
        Some(self.cells.remove(i).1)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(Square, Piece)> {
        self.cells.iter()
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// The 32 dark squares, row by row -- the observation's order.
pub fn playable_squares() -> impl Iterator<Item = Square> {
    (0..8)
        .flat_map(|y| (0..8).map(move |x| (x, y)))
        .filter(|(x, y)| (x + y) % 2 == 1)
}

fn in_bounds((x, y): Square) -> bool {
    (0..8).contains(&x) && (0..8).contains(&y)
}

fn king_row(player: u8) -> i32 {
    if player == 0 {
        7
    } else {
        0
    }
}

fn directions(player: u8, king: bool) -> &'static [(i32, i32)] {
    if king {
        &[(1, 1), (-1, 1), (1, -1), (-1, -1)]
    } else if player == 0 {
        &[(1, 1), (-1, 1)]
    } else {
        &[(1, -1), (-1, -1)]
    }
}

/// Every maximal capture chain from `pos` (landing squares, `pos` excluded); empty if none. `board`
/// has the mover's own square cleared and every piece captured earlier in this chain removed.
fn capture_chains(pos: Square, player: u8, king: bool, board: &Board) -> Vec<Vec<Square>> {
    let mut chains = Vec::new();
    for &(dx, dy) in directions(player, king) {
        let over = (pos.0 + dx, pos.1 + dy);
        let landing = (pos.0 + 2 * dx, pos.1 + 2 * dy);
        if !in_bounds(landing) || board.contains(landing) {
            continue;
        }
        match board.get(over) {
            Some(occupant) if occupant.owner != player => {}
            _ => continue,
        }
        let mut next = board.clone();
        next.remove(over);
        let continuing_king = king || landing.1 == king_row(player);
        let further = capture_chains(landing, player, continuing_king, &next);
        if further.is_empty() {
            chains.push(vec![landing]);
        } else {
            for rest in further {
                let mut chain = vec![landing];
                chain.extend(rest);
                chains.push(chain);
            }
        }
    }
    chains
}

/// (resulting board, pieces captured). Errors on a move that doesn't start at a piece.
pub fn apply(board: &Board, mv: &[Square]) -> Result<(Board, u32), String> {
    if mv.len() < 2 {
        return Err(format!(
            "a move needs a start and at least one landing square, got {mv:?}"
        ));
    }
    let mut next = board.clone();
    let mut piece = next.remove(mv[0]).ok_or_else(|| format!("no piece on {:?}", mv[0]))?;
    let mut captured = 0;
    for pair in mv.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if (b.0 - a.0).abs() == 2 {
            next.remove(((a.0 + b.0).div_euclid(2), (a.1 + b.1).div_euclid(2)));
            captured += 1;
        }
    }
    let end = *mv.last().unwrap();
    if end.1 == king_row(piece.owner) {
        piece.king = true;
    }
    next.insert(end, piece);
    Ok((next, captured))
}

pub fn encode(board: &Board, perspective: u8) -> Vec<f64> {
    playable_squares()
        .map(|square| match board.get(square) {
            None => 0.0,
            Some(p) => {
                let sign = if p.owner == perspective { 1.0 } else { -1.0 };
                sign * if p.king { 2.0 } else { 1.0 }
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Checkers {
    pub board: Board,
    pub current_player: u8,
    pub max_moves_without_capture: u32,
    pub moves_without_capture: u32,
    pub winner: Option<u8>,
    pub done: bool,
}

impl Checkers {
    pub fn new(max_moves_without_capture: u32) -> Checkers {
        let mut game = Checkers {
            board: Board::default(),
            current_player: 0,
            max_moves_without_capture,
            moves_without_capture: 0,
            winner: None,
            done: false,
        };
        game.reset();
        game
    }

    pub fn reset(&mut self) {
        let mut board = Board::default();
        for (owner, rows) in [(0u8, 0..3), (1u8, 5..8)] {
            for y in rows {
                for x in 0..8 {
                    if (x + y) % 2 == 1 {
                        board.insert((x, y), Piece { owner, king: false });
                    }
                }
            }
        }
        self.board = board;
        self.current_player = 0;
        self.moves_without_capture = 0;
        self.winner = None;
        self.done = false;
    }

    /// Captures (maximal chains) if any piece of the mover has one, else simple moves -- in the
    /// board's insertion order.
    pub fn legal_moves(&self) -> Vec<Move> {
        let player = self.current_player;
        let mut captures = Vec::new();
        let mut simple = Vec::new();
        for &(pos, piece) in self.board.iter() {
            if piece.owner != player {
                continue;
            }
            let mut in_transit = self.board.clone();
            in_transit.remove(pos);
            let chains = capture_chains(pos, player, piece.king, &in_transit);
            if chains.is_empty() {
                for &(dx, dy) in directions(player, piece.king) {
                    let dest = (pos.0 + dx, pos.1 + dy);
                    if in_bounds(dest) && !self.board.contains(dest) {
                        simple.push(vec![pos, dest]);
                    }
                }
            } else {
                for chain in chains {
                    let mut mv = vec![pos];
                    mv.extend(chain);
                    captures.push(mv);
                }
            }
        }
        if captures.is_empty() {
            simple
        } else {
            captures
        }
    }

    /// Play a whole turn. Returns done; the winner (None = draw) is in `winner`.
    pub fn step(&mut self, mv: &[Square]) -> Result<bool, String> {
        let (board, captured) = apply(&self.board, mv)?;
        self.board = board;
        self.moves_without_capture = if captured > 0 {
            0
        } else {
            self.moves_without_capture + 1
        };
        self.current_player = 1 - self.current_player;
        if self.moves_without_capture >= self.max_moves_without_capture {
            self.done = true;
            self.winner = None;
        } else if self.legal_moves().is_empty() {
            self.done = true;
            self.winner = Some(1 - self.current_player);
        } else {
            self.done = false;
            self.winner = None;
        }
        Ok(self.done)
    }

    /// The observation `mv` would lead to (from the next mover's perspective), without playing it.
    pub fn simulate(&self, mv: &[Square]) -> Result<Vec<f64>, String> {
        let (board, _) = apply(&self.board, mv)?;
        Ok(encode(&board, 1 - self.current_player))
    }

    pub fn observation(&self) -> Vec<f64> {
        encode(&self.board, self.current_player)
    }

    /// (x, y, label) per piece in board order; labels `red_man`, `black_king`, ... (player 0 is red).
    pub fn cells(&self) -> Vec<(i32, i32, &'static str)> {
        self.board
            .iter()
            .map(|&((x, y), p)| {
                let label = match (p.owner, p.king) {
                    (0, false) => "red_man",
                    (0, true) => "red_king",
                    (_, false) => "black_man",
                    (_, true) => "black_king",
                };
                (x, y, label)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_has_seven_moves() {
        let game = Checkers::new(40);
        assert_eq!(game.board.len(), 24);
        let moves = game.legal_moves();
        assert_eq!(moves.len(), 7);
        assert!(moves.iter().all(|m| m.len() == 2));
    }

    #[test]
    fn double_jump_is_one_move() {
        let mut game = Checkers::new(40);
        game.board = Board::from_cells(vec![
            ((0, 0), Piece { owner: 0, king: false }),
            ((1, 1), Piece { owner: 1, king: false }),
            ((3, 3), Piece { owner: 1, king: false }),
        ]);
        assert_eq!(game.legal_moves(), vec![vec![(0, 0), (2, 2), (4, 4)]]);
        assert!(game.step(&[(0, 0), (2, 2), (4, 4)]).unwrap());
        assert_eq!(game.winner, Some(0));
    }

    #[test]
    fn dict_semantics_keep_insertion_order() {
        let mut board = Board::default();
        let p = Piece { owner: 0, king: false };
        board.insert((1, 0), p);
        board.insert((3, 0), p);
        board.insert((1, 0), Piece { owner: 0, king: true });
        assert_eq!(board.iter().map(|(s, _)| *s).collect::<Vec<_>>(), vec![(1, 0), (3, 0)]);
        board.remove((1, 0));
        board.insert((1, 0), p);
        assert_eq!(board.iter().map(|(s, _)| *s).collect::<Vec<_>>(), vec![(3, 0), (1, 0)]);
    }
}

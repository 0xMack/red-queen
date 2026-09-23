//! Checkers strategies, and the trained position evaluators they can search with (docs/design/0006, 0009):
//! the fixed opponents every Checkers experiment measures against, written once so training (Python, via
//! PyO3) and the browser (WebAssembly) play the *same* players.
//!
//! A `Strategy` owns its PRNG (PCG32, so a seed is the same player everywhere) and picks an *index* into the
//! game's current `legal_moves()` -- the list and order every caller already holds. Ties between equally
//! scored moves are broken uniformly at random.
//!
//! - `random`       -- a uniformly random legal move.
//! - `first-legal`  -- always the first (dumb but perfectly consistent).
//! - `material-1`   -- the move leaving the best material balance (men 1, kings 2). Barely beats random:
//!                     captures are mandatory, so whenever one exists every legal move is a capture.
//! - `material-2`   -- assumes the opponent answers with their material-minimizing reply and plays the
//!                     move that holds up: a tiny minimax.
//! - `material-N`   -- (N >= 3) the same idea searched N plies deep with alpha-beta: the *fair* opponent for
//!                     a trained evaluator that searches as deep.
//! - `evaluator`    -- a trained position evaluator (`Brain`), one ply ahead, or `depth` plies with alpha-beta
//!                     (depth 1 is the original one-ply strategy, kept exactly for parity with its oracle).
//!
//! A `Brain` is a trained network mapping the 32-square encoding (`checkers::encode`, from the mover's side)
//! to one score: either a fixed-topology `Network` (`evolve.WeightVector`'s flat layout, tanh on every layer)
//! or a `GraphNet` (a NEAT genome compiled to its evaluation plan, tanh on every node).

use crate::checkers::{encode, Checkers};
use crate::pcg::Pcg32;

/// Score of a move that wins outright, in the material searches.
pub const WIN_SCORE: f64 = 1000.0;

/// Number of network inputs: the 32 playable squares (`checkers::encode`).
pub const INPUTS: usize = 32;

/// A trained position evaluator: how good is this board for the player to move (`crate::nets`; only its first
/// output is read, and it must take the 32-square encoding -- see `check_evaluator`).
pub use crate::nets::Net as Brain;
pub use crate::nets::{GraphNet, Network};

/// A network can evaluate Checkers positions only if it reads the 32-square encoding; a layered one must have
/// exactly one output (a compiled NEAT graph may have more -- only the first is read).
fn check_evaluator(brain: &Brain) -> Result<(), String> {
    let (inputs, outputs) = (brain.inputs(), brain.outputs());
    if inputs != INPUTS || (matches!(brain, Brain::Layered(_)) && outputs != 1) {
        return Err(format!("an evaluator maps {INPUTS} inputs to 1 score, got {inputs} inputs and {outputs} outputs"));
    }
    Ok(())
}

/// What a search scores a leaf position with, always from the side to move.
#[derive(Clone, Debug)]
pub enum Eval {
    Material,
    Brain(Brain),
}

impl Eval {
    fn value(&self, game: &Checkers) -> f64 {
        let observation = encode(&game.board, game.current_player);
        match self {
            Eval::Material => observation.iter().sum(),
            Eval::Brain(brain) => brain.score(&observation),
        }
    }
}

/// The value of `game` for the player to move, searching `depth` more plies (negamax with alpha-beta). A
/// finished game is worth +/- `WIN_SCORE` (plus the depth left, so a quicker win is worth more), a draw 0.
fn search(game: &Checkers, eval: &Eval, depth: u32, mut alpha: f64, beta: f64) -> f64 {
    if game.done {
        return match game.winner {
            Some(w) if w == game.current_player => WIN_SCORE + depth as f64,
            Some(_) => -(WIN_SCORE + depth as f64),
            None => 0.0,
        };
    }
    if depth == 0 {
        return eval.value(game);
    }
    let mut best = f64::NEG_INFINITY;
    for mv in game.legal_moves() {
        let mut child = game.clone();
        child.step(&mv).unwrap();
        let value = -search(&child, eval, depth - 1, -beta, -alpha);
        best = best.max(value);
        alpha = alpha.max(value);
        if alpha >= beta {
            break;
        }
    }
    best
}

#[derive(Clone, Debug)]
pub enum Kind {
    Random,
    FirstLegal,
    Material1,
    Material2,
    /// A layered network, one ply ahead (the original evaluator strategy).
    Evaluator(Network),
    /// Alpha-beta to `depth` plies with `eval` at the leaves.
    Search { depth: u32, eval: Eval },
}

#[derive(Clone, Debug)]
pub struct Strategy {
    kind: Kind,
    rng: Pcg32,
}

fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

impl Strategy {
    /// `name` is one of `random`, `first-legal`, `material-1`, `material-2`, `material-N` (N >= 3), `evaluator`
    /// (which needs `network`, one ply).
    pub fn parse(name: &str, seed: u64, network: Option<Network>) -> Result<Strategy, String> {
        let kind = match name {
            "random" => Kind::Random,
            "first-legal" => Kind::FirstLegal,
            "material-1" => Kind::Material1,
            "material-2" => Kind::Material2,
            "evaluator" => {
                let network = network.ok_or("the evaluator strategy needs a network")?;
                check_evaluator(&Brain::Layered(network.clone()))?;
                Kind::Evaluator(network)
            }
            other => match other.strip_prefix("material-").and_then(|n| n.parse::<u32>().ok()) {
                Some(depth) if depth >= 3 => Kind::Search { depth, eval: Eval::Material },
                _ => return Err(format!("no such checkers strategy: {other}")),
            },
        };
        Ok(Strategy { kind, rng: Pcg32::new(seed) })
    }

    /// The one constructor the bindings share: `name` as in `parse`; a trained evaluator is described by
    /// either `weights` + `layer_sizes` (a layered network) or `graph` (`GraphNet::from_flat`), searched to
    /// `depth` plies. Only `evaluator` takes a network.
    pub fn build(
        name: &str,
        seed: u64,
        weights: Option<Vec<f64>>,
        layer_sizes: Option<Vec<usize>>,
        depth: u32,
        graph: Option<Vec<f64>>,
    ) -> Result<Strategy, String> {
        let brain = match (weights, layer_sizes, graph) {
            (Some(w), Some(l), None) => Some(Brain::Layered(Network::new(w, l)?)),
            (None, None, Some(g)) => Some(Brain::Graph(GraphNet::from_flat(&g)?)),
            (None, None, None) => None,
            _ => return Err("describe a network by weights + layer sizes, or by a graph -- not both, not half".into()),
        };
        match (name, brain) {
            ("evaluator", Some(brain)) => Strategy::evaluator(seed, brain, depth),
            ("evaluator", None) => Err("the evaluator strategy needs a network".into()),
            (name, None) => Strategy::parse(name, seed, None),
            (name, Some(_)) => Err(format!("{name} takes no network")),
        }
    }

    /// A trained evaluator searching `depth` plies (depth 1 with a layered network is the original one-ply
    /// `evaluator`, exactly).
    pub fn evaluator(seed: u64, brain: Brain, depth: u32) -> Result<Strategy, String> {
        if depth == 0 {
            return Err("search depth is at least 1".into());
        }
        check_evaluator(&brain)?;
        let kind = match (brain, depth) {
            (Brain::Layered(network), 1) => Kind::Evaluator(network),
            (brain, depth) => Kind::Search { depth, eval: Eval::Brain(brain) },
        };
        Ok(Strategy { kind, rng: Pcg32::new(seed) })
    }

    /// Index into `game.legal_moves()`. The game must have at least one legal move.
    pub fn pick(&mut self, game: &Checkers) -> usize {
        let moves = game.legal_moves();
        assert!(!moves.is_empty(), "no legal moves to pick from");
        match self.scores(game) {
            Some(scores) => self.best(&scores),
            None if matches!(self.kind, Kind::FirstLegal) => 0,
            None => self.rng.bounded(moves.len() as u32) as usize,
        }
    }

    /// What the strategy thinks each legal move is worth, in `legal_moves()` order (higher is better,
    /// from the mover's side) -- exactly what `pick` maximizes -- or `None` for a strategy that doesn't
    /// score moves (random, first-legal). This is the strategy's reasoning made visible: the page's
    /// diagnostics plot it.
    pub fn scores(&self, game: &Checkers) -> Option<Vec<f64>> {
        let moves = game.legal_moves();
        match &self.kind {
            Kind::Random | Kind::FirstLegal => None,
            Kind::Material1 => Some(moves.iter().map(|m| -sum(&game.simulate(m).unwrap())).collect()),
            // simulate() encodes the position from the *opponent's* side (they move next), so a move
            // is as good as that position is bad for them.
            Kind::Evaluator(network) => Some(moves.iter().map(|m| -network.score(&game.simulate(m).unwrap())).collect()),
            Kind::Material2 => {
                let me = game.current_player;
                Some(
                    moves
                        .iter()
                        .map(|m| {
                            let mut child = game.clone();
                            if child.step(m).unwrap() {
                                return if child.winner == Some(me) { WIN_SCORE } else { 0.0 };
                            }
                            // After the opponent's reply it's my move again, so simulate() is from my side.
                            child.legal_moves().iter().map(|r| sum(&child.simulate(r).unwrap())).fold(f64::INFINITY, f64::min)
                        })
                        .collect(),
                )
            }
            Kind::Search { depth, eval } => Some(
                moves
                    .iter()
                    .map(|m| {
                        let mut child = game.clone();
                        child.step(m).unwrap();
                        -search(&child, eval, depth - 1, f64::NEG_INFINITY, f64::INFINITY)
                    })
                    .collect(),
            ),
        }
    }

    /// The network's activations when it evaluated the position `moves[index]` leaves behind (as
    /// `simulate` encodes it, from the opponent's side), or `None` for a strategy without a fixed-topology
    /// network. This is the static view of `scores()[index]` for an evaluator, layer by layer (a search's
    /// value also looks deeper than this position).
    pub fn activations(&self, game: &Checkers, index: usize) -> Option<Vec<Vec<f64>>> {
        let network = match &self.kind {
            Kind::Evaluator(network) => network,
            Kind::Search { eval: Eval::Brain(Brain::Layered(network)), .. } => network,
            _ => return None,
        };
        let moves = game.legal_moves();
        Some(network.activations(&game.simulate(moves.get(index)?).unwrap()))
    }

    /// Index of the highest score, ties broken uniformly at random (reservoir sampling).
    fn best(&mut self, scores: &[f64]) -> usize {
        let mut best = 0;
        let mut ties = 1;
        for i in 1..scores.len() {
            if scores[i] > scores[best] {
                best = i;
                ties = 1;
            } else if scores[i] == scores[best] {
                ties += 1;
                if self.rng.bounded(ties) == 0 {
                    best = i;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkers::{Board, Piece};

    fn play(a: &mut Strategy, b: &mut Strategy) -> Option<u8> {
        let mut game = Checkers::new(40);
        for _ in 0..300 {
            let mover = if game.current_player == 0 { &mut *a } else { &mut *b };
            let index = mover.pick(&game);
            let moves = game.legal_moves();
            if game.step(&moves[index]).unwrap() {
                break;
            }
        }
        game.winner
    }

    fn material(depth: u32, seed: u64) -> Strategy {
        Strategy::parse(&format!("material-{depth}"), seed, None).unwrap()
    }

    #[test]
    fn first_legal_is_index_zero_and_random_is_reproducible() {
        let game = Checkers::new(40);
        assert_eq!(Strategy::parse("first-legal", 0, None).unwrap().pick(&game), 0);
        let picks = |seed| {
            let mut s = Strategy::parse("random", seed, None).unwrap();
            (0..20).map(|_| s.pick(&game)).collect::<Vec<_>>()
        };
        assert_eq!(picks(7), picks(7));
        assert!(picks(1).iter().any(|&i| i != 0), "random should not always pick the first move");
    }

    #[test]
    fn scores_are_what_pick_maximizes() {
        let game = Checkers::new(40);
        assert!(Strategy::parse("random", 0, None).unwrap().scores(&game).is_none());
        for name in ["material-1", "material-2", "material-3"] {
            let mut strategy = Strategy::parse(name, 3, None).unwrap();
            let scores = strategy.scores(&game).unwrap();
            assert_eq!(scores.len(), game.legal_moves().len());
            let picked = strategy.pick(&game);
            assert!(scores.iter().all(|&s| s <= scores[picked]), "{name} picked a lower-scoring move");
        }
    }

    #[test]
    fn material_2_takes_the_win() {
        // Red (0) man at (2,2) can capture (3,3) landing on (4,4) -- the last black piece.
        let mut game = Checkers::new(40);
        game.board = Board::from_cells(vec![((2, 2), Piece { owner: 0, king: false }), ((3, 3), Piece { owner: 1, king: false })]);
        let moves = game.legal_moves();
        let index = material(2, 0).pick(&game);
        assert_eq!(moves[index], vec![(2, 2), (4, 4)]);
    }

    #[test]
    fn two_ply_beats_random_from_both_seats() {
        let mut wins = 0;
        for seed in 0..6 {
            wins += (play(&mut material(2, seed), &mut Strategy::parse("random", seed + 100, None).unwrap()) == Some(0)) as u32;
            wins += (play(&mut Strategy::parse("random", seed + 100, None).unwrap(), &mut material(2, seed)) == Some(1)) as u32;
        }
        assert!(wins >= 11, "material-2 won only {wins}/12");
    }

    #[test]
    fn alpha_beta_equals_plain_minimax_and_depth_pays() {
        // Alpha-beta must return exactly the value of an unpruned minimax at the same depth.
        fn minimax(game: &Checkers, depth: u32) -> f64 {
            if game.done {
                return match game.winner {
                    Some(w) if w == game.current_player => WIN_SCORE + depth as f64,
                    Some(_) => -(WIN_SCORE + depth as f64),
                    None => 0.0,
                };
            }
            if depth == 0 {
                return Eval::Material.value(game);
            }
            game.legal_moves()
                .iter()
                .map(|m| {
                    let mut child = game.clone();
                    child.step(m).unwrap();
                    -minimax(&child, depth - 1)
                })
                .fold(f64::NEG_INFINITY, f64::max)
        }
        let mut game = Checkers::new(40);
        let mut rng = Pcg32::new(5);
        for _ in 0..12 {
            for depth in 1..=3 {
                let pruned = search(&game, &Eval::Material, depth, f64::NEG_INFINITY, f64::INFINITY);
                assert_eq!(pruned, minimax(&game, depth), "depth {depth}");
            }
            let moves = game.legal_moves();
            if moves.is_empty() || game.step(&moves[rng.bounded(moves.len() as u32) as usize]).unwrap() {
                break;
            }
        }
        // And depth pays: 4 plies beat 2 in a handful of games from either seat.
        let mut wins = 0;
        for seed in 0..4 {
            wins += (play(&mut material(4, seed), &mut material(2, seed + 9)) == Some(0)) as u32;
            wins += (play(&mut material(2, seed + 9), &mut material(4, seed)) == Some(1)) as u32;
        }
        assert!(wins >= 5, "material-4 won only {wins}/8 against material-2");
    }

    #[test]
    fn activations_end_in_the_score_and_only_layered_evaluators_have_them() {
        let game = Checkers::new(40);
        let mut weights = vec![0.0; 32 * 3 + 3 + 3 + 1];
        weights[0] = 0.4;
        weights[100] = -0.7;
        let network = Network::new(weights, vec![32, 3, 1]).unwrap();
        let evaluator = Strategy::parse("evaluator", 0, Some(network.clone())).unwrap();
        let scores = evaluator.scores(&game).unwrap();
        for (index, score) in scores.iter().enumerate() {
            let layers = evaluator.activations(&game, index).unwrap();
            assert_eq!(layers.iter().map(Vec::len).collect::<Vec<_>>(), vec![32, 3, 1]);
            assert_eq!(-layers[2][0], *score, "scores() is the negated network output for that move's position");
        }
        assert!(evaluator.activations(&game, 99).is_none());
        assert!(material(2, 0).activations(&game, 0).is_none());
        // A deeper search keeps the network, so the diagram still works.
        assert!(Strategy::evaluator(0, Brain::Layered(network), 3).unwrap().activations(&game, 0).is_some());
    }

    #[test]
    fn an_evaluator_must_read_the_32_square_encoding() {
        let network = |layers: Vec<usize>| Network::new(vec![0.0; layers[0] * layers[1] + layers[1]], layers).unwrap();
        assert!(Strategy::parse("evaluator", 0, Some(network(vec![31, 1]))).is_err());
        assert!(Strategy::evaluator(0, Brain::Layered(network(vec![32, 2])), 2).is_err());
        assert!(Strategy::evaluator(0, Brain::Layered(network(vec![32, 1])), 2).is_ok());
    }

    #[test]
    fn graph_net_matches_a_hand_computed_plan() {
        // slots: 0..32 inputs, 32 bias, 33 hidden, 34 output. hidden = tanh(2*in0 + 0.5*bias);
        // output = tanh(-1*hidden + 0.25*in1).
        let flat = [
            32.0, 35.0, 2.0, //
            33.0, 2.0, 0.0, 2.0, 32.0, 0.5, //
            34.0, 2.0, 33.0, -1.0, 1.0, 0.25, //
            1.0, 34.0,
        ];
        let net = GraphNet::from_flat(&flat).unwrap();
        let mut obs = vec![0.0; 32];
        obs[0] = 1.0;
        obs[1] = 2.0;
        let hidden = (2.0f64 + 0.5).tanh();
        assert!((net.score(&obs) - (-hidden + 0.5).tanh()).abs() < 1e-15);
        assert!(GraphNet::from_flat(&flat[..10]).is_err());
        let mut bad = flat;
        bad[11] = 34.0; // slot 34 reading slot 34 (itself)
        assert!(GraphNet::from_flat(&bad).is_err());
    }

}

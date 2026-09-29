//! Checkers by self-play (docs/design/0010 Phase 4): TD(λ) on position values, TD-Gammon style.
//!
//! The core's `Trainer` is single-agent; a two-player game needs its own loop, and it lives here because it needs
//! the games crate. One value network V scores a position for the player to move (`checkers::encode`, from the
//! mover's side) -- exactly the `Brain` the games crate's `evaluator` strategy searches with, in `evolve`'s flat
//! layout, tanh on every layer -- so a trained network plugs into the versus leaderboard and the Checkers page with
//! no new code.
//!
//! - **Play.** Both sides are the same network: a move is worth `-V(position it leaves)` (that position belongs to
//!   the opponent), and the mover takes the best, ties to the first. Exploration, since Checkers has no dice: the
//!   first `random_opening_plies` moves of every game are uniformly random, and after that each move is random with
//!   probability ε (decaying linearly over `epsilon_decay_games`).
//! - **Learn.** After a game, every position s_t it passed through gets a λ-return target, computed backwards: the
//!   last position before the end gets the outcome from its mover's side (+1 win, -1 loss, 0 draw), and
//!   `G_t = -((1 - λ) V(s_t+1) + λ G_t+1)` -- negated because s_t+1 belongs to the other player (zero-sum, no
//!   discount). That is TD(λ)'s forward view, applied offline once the game is over. One Adam step on
//!   `0.5 (V(s_t) - G_t)^2` averaged over the game's positions.
//! - **Opponent pool** (`pool_every` > 0): every `pool_every` games the current network is frozen into a pool of
//!   the last `pool_size`, and a `pool_fraction` of games are played against a random pool member instead of
//!   itself -- the hall-of-fame idea from the evolved runs. Every position still trains the current network.
//! - **TD-Leaf(λ)** (`search_depth` > 1; Baxter, Tridgell and Weaver's KnightCap): the network plays the way the
//!   leaderboard plays it -- alpha-beta, `search_depth` plies deep -- and learns *through* that search. A position's
//!   value is its searched value, and its gradient goes to the leaf of the principal variation (the position whose
//!   evaluation the search actually returned), sign-adjusted for whose move it is there. Plain TD trains the positions
//!   greedy one-ply play passes through; this trains the ones the search relies on. Depth 1 is the original loop.

use redqueen_games::checkers::{encode, Checkers};
use redqueen_rl::agent::Params;
use redqueen_rl::nn::{Activation, Adam, Mlp, Shape};
use redqueen_rl::rng::{Rng, Stream};

/// The 32 playable squares.
pub const INPUTS: usize = 32;

pub struct SelfPlay {
    net: Mlp,
    adam: Adam,
    lambda: f64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay_games: f64,
    random_opening_plies: u32,
    max_moves_without_capture: u32,
    max_plies: u32,
    pool_every: u64,
    pool_size: usize,
    pool_fraction: f64,
    pool: Vec<Mlp>,
    search_depth: u32,
    explore: Rng,
    games: u64,
}

/// A position's value under search, for TD-Leaf: from its mover's side, and the principal variation's leaf that
/// produced it (its encoding, and +1/-1 for whether the leaf's mover is this position's mover) -- or no leaf when the
/// search ended in a finished game (a fixed value, nothing to learn at).
struct Searched {
    value: f64,
    leaf: Option<(Vec<f64>, f64)>,
}

/// What `train` did.
#[derive(Clone, Debug, Default)]
pub struct TrainStats {
    pub games: u64,
    /// Games won by the side that moved first, the side that moved second, and drawn (the move cap included).
    pub first_wins: u64,
    pub second_wins: u64,
    pub draws: u64,
    pub mean_plies: f64,
    /// Mean squared distance from the λ-return targets, before each update.
    pub loss: f64,
    pub epsilon: f64,
    pub pool_games: u64,
}

impl SelfPlay {
    pub const PARAMS: [&'static str; 12] = [
        "hidden",
        "hidden_layers",
        "learning_rate",
        "lambda",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_games",
        "random_opening_plies",
        "pool_every",
        "pool_size",
        "pool_fraction",
        "search_depth",
    ];

    pub fn new(seed: u64, params: &Params, max_moves_without_capture: u32, max_plies: u32) -> Result<SelfPlay, String> {
        params.check("td_lambda", &Self::PARAMS)?;
        let whole = |key: &str, default: f64, min: f64| -> Result<usize, String> {
            let v = params.get(key, default);
            if v < min || v.fract() != 0.0 {
                return Err(format!("{key} must be a whole number >= {min}, got {v}"));
            }
            Ok(v as usize)
        };
        let hidden = vec![whole("hidden", 16.0, 1.0)?; whole("hidden_layers", 1.0, 1.0)?];
        let lambda = params.get("lambda", 0.7);
        if !(0.0..=1.0).contains(&lambda) {
            return Err(format!("lambda must be in [0, 1], got {lambda}"));
        }
        let mut sizes = vec![INPUTS];
        sizes.extend(&hidden);
        sizes.push(1);
        let layers = sizes.len() - 1;
        let shape = Shape::new(sizes, vec![Activation::Tanh; layers]).expect("a valid shape");
        let net = Mlp::init(shape, &mut Rng::new(seed, Stream::Init));
        Ok(SelfPlay {
            adam: Adam::new(net.params.len(), params.get("learning_rate", 1e-3)),
            net,
            lambda,
            epsilon_start: params.get("epsilon_start", 0.1),
            epsilon_end: params.get("epsilon_end", 0.02),
            epsilon_decay_games: params.get("epsilon_decay_games", 10_000.0).max(1.0),
            random_opening_plies: whole("random_opening_plies", 4.0, 0.0)? as u32,
            max_moves_without_capture,
            max_plies,
            pool_every: whole("pool_every", 0.0, 0.0)? as u64,
            pool_size: whole("pool_size", 10.0, 1.0)?,
            pool_fraction: params.get("pool_fraction", 0.5),
            pool: Vec::new(),
            search_depth: whole("search_depth", 1.0, 1.0)? as u32,
            explore: Rng::new(seed, Stream::Explore),
            games: 0,
        })
    }

    pub fn network(&self) -> &Mlp {
        &self.net
    }

    /// Replace the network's weights (same shape), to continue training a saved one -- e.g. fine-tuning a plain TD
    /// champion with TD-Leaf. The optimizer starts fresh.
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<(), String> {
        if weights.len() != self.net.params.len() {
            return Err(format!(
                "expected {} weights for layer sizes {:?}, got {}",
                self.net.params.len(),
                self.net.shape.layer_sizes,
                weights.len()
            ));
        }
        self.net.params = weights;
        self.adam = Adam::new(self.net.params.len(), self.adam.learning_rate);
        Ok(())
    }

    pub fn games_played(&self) -> u64 {
        self.games
    }

    fn epsilon(&self) -> f64 {
        let t = (self.games as f64 / self.epsilon_decay_games).min(1.0);
        self.epsilon_start + (self.epsilon_end - self.epsilon_start) * t
    }

    /// The move `net` would play: the one leaving the position worst for the opponent (ties to the first).
    fn best_move(net: &Mlp, game: &Checkers) -> usize {
        let moves = game.legal_moves();
        let mut best = (0, f64::NEG_INFINITY);
        for (i, mv) in moves.iter().enumerate() {
            let value = -net.forward(&game.simulate(mv).expect("a legal move"))[0];
            if value > best.1 {
                best = (i, value);
            }
        }
        best.0
    }

    /// Negamax with alpha-beta to `depth` plies, `net` at the leaves (a finished game is worth +1, -1 or 0: the ends
    /// of the net's tanh range), keeping the principal variation's leaf. Moves are tried in the engine's order and
    /// ties keep the first, as the games crate's evaluator strategy does.
    fn search(net: &Mlp, game: &Checkers, depth: u32, mut alpha: f64, beta: f64) -> Searched {
        if game.done {
            let value = match game.winner {
                Some(w) if w == game.current_player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            return Searched { value, leaf: None };
        }
        if depth == 0 {
            let observation = game.observation();
            let value = net.forward(&observation)[0];
            return Searched {
                value,
                leaf: Some((observation, 1.0)),
            };
        }
        let mut best = Searched {
            value: f64::NEG_INFINITY,
            leaf: None,
        };
        for mv in game.legal_moves() {
            let mut child = game.clone();
            child.step(&mv).expect("a legal move");
            let reply = Self::search(net, &child, depth - 1, -beta, -alpha);
            let value = -reply.value;
            if value > best.value {
                best = Searched {
                    value,
                    leaf: reply.leaf.map(|(leaf, sign)| (leaf, -sign)),
                };
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// The root of a search `depth` plies deep: the index of the best move, and the position's searched value.
    fn search_root(net: &Mlp, game: &Checkers, depth: u32) -> (usize, Searched) {
        let mut best = (
            0,
            Searched {
                value: f64::NEG_INFINITY,
                leaf: None,
            },
        );
        for (i, mv) in game.legal_moves().iter().enumerate() {
            let mut child = game.clone();
            child.step(mv).expect("a legal move");
            let reply = Self::search(net, &child, depth - 1, f64::NEG_INFINITY, -best.1.value);
            let value = -reply.value;
            if value > best.1.value {
                best = (
                    i,
                    Searched {
                        value,
                        leaf: reply.leaf.map(|(leaf, sign)| (leaf, -sign)),
                    },
                );
            }
        }
        best
    }

    /// One TD-Leaf game: every position's searched value (by the current network), the outcome for the last mover,
    /// the winner, and whether a pool opponent played. Moves: random in the opening and with probability ε, else the
    /// search's choice -- the pool opponent's own search when it's on move.
    fn play_searched(&mut self) -> (Vec<Searched>, f64, Option<u8>, bool) {
        let mut game = Checkers::new(self.max_moves_without_capture);
        let epsilon = self.epsilon();
        let depth = self.search_depth;
        let opponent = (!self.pool.is_empty() && self.explore.uniform() < self.pool_fraction).then(|| {
            let pick = self.explore.below(self.pool.len() as u32) as usize;
            (self.pool[pick].clone(), self.explore.below(2) as u8)
        });
        let mut positions = Vec::new();
        let mut plies = 0;
        while !game.done && plies < self.max_plies {
            let moves = game.legal_moves();
            let (mine, searched) = Self::search_root(&self.net, &game, depth);
            let random = plies < self.random_opening_plies || self.explore.uniform() < epsilon;
            let index = if random {
                self.explore.below(moves.len() as u32) as usize
            } else {
                match &opponent {
                    Some((net, seat)) if *seat == game.current_player => Self::search_root(net, &game, depth).0,
                    _ => mine,
                }
            };
            positions.push(searched);
            game.step(&moves[index]).expect("a legal move");
            plies += 1;
        }
        let last_mover = 1 - game.current_player;
        let winner = if game.done { game.winner } else { None };
        let outcome = match winner {
            Some(w) if w == last_mover => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
        (positions, outcome, winner, opponent.is_some())
    }

    /// TD-Leaf's update: λ-returns over the searched values, each position's error applied at its leaf.
    fn learn_searched(&mut self, positions: &[Searched], outcome: f64) -> f64 {
        let n = positions.len();
        if n == 0 {
            return 0.0;
        }
        let values: Vec<f64> = positions.iter().map(|p| p.value).collect();
        let targets = Self::lambda_returns(&values, outcome, self.lambda);
        let loss = values
            .iter()
            .zip(&targets)
            .map(|(v, g)| 0.5 * (v - g) * (v - g))
            .sum::<f64>()
            / n as f64;
        let learnable: Vec<(usize, &Vec<f64>, f64)> = positions
            .iter()
            .enumerate()
            .filter_map(|(t, p)| p.leaf.as_ref().map(|(leaf, sign)| (t, leaf, *sign)))
            .collect();
        if learnable.is_empty() {
            return loss;
        }
        let flat: Vec<f64> = learnable.iter().flat_map(|(_, leaf, _)| leaf.iter().copied()).collect();
        let cache = self.net.forward_batch(&flat, learnable.len());
        // d/dV of 0.5 (sign V(leaf) - G)^2 is sign (sign V(leaf) - G); sign V(leaf) is the searched value.
        let grad: Vec<f64> = learnable
            .iter()
            .zip(cache.outputs())
            .map(|((t, _, sign), v)| sign * (sign * v - targets[*t]) / n as f64)
            .collect();
        let grads = self.net.backward(&cache, &grad);
        self.adam.step(&mut self.net.params, &grads);
        loss
    }

    /// One game; returns the positions it passed through (each from its mover's side), the outcome for the player
    /// who moved last (+1 / 0 / -1), which side won (None = draw), and whether a pool opponent played.
    fn play(&mut self) -> (Vec<Vec<f64>>, f64, Option<u8>, bool) {
        let mut game = Checkers::new(self.max_moves_without_capture);
        let epsilon = self.epsilon();
        let opponent = (!self.pool.is_empty() && self.explore.uniform() < self.pool_fraction).then(|| {
            let pick = self.explore.below(self.pool.len() as u32) as usize;
            (self.pool[pick].clone(), self.explore.below(2) as u8) // the frozen net, and the seat it takes
        });
        let mut positions = Vec::new();
        let mut plies = 0;
        while !game.done && plies < self.max_plies {
            positions.push(game.observation());
            let moves = game.legal_moves();
            let random = plies < self.random_opening_plies || self.explore.uniform() < epsilon;
            let index = if random {
                self.explore.below(moves.len() as u32) as usize
            } else {
                match &opponent {
                    Some((net, seat)) if *seat == game.current_player => Self::best_move(net, &game),
                    _ => Self::best_move(&self.net, &game),
                }
            };
            game.step(&moves[index]).expect("a legal move");
            plies += 1;
        }
        // The last mover is the player *not* to move now. A cut-off game (max_plies) is a draw.
        let last_mover = 1 - game.current_player;
        let winner = if game.done { game.winner } else { None };
        let outcome = match winner {
            Some(w) if w == last_mover => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
        (positions, outcome, winner, opponent.is_some())
    }

    /// λ-return targets for one game's positions, given the network's values of them and the outcome for the last
    /// mover. Public for the tests and the Python oracle.
    pub fn lambda_returns(values: &[f64], outcome: f64, lambda: f64) -> Vec<f64> {
        let n = values.len();
        let mut targets = vec![0.0; n];
        if n == 0 {
            return targets;
        }
        targets[n - 1] = outcome;
        for t in (0..n - 1).rev() {
            targets[t] = -((1.0 - lambda) * values[t + 1] + lambda * targets[t + 1]);
        }
        targets
    }

    fn learn(&mut self, positions: &[Vec<f64>], outcome: f64) -> f64 {
        let n = positions.len();
        let flat: Vec<f64> = positions.iter().flatten().copied().collect();
        let cache = self.net.forward_batch(&flat, n);
        let values = cache.outputs().to_vec();
        let targets = Self::lambda_returns(&values, outcome, self.lambda);
        let grad: Vec<f64> = values.iter().zip(&targets).map(|(v, g)| (v - g) / n as f64).collect();
        let loss = values
            .iter()
            .zip(&targets)
            .map(|(v, g)| 0.5 * (v - g) * (v - g))
            .sum::<f64>()
            / n as f64;
        let grads = self.net.backward(&cache, &grad);
        self.adam.step(&mut self.net.params, &grads);
        loss
    }

    /// Play and learn from `games` games.
    pub fn train(&mut self, games: u64) -> TrainStats {
        let mut stats = TrainStats::default();
        let (mut plies, mut loss) = (0usize, 0.0);
        for _ in 0..games {
            let (n, game_loss, winner, pooled) = if self.search_depth > 1 {
                let (positions, outcome, winner, pooled) = self.play_searched();
                (
                    positions.len(),
                    self.learn_searched(&positions, outcome),
                    winner,
                    pooled,
                )
            } else {
                let (positions, outcome, winner, pooled) = self.play();
                (positions.len(), self.learn(&positions, outcome), winner, pooled)
            };
            plies += n;
            loss += game_loss;
            match winner {
                Some(0) => stats.first_wins += 1,
                Some(_) => stats.second_wins += 1,
                None => stats.draws += 1,
            }
            stats.pool_games += pooled as u64;
            self.games += 1;
            if self.pool_every > 0 && self.games.is_multiple_of(self.pool_every) {
                self.pool.push(self.net.clone());
                if self.pool.len() > self.pool_size {
                    self.pool.remove(0);
                }
            }
        }
        stats.games = games;
        stats.mean_plies = plies as f64 / games.max(1) as f64;
        stats.loss = loss / games.max(1) as f64;
        stats.epsilon = self.epsilon();
        stats
    }

    /// The network as `evolve.WeightVector` JSON: what the Checkers evaluator strategy, the versus leaderboard and the
    /// page load (`{"weights", "layer_sizes"}`; tanh on every layer is the format's convention, and this net's).
    pub fn snapshot(&self) -> String {
        let weights: Vec<String> = self.net.params.iter().map(|w| format!("{w:?}")).collect();
        let sizes: Vec<String> = self.net.shape.layer_sizes.iter().map(|s| s.to_string()).collect();
        format!(
            r#"{{"weights": [{}], "layer_sizes": [{}]}}"#,
            weights.join(", "),
            sizes.join(", ")
        )
    }

    /// Points per game (win 1, draw 1/2, loss 0) for the current network, searching `depth` plies as the games
    /// crate's `evaluator` strategy (exactly how the leaderboard plays it), against the fixed strategy `opponent`
    /// (`material-2`, ...) over `games` games, seats alternating, each game seeded from `seed`.
    pub fn points_against(&self, opponent: &str, depth: u32, games: u32, seed: u64) -> Result<f64, String> {
        use redqueen_games::checkers_strategies::{Brain, Network, Strategy};
        let brain = Brain::Layered(Network::new(
            self.net.params.clone(),
            self.net.shape.layer_sizes.clone(),
        )?);
        let mut points = 0.0;
        for g in 0..games {
            let game_seed = seed + g as u64;
            let mut me = Strategy::evaluator(game_seed, brain.clone(), depth)?;
            let mut other = Strategy::parse(opponent, game_seed + 7919, None)?;
            let mut game = Checkers::new(self.max_moves_without_capture);
            let seat = (g % 2) as u8;
            let mut plies = 0;
            while !game.done && plies < self.max_plies {
                let moves = game.legal_moves();
                let index = if game.current_player == seat {
                    me.pick(&game)
                } else {
                    other.pick(&game)
                };
                game.step(&moves[index])?;
                plies += 1;
            }
            points += match (game.done, game.winner) {
                (true, Some(w)) if w == seat => 1.0,
                (true, Some(_)) => 0.0,
                _ => 0.5,
            };
        }
        Ok(points / games.max(1) as f64)
    }

    /// The value of the starting position for the side to move, and of a position one side is a king up -- two
    /// numbers a run can log to see the network's sense of the game develop.
    pub fn probe(&self) -> (f64, f64) {
        let game = Checkers::new(self.max_moves_without_capture);
        let start = self.net.forward(&game.observation())[0];
        let mut up = encode(&game.board, game.current_player);
        up[0] = 2.0; // a king of our own where a man was
        (start, self.net.forward(&up)[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(pairs: &[(&str, f64)]) -> Params {
        Params::new(pairs.iter().map(|&(k, v)| (k.to_string(), v)))
    }

    #[test]
    fn lambda_returns_alternate_sign_and_end_at_the_outcome() {
        // λ = 0: one-step TD -- each target is minus the next position's value; the last is the outcome
        assert_eq!(
            SelfPlay::lambda_returns(&[0.1, 0.2, 0.3], 1.0, 0.0),
            vec![-0.2, -0.3, 1.0]
        );
        // λ = 1: Monte-Carlo -- the outcome, sign flipping back every ply
        assert_eq!(
            SelfPlay::lambda_returns(&[0.1, 0.2, 0.3], 1.0, 1.0),
            vec![1.0, -1.0, 1.0]
        );
        let mid = SelfPlay::lambda_returns(&[0.1, 0.2, 0.3], -1.0, 0.5);
        assert_eq!(mid[2], -1.0);
        assert!((mid[1] - -(0.5 * 0.3 - 0.5)).abs() < 1e-15);
        assert!((mid[0] - -(0.5 * 0.2 + 0.5 * mid[1])).abs() < 1e-15);
    }

    /// Points per game (win 1, draw 1/2) for `a` against `b`, both greedy, seats alternating; each game opens with
    /// two random plies (seeded), or two deterministic players would replay one game.
    fn head_to_head(a: &Mlp, b: &Mlp, games: u32) -> f64 {
        let mut rng = Rng::new(99, Stream::Data);
        let mut points = 0.0;
        for g in 0..games {
            let mut game = Checkers::new(40);
            let seat = (g % 2) as u8;
            let mut plies = 0;
            while !game.done && plies < 200 {
                let moves = game.legal_moves();
                let index = if plies < 2 {
                    rng.below(moves.len() as u32) as usize
                } else {
                    SelfPlay::best_move(if game.current_player == seat { a } else { b }, &game)
                };
                game.step(&moves[index]).unwrap();
                plies += 1;
            }
            points += match (game.done, game.winner) {
                (true, Some(w)) if w == seat => 1.0,
                (true, Some(_)) => 0.0,
                _ => 0.5,
            };
        }
        points / games as f64
    }

    #[test]
    fn self_play_beats_its_own_starting_point() {
        let mut trainer = SelfPlay::new(1, &params(&[]), 40, 200).unwrap();
        let untrained = trainer.net.clone();
        let stats = trainer.train(3000);
        assert_eq!(stats.first_wins + stats.second_wins + stats.draws, 3000);
        assert!(stats.mean_plies > 10.0 && stats.loss.is_finite());
        // Deterministic (every draw is seeded): 0.595 here. Self-play keeps improving well past this -- ~0.8 by 30k
        // games -- but a unit test can't afford that; the versus leaderboard is where strength is measured.
        let score = head_to_head(&trainer.net, &untrained, 100);
        assert!(score > 0.55, "{score}");
        // and through the games crate's own evaluator strategy, as the leaderboard plays it
        let vs_random = trainer.points_against("random", 1, 20, 5).unwrap();
        assert!(vs_random > 0.65, "{vs_random}"); // 0.75 (deterministic): well clear of chance, early in training
        assert!(trainer.points_against("nonsense", 1, 2, 0).is_err());
        let json = trainer.snapshot();
        assert!(json.starts_with(r#"{"weights": ["#) && json.ends_with(r#""layer_sizes": [32, 16, 1]}"#));
    }

    /// Plain negamax without pruning, for checking the alpha-beta search.
    fn minimax(net: &Mlp, game: &Checkers, depth: u32) -> f64 {
        if game.done {
            return match game.winner {
                Some(w) if w == game.current_player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
        }
        if depth == 0 {
            return net.forward(&game.observation())[0];
        }
        game.legal_moves()
            .iter()
            .map(|mv| {
                let mut child = game.clone();
                child.step(mv).unwrap();
                -minimax(net, &child, depth - 1)
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn td_leaf_search_is_minimax_and_its_value_is_its_leafs() {
        let trainer = SelfPlay::new(3, &params(&[]), 40, 200).unwrap();
        let mut rng = Rng::new(5, Stream::Data);
        let mut game = Checkers::new(40);
        let mut checked = 0;
        for _ in 0..40 {
            if game.done {
                break;
            }
            for depth in 1..=3 {
                let (_, searched) = SelfPlay::search_root(&trainer.net, &game, depth);
                let exact = minimax(&trainer.net, &game, depth);
                assert!(
                    (searched.value - exact).abs() < 1e-12,
                    "depth {depth}: {} vs {exact}",
                    searched.value
                );
                // the value the search returned is its principal leaf's evaluation, sign-adjusted
                if let Some((leaf, sign)) = &searched.leaf {
                    assert!((sign * trainer.net.forward(leaf)[0] - searched.value).abs() < 1e-12);
                    checked += 1;
                }
            }
            let moves = game.legal_moves();
            game.step(&moves[rng.below(moves.len() as u32) as usize]).unwrap();
        }
        assert!(checked > 60, "{checked}");
    }

    #[test]
    fn td_leaf_trains_and_weights_can_be_loaded() {
        let mut trainer = SelfPlay::new(4, &params(&[("search_depth", 2.0)]), 40, 200).unwrap();
        let untrained = trainer.net.clone();
        let stats = trainer.train(1000);
        assert_eq!(stats.first_wins + stats.second_wins + stats.draws, 1000);
        assert!(stats.loss.is_finite() && trainer.net.params != untrained.params);
        // Deterministic: ~0.66 after 1k games (0.55 after 300 -- a searched game teaches slower per game at first).
        let score = head_to_head(&trainer.net, &untrained, 100);
        assert!(score > 0.58, "{score}");

        let mut other = SelfPlay::new(9, &params(&[]), 40, 200).unwrap();
        other.set_weights(trainer.net.params.clone()).unwrap();
        assert_eq!(other.snapshot(), trainer.snapshot());
        assert!(other.set_weights(vec![0.0; 3]).is_err());
        assert!(SelfPlay::new(0, &params(&[("search_depth", 0.0)]), 40, 200).is_err());
    }

    #[test]
    fn pools_fill_and_bad_params_are_rejected() {
        let mut trainer = SelfPlay::new(
            2,
            &params(&[("pool_every", 20.0), ("pool_size", 3.0), ("pool_fraction", 1.0)]),
            40,
            200,
        )
        .unwrap();
        let stats = trainer.train(100);
        assert_eq!(trainer.pool.len(), 3);
        assert!(
            stats.pool_games > 50,
            "every game after the first snapshot is a pool game: {}",
            stats.pool_games
        );
        assert!(SelfPlay::new(0, &params(&[("lambda", 1.5)]), 40, 200).is_err());
        assert!(SelfPlay::new(0, &params(&[("alpha", 0.1)]), 40, 200).is_err());
    }
}

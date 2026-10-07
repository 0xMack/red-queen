//! Checkers by AlphaZero-style self-play (docs/design/0017): a policy and a value from one network, Monte-Carlo tree
//! search guided by both, and the search's results as the training targets.
//!
//! TD(λ) and TD-Leaf (`selfplay.rs`) learn a value only, and the search only chooses which position to learn at. Here
//! the search *is* the teacher: every self-play move is a PUCT search, the visit counts it ends with become the
//! policy's target, and the game's outcome the value's.
//!
//! - **Network.** One MLP, tanh hidden layers, a linear output layer of `1 + POLICY_SIZE` units: unit 0 is the value
//!   (tanh of it, from the mover's side, like `selfplay.rs`'s), the rest are move logits. The trunk and the value
//!   unit together are exactly a `checkers::evaluator` network (tanh on every layer), so `snapshot()` exports them as
//!   `evolve.WeightVector` JSON and a trained value head joins the versus leaderboard, the head-to-head SPRT and the
//!   page with no new code. `network_json()` is the whole two-headed network.
//! - **Moves as indices.** A move is a whole turn (a capture chain is one move), so the policy indexes a move by its
//!   start square (0..32, `playable_squares` order) and its first step's direction (4): `POLICY_SIZE` = 128. Two
//!   chains that start alike and branch later share an index; they share its probability equally, and their visits
//!   are summed into its target. That is rare (it needs a branching multi-jump) and costs only resolution.
//! - **Search (PUCT).** From the root, descend by `Q + c_puct P sqrt(N) / (1 + N_a)` (Q from the mover's side,
//!   0 for an unvisited move; ties to the first), expand one new position by the network (or score a finished game
//!   +1 / -1 / 0), and back the value up with its sign flipping each ply. `simulations` descents per move. In
//!   self-play the root's priors are mixed with Dirichlet noise (`dirichlet_alpha`, `dirichlet_weight`), so the
//!   search tries moves the network doesn't yet like. A position with one legal move isn't searched.
//! - **Play.** The first `random_opening_plies` moves are uniformly random (not searched, not learned from); then for
//!   `temperature_plies` plies the move is drawn in proportion to its visits, after that it is the most visited.
//! - **Learn.** Each searched position is stored with its visit distribution and, once the game ends, its outcome from
//!   its mover's side. After every game, `updates_per_game` Adam steps on minibatches of `batch_size` drawn from the
//!   last `buffer_size` positions: `value_weight (v - z)^2` plus the cross-entropy of the policy (a softmax over the
//!   legal moves' indices only) against the visit distribution.

use redqueen_games::checkers::{playable_squares, Checkers, Move, Square};
use redqueen_rl::agent::Params;
use redqueen_rl::nn::{Activation, Adam, Mlp, Shape};
use redqueen_rl::rng::{Rng, Stream};

/// The 32 playable squares.
pub const INPUTS: usize = 32;
/// 32 start squares x 4 first-step directions.
pub const POLICY_SIZE: usize = 128;

/// A move's policy index: its start square and the direction of its first step.
pub fn move_index(mv: &[Square]) -> usize {
    let square = playable_squares()
        .position(|s| s == mv[0])
        .expect("a move starts on a playable square");
    let (dx, dy) = (mv[1].0 - mv[0].0, mv[1].1 - mv[0].1);
    let direction = match (dx > 0, dy > 0) {
        (true, true) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (false, false) => 3,
    };
    square * 4 + direction
}

/// The distinct policy indices of `moves`, in first-seen order, and each move's position in that list.
fn unique_indices(moves: &[Move]) -> (Vec<usize>, Vec<usize>) {
    let mut unique: Vec<usize> = Vec::new();
    let slots = moves
        .iter()
        .map(|mv| {
            let index = move_index(mv);
            match unique.iter().position(|&u| u == index) {
                Some(slot) => slot,
                None => {
                    unique.push(index);
                    unique.len() - 1
                }
            }
        })
        .collect();
    (unique, slots)
}

/// A softmax over `logits` restricted to `indices`, in their order.
fn masked_softmax(logits: &[f64], indices: &[usize]) -> Vec<f64> {
    let top = indices.iter().map(|&i| logits[i]).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = indices.iter().map(|&i| libm::exp(logits[i] - top)).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}

/// Gamma(shape, 1) by Marsaglia and Tsang (shape < 1 by boosting from shape + 1).
fn gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = 1.0 - rng.uniform(); // (0, 1]
        return gamma(rng, shape + 1.0) * libm::pow(u, 1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / libm::sqrt(9.0 * d);
    loop {
        let x = rng.normal();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = 1.0 - rng.uniform();
        if libm::log(u) < 0.5 * x * x + d - d * v + d * libm::log(v) {
            return d * v;
        }
    }
}

/// A draw from a symmetric Dirichlet over `n` outcomes.
fn dirichlet(rng: &mut Rng, alpha: f64, n: usize) -> Vec<f64> {
    let draws: Vec<f64> = (0..n).map(|_| gamma(rng, alpha)).collect();
    let total: f64 = draws.iter().sum();
    if total > 0.0 {
        draws.iter().map(|g| g / total).collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// The value of a finished game for the player to move in it.
fn terminal_value(game: &Checkers) -> f64 {
    match game.winner {
        Some(w) if w == game.current_player => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    }
}

/// The network's two heads at one position: the value (tanh, mover's side) and a prior per legal move.
pub struct Evaluation {
    pub value: f64,
    pub priors: Vec<f64>,
}

/// Value and move priors for `game`'s position (not finished), from `net`'s raw outputs.
pub fn evaluate(net: &Mlp, game: &Checkers, moves: &[Move]) -> Evaluation {
    let out = net.forward(&game.observation());
    let (unique, slots) = unique_indices(moves);
    let probabilities = masked_softmax(&out[1..], &unique);
    let mut shares = vec![0usize; unique.len()];
    slots.iter().for_each(|&s| shares[s] += 1);
    Evaluation {
        value: libm::tanh(out[0]),
        priors: slots.iter().map(|&s| probabilities[s] / shares[s] as f64).collect(),
    }
}

/// One position in the search tree. Per move: the prior, visits, the sum of values backed up through it (from this
/// position's mover's side), and the child once expanded.
struct Node {
    game: Checkers,
    moves: Vec<Move>,
    priors: Vec<f64>,
    visits: Vec<u32>,
    value_sum: Vec<f64>,
    children: Vec<Option<usize>>,
    total_visits: u32,
}

/// A PUCT search tree over one root position.
pub struct Search {
    nodes: Vec<Node>,
    c_puct: f64,
}

/// What a finished search says about its root.
pub struct SearchResult {
    pub moves: Vec<Move>,
    pub visits: Vec<u32>,
    /// The network's value of the root, and the search's (visit-weighted mean of the backed-up values).
    pub network_value: f64,
    pub value: f64,
}

impl Search {
    /// A tree whose root is `game` (not finished), expanded by `net`; `noise` mixes Dirichlet noise into the root's
    /// priors: (alpha, weight, rng).
    pub fn new(net: &Mlp, game: &Checkers, c_puct: f64, noise: Option<(f64, f64, &mut Rng)>) -> (Search, f64) {
        let moves = game.legal_moves();
        let Evaluation { value, mut priors } = evaluate(net, game, &moves);
        if let Some((alpha, weight, rng)) = noise {
            let eta = dirichlet(rng, alpha, moves.len());
            for (p, n) in priors.iter_mut().zip(eta) {
                *p = (1.0 - weight) * *p + weight * n;
            }
        }
        let root = Node {
            game: game.clone(),
            visits: vec![0; moves.len()],
            value_sum: vec![0.0; moves.len()],
            children: vec![None; moves.len()],
            moves,
            priors,
            total_visits: 0,
        };
        (
            Search {
                nodes: vec![root],
                c_puct,
            },
            value,
        )
    }

    /// The move PUCT descends through at `node`.
    fn select(&self, node: &Node) -> usize {
        let scale = self.c_puct * libm::sqrt(node.total_visits as f64);
        let mut best = (0, f64::NEG_INFINITY);
        for i in 0..node.moves.len() {
            let n = node.visits[i];
            let q = if n > 0 { node.value_sum[i] / n as f64 } else { 0.0 };
            let score = q + scale * node.priors[i] / (1.0 + n as f64);
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    }

    /// One descent: select to an unexpanded move, expand it, back its value up. Returns the value backed into the
    /// root (from the root mover's side).
    pub fn simulate(&mut self, net: &Mlp) -> f64 {
        let mut path: Vec<(usize, usize)> = Vec::new(); // (node, move)
        let mut at = 0;
        // the value of the position reached, from *its* mover's side
        let mut value = loop {
            let choice = self.select(&self.nodes[at]);
            path.push((at, choice));
            match self.nodes[at].children[choice] {
                Some(child) => at = child,
                None => {
                    let mut game = self.nodes[at].game.clone();
                    game.step(&self.nodes[at].moves[choice]).expect("a legal move");
                    if game.done {
                        break terminal_value(&game);
                    }
                    let moves = game.legal_moves();
                    let Evaluation { value, priors } = evaluate(net, &game, &moves);
                    let child = Node {
                        game,
                        visits: vec![0; moves.len()],
                        value_sum: vec![0.0; moves.len()],
                        children: vec![None; moves.len()],
                        moves,
                        priors,
                        total_visits: 0,
                    };
                    self.nodes.push(child);
                    let index = self.nodes.len() - 1;
                    self.nodes[at].children[choice] = Some(index);
                    break value;
                }
            }
        };
        for &(node, choice) in path.iter().rev() {
            value = -value; // now from `node`'s mover's side: the position below belonged to the other player
            let node = &mut self.nodes[node];
            node.visits[choice] += 1;
            node.value_sum[choice] += value;
            node.total_visits += 1;
        }
        value
    }

    pub fn result(&self, network_value: f64) -> SearchResult {
        let root = &self.nodes[0];
        let total = root.total_visits.max(1) as f64;
        SearchResult {
            moves: root.moves.clone(),
            visits: root.visits.clone(),
            network_value,
            value: root.value_sum.iter().sum::<f64>() / total,
        }
    }
}

/// `simulations` descents from `game` (not finished).
pub fn search(
    net: &Mlp,
    game: &Checkers,
    simulations: u32,
    c_puct: f64,
    noise: Option<(f64, f64, &mut Rng)>,
) -> SearchResult {
    let (mut tree, network_value) = Search::new(net, game, c_puct, noise);
    for _ in 0..simulations {
        tree.simulate(net);
    }
    tree.result(network_value)
}

/// The most visited move (ties to the first).
pub fn most_visited(visits: &[u32]) -> usize {
    let mut best = 0;
    for (i, &v) in visits.iter().enumerate() {
        if v > visits[best] {
            best = i;
        }
    }
    best
}

/// One training example: the position, its legal moves' distinct policy indices, the visit distribution over them,
/// and the outcome from its mover's side.
#[derive(Clone, Debug)]
pub struct Example {
    pub observation: Vec<f64>,
    pub indices: Vec<usize>,
    pub policy: Vec<f64>,
    pub outcome: f64,
}

/// Loss and gradients with respect to the network's raw outputs, for a batch of examples whose outputs are
/// `outputs` (`examples.len()` rows of `1 + POLICY_SIZE`): mean over the batch of `value_weight (tanh(o_0) - z)^2`
/// plus the cross-entropy of the legal-move softmax against the target. Returns (value loss, policy loss, gradient).
/// A pure function, so the tests can check it by finite differences.
pub fn az_gradients(outputs: &[f64], examples: &[Example], value_weight: f64) -> (f64, f64, Vec<f64>) {
    let width = 1 + POLICY_SIZE;
    let n = examples.len() as f64;
    let mut grad = vec![0.0; outputs.len()];
    let (mut value_loss, mut policy_loss) = (0.0, 0.0);
    for (row, example) in examples.iter().enumerate() {
        let out = &outputs[row * width..(row + 1) * width];
        let g = &mut grad[row * width..(row + 1) * width];
        let v = libm::tanh(out[0]);
        value_loss += value_weight * (v - example.outcome) * (v - example.outcome) / n;
        g[0] = value_weight * 2.0 * (v - example.outcome) * (1.0 - v * v) / n;
        let p = masked_softmax(&out[1..], &example.indices);
        for ((&index, &pi), &q) in example.indices.iter().zip(&example.policy).zip(&p) {
            if pi > 0.0 {
                policy_loss -= pi * libm::log(q) / n;
            }
            g[1 + index] = (q - pi) / n;
        }
    }
    (value_loss, policy_loss, grad)
}

pub struct AlphaZero {
    net: Mlp,
    adam: Adam,
    simulations: u32,
    c_puct: f64,
    dirichlet_alpha: f64,
    dirichlet_weight: f64,
    temperature_plies: u32,
    random_opening_plies: u32,
    buffer_size: usize,
    batch_size: usize,
    updates_per_game: u32,
    value_weight: f64,
    max_moves_without_capture: u32,
    max_plies: u32,
    buffer: Vec<Example>,
    /// Where the next example overwrites, once the buffer is full.
    cursor: usize,
    explore: Rng,
    replay: Rng,
    games: u64,
}

/// What `train` did.
#[derive(Clone, Debug, Default)]
pub struct TrainStats {
    pub games: u64,
    pub first_wins: u64,
    pub second_wins: u64,
    pub draws: u64,
    pub mean_plies: f64,
    /// Mean losses over this call's updates (0 before the buffer holds a batch).
    pub value_loss: f64,
    pub policy_loss: f64,
    pub updates: u64,
    /// Positions searched, and the mean of how far the search's root value moved from the network's.
    pub searched: u64,
    pub search_shift: f64,
}

impl AlphaZero {
    pub const PARAMS: [&'static str; 14] = [
        "hidden",
        "hidden_layers",
        "learning_rate",
        "simulations",
        "c_puct",
        "dirichlet_alpha",
        "dirichlet_weight",
        "temperature_plies",
        "random_opening_plies",
        "buffer_size",
        "batch_size",
        "updates_per_game",
        "value_weight",
        "seed_offset",
    ];

    pub fn new(
        seed: u64,
        params: &Params,
        max_moves_without_capture: u32,
        max_plies: u32,
    ) -> Result<AlphaZero, String> {
        params.check("alphazero", &Self::PARAMS)?;
        let whole = |key: &str, default: f64, min: f64| -> Result<usize, String> {
            let v = params.get(key, default);
            if v < min || v.fract() != 0.0 {
                return Err(format!("{key} must be a whole number >= {min}, got {v}"));
            }
            Ok(v as usize)
        };
        let positive = |key: &str, default: f64| -> Result<f64, String> {
            match params.get(key, default) {
                v if v > 0.0 => Ok(v),
                v => Err(format!("{key} must be > 0, got {v}")),
            }
        };
        let hidden = vec![whole("hidden", 64.0, 1.0)?; whole("hidden_layers", 2.0, 1.0)?];
        let dirichlet_weight = params.get("dirichlet_weight", 0.25);
        if !(0.0..=1.0).contains(&dirichlet_weight) {
            return Err(format!("dirichlet_weight must be in [0, 1], got {dirichlet_weight}"));
        }
        let net = Mlp::init(Self::shape(&hidden), &mut Rng::new(seed, Stream::Init));
        Ok(AlphaZero {
            adam: Adam::new(net.params.len(), positive("learning_rate", 1e-3)?),
            net,
            simulations: whole("simulations", 50.0, 1.0)? as u32,
            c_puct: positive("c_puct", 1.5)?,
            dirichlet_alpha: positive("dirichlet_alpha", 1.0)?,
            dirichlet_weight,
            temperature_plies: whole("temperature_plies", 10.0, 0.0)? as u32,
            random_opening_plies: whole("random_opening_plies", 0.0, 0.0)? as u32,
            buffer_size: whole("buffer_size", 50_000.0, 1.0)?,
            batch_size: whole("batch_size", 64.0, 1.0)?,
            updates_per_game: whole("updates_per_game", 4.0, 0.0)? as u32,
            value_weight: positive("value_weight", 1.0)?,
            max_moves_without_capture,
            max_plies,
            buffer: Vec::new(),
            cursor: 0,
            // `seed_offset` (default 0) shifts only the self-play draws, not the initial weights: a second run from
            // the same starting network that explores differently.
            explore: Rng::new(seed + whole("seed_offset", 0.0, 0.0)? as u64, Stream::Explore),
            replay: Rng::new(seed, Stream::Replay),
            games: 0,
        })
    }

    fn shape(hidden: &[usize]) -> Shape {
        let mut sizes = vec![INPUTS];
        sizes.extend(hidden);
        sizes.push(1 + POLICY_SIZE);
        let mut activations = vec![Activation::Tanh; hidden.len()];
        activations.push(Activation::Linear);
        Shape::new(sizes, activations).expect("a valid shape")
    }

    pub fn network(&self) -> &Mlp {
        &self.net
    }

    pub fn games_played(&self) -> u64 {
        self.games
    }

    /// Start from a trained value network (`evolve.WeightVector`: 32 -> hidden... -> 1, tanh): its hidden layers
    /// become the trunk and its output unit the value unit; the policy starts uniform (zero weights). Lets search
    /// targets fine-tune a TD champion, as TD-Leaf did. The optimizer starts fresh.
    pub fn set_value_network(&mut self, weights: &[f64], layer_sizes: &[usize]) -> Result<(), String> {
        let sizes = &self.net.shape.layer_sizes;
        let hidden = &sizes[..sizes.len() - 1];
        if layer_sizes.len() != sizes.len()
            || &layer_sizes[..layer_sizes.len() - 1] != hidden
            || layer_sizes.last() != Some(&1)
        {
            return Err(format!(
                "expected a value network with layer sizes {hidden:?} + [1], got {layer_sizes:?}"
            ));
        }
        let trunk: usize = hidden.windows(2).map(|w| w[1] * (w[0] + 1)).sum();
        let last = *hidden.last().unwrap();
        if weights.len() != trunk + last + 1 {
            return Err(format!("expected {} weights, got {}", trunk + last + 1, weights.len()));
        }
        let mut params = weights[..trunk].to_vec();
        let mut out_weights = vec![0.0; (1 + POLICY_SIZE) * last];
        out_weights[..last].copy_from_slice(&weights[trunk..trunk + last]);
        params.extend(out_weights);
        let mut biases = vec![0.0; 1 + POLICY_SIZE];
        biases[0] = weights[trunk + last];
        params.extend(biases);
        self.net = Mlp::new(self.net.shape.clone(), params)?;
        self.adam = Adam::new(self.net.params.len(), self.adam.learning_rate);
        Ok(())
    }

    /// The trunk and the value unit as `evolve.WeightVector` JSON (32 -> hidden... -> 1, tanh on every layer): a
    /// Checkers evaluator, exactly what `selfplay.rs`'s snapshot is.
    pub fn snapshot(&self) -> String {
        let (weights, sizes) = self.value_network();
        Self::weight_vector_json(&weights, &sizes)
    }

    /// The value network (`snapshot`) as numbers.
    pub fn value_network(&self) -> (Vec<f64>, Vec<usize>) {
        let sizes = &self.net.shape.layer_sizes;
        let hidden = &sizes[..sizes.len() - 1];
        let trunk: usize = hidden.windows(2).map(|w| w[1] * (w[0] + 1)).sum();
        let last = *hidden.last().unwrap();
        let mut weights = self.net.params[..trunk].to_vec();
        weights.extend_from_slice(&self.net.params[trunk..trunk + last]); // the value unit's row
        weights.push(self.net.params[trunk + (1 + POLICY_SIZE) * last]); // and its bias
        let mut value_sizes = hidden.to_vec();
        value_sizes.push(1);
        (weights, value_sizes)
    }

    fn weight_vector_json(weights: &[f64], sizes: &[usize]) -> String {
        let weights: Vec<String> = weights.iter().map(|w| format!("{w:?}")).collect();
        let sizes: Vec<String> = sizes.iter().map(|s| s.to_string()).collect();
        format!(
            r#"{{"weights": [{}], "layer_sizes": [{}]}}"#,
            weights.join(", "),
            sizes.join(", ")
        )
    }

    /// The whole two-headed network: `{"weights", "layer_sizes", "activations", "heads"}` (the last layer linear;
    /// output 0 is the value before its tanh, outputs 1.. the move logits by `move_index`).
    pub fn network_json(&self) -> String {
        let base = Self::weight_vector_json(&self.net.params, &self.net.shape.layer_sizes);
        let activations: Vec<String> = self
            .net
            .shape
            .activations
            .iter()
            .map(|a| format!("\"{}\"", a.name()))
            .collect();
        format!(
            r#"{}, "activations": [{}], "heads": "value1+policy{POLICY_SIZE}.v1"}}"#,
            &base[..base.len() - 1],
            activations.join(", ")
        )
    }

    /// One self-play game: its training examples, plies, winner, positions searched and summed search shift.
    fn play(&mut self) -> (Vec<Example>, u32, Option<u8>, f64) {
        let mut game = Checkers::new(self.max_moves_without_capture);
        let mut searched: Vec<(Example, u8)> = Vec::new(); // and its mover, until the outcome is known
        let mut shift = 0.0;
        let mut plies = 0;
        while !game.done && plies < self.max_plies {
            let moves = game.legal_moves();
            let index = if plies < self.random_opening_plies {
                self.explore.below(moves.len() as u32) as usize
            } else if moves.len() == 1 {
                0
            } else {
                let noise = Some((self.dirichlet_alpha, self.dirichlet_weight, &mut self.explore));
                let result = search(&self.net, &game, self.simulations, self.c_puct, noise);
                shift += (result.value - result.network_value).abs();
                let (unique, slots) = unique_indices(&result.moves);
                let mut policy = vec![0.0; unique.len()];
                let total: u32 = result.visits.iter().sum();
                for (&slot, &v) in slots.iter().zip(&result.visits) {
                    policy[slot] += v as f64 / total as f64;
                }
                searched.push((
                    Example {
                        observation: game.observation(),
                        indices: unique,
                        policy,
                        outcome: 0.0,
                    },
                    game.current_player,
                ));
                if plies < self.random_opening_plies + self.temperature_plies {
                    let mut target = self.explore.uniform() * total as f64;
                    let mut pick = result.visits.len() - 1;
                    for (i, &v) in result.visits.iter().enumerate() {
                        if target < v as f64 {
                            pick = i;
                            break;
                        }
                        target -= v as f64;
                    }
                    pick
                } else {
                    most_visited(&result.visits)
                }
            };
            game.step(&moves[index]).expect("a legal move");
            plies += 1;
        }
        let winner = if game.done { game.winner } else { None };
        let examples = searched
            .into_iter()
            .map(|(example, mover)| Example {
                outcome: match winner {
                    Some(w) if w == mover => 1.0,
                    Some(_) => -1.0,
                    None => 0.0,
                },
                ..example
            })
            .collect::<Vec<_>>();
        (examples, plies, winner, shift)
    }

    fn remember(&mut self, examples: Vec<Example>) {
        for example in examples {
            if self.buffer.len() < self.buffer_size {
                self.buffer.push(example);
            } else {
                self.buffer[self.cursor] = example;
                self.cursor = (self.cursor + 1) % self.buffer_size;
            }
        }
    }

    /// One Adam step on a minibatch from the buffer: (value loss, policy loss).
    fn update(&mut self) -> (f64, f64) {
        let batch: Vec<Example> = (0..self.batch_size)
            .map(|_| self.buffer[self.replay.below(self.buffer.len() as u32) as usize].clone())
            .collect();
        let flat: Vec<f64> = batch.iter().flat_map(|e| e.observation.iter().copied()).collect();
        let cache = self.net.forward_batch(&flat, batch.len());
        let (value_loss, policy_loss, grad) = az_gradients(cache.outputs(), &batch, self.value_weight);
        let grads = self.net.backward(&cache, &grad);
        self.adam.step(&mut self.net.params, &grads);
        (value_loss, policy_loss)
    }

    /// Play and learn from `games` games.
    pub fn train(&mut self, games: u64) -> TrainStats {
        let mut stats = TrainStats::default();
        let mut plies = 0u64;
        let mut shift = 0.0;
        for _ in 0..games {
            let (examples, n, winner, game_shift) = self.play();
            plies += n as u64;
            stats.searched += examples.len() as u64;
            shift += game_shift;
            match winner {
                Some(0) => stats.first_wins += 1,
                Some(_) => stats.second_wins += 1,
                None => stats.draws += 1,
            }
            self.remember(examples);
            if self.buffer.len() >= self.batch_size {
                for _ in 0..self.updates_per_game {
                    let (v, p) = self.update();
                    stats.value_loss += v;
                    stats.policy_loss += p;
                    stats.updates += 1;
                }
            }
            self.games += 1;
        }
        stats.games = games;
        stats.mean_plies = plies as f64 / games.max(1) as f64;
        if stats.updates > 0 {
            stats.value_loss /= stats.updates as f64;
            stats.policy_loss /= stats.updates as f64;
        }
        stats.search_shift = shift / stats.searched.max(1) as f64;
        stats
    }

    /// Points per game (win 1, draw 1/2) for the current network playing by search -- `simulations` descents, no
    /// noise, the most visited move -- against the games crate's strategy `opponent` (`material-2`, ...), over
    /// `games` games, seats alternating, each opening with two random plies from `seed` (the network is
    /// deterministic, so without them every game against a deterministic opponent would be the same one).
    pub fn points_against(&self, opponent: &str, simulations: u32, games: u32, seed: u64) -> Result<f64, String> {
        use redqueen_games::checkers_strategies::Strategy;
        let mut points = 0.0;
        for g in 0..games {
            let game_seed = seed + g as u64;
            let mut other = Strategy::parse(opponent, game_seed + 7919, None)?;
            let mut opening = Rng::new(game_seed, Stream::Data);
            let mut game = Checkers::new(self.max_moves_without_capture);
            let seat = (g % 2) as u8;
            let mut plies = 0;
            while !game.done && plies < self.max_plies {
                let moves = game.legal_moves();
                let index = if plies < 2 {
                    opening.below(moves.len() as u32) as usize
                } else if game.current_player != seat {
                    other.pick(&game)
                } else if moves.len() == 1 {
                    0
                } else {
                    most_visited(&search(&self.net, &game, simulations, self.c_puct, None).visits)
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

    /// The value of the starting position for the side to move, and of a position one side is a king up (as
    /// `selfplay.rs`'s probe), plus the policy's entropy at the start (log 7 = 1.95 is uniform over its 7 moves).
    pub fn probe(&self) -> (f64, f64, f64) {
        let game = Checkers::new(self.max_moves_without_capture);
        let start = evaluate(&self.net, &game, &game.legal_moves());
        let mut up = game.observation();
        up[0] = 2.0;
        let king_up = libm::tanh(self.net.forward(&up)[0]);
        let entropy = -start
            .priors
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|p| p * libm::log(*p))
            .sum::<f64>();
        (start.value, king_up, entropy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redqueen_games::checkers::{Board, Piece};

    fn params(pairs: &[(&str, f64)]) -> Params {
        Params::new(pairs.iter().map(|&(k, v)| (k.to_string(), v)))
    }

    fn small() -> Params {
        params(&[("hidden", 16.0), ("hidden_layers", 1.0), ("simulations", 12.0)])
    }

    #[test]
    fn move_indices_are_in_range_and_almost_never_collide() {
        let mut rng = Rng::new(1, Stream::Data);
        let (mut moves_seen, mut collisions) = (0, 0);
        for _ in 0..200 {
            let mut game = Checkers::new(40);
            while !game.done {
                let moves = game.legal_moves();
                let (unique, slots) = unique_indices(&moves);
                assert!(unique.iter().all(|&i| i < POLICY_SIZE));
                assert_eq!(slots.len(), moves.len());
                moves_seen += moves.len();
                collisions += moves.len() - unique.len();
                game.step(&moves[rng.below(moves.len() as u32) as usize]).unwrap();
            }
        }
        assert!(moves_seen > 10_000, "{moves_seen}");
        assert!(collisions * 1000 < moves_seen, "{collisions} of {moves_seen}");
        // the opening's seven moves: seven distinct indices
        let game = Checkers::new(40);
        assert_eq!(unique_indices(&game.legal_moves()).0.len(), 7);
    }

    #[test]
    fn priors_are_a_distribution_over_the_legal_moves() {
        let trainer = AlphaZero::new(0, &small(), 40, 200).unwrap();
        let game = Checkers::new(40);
        let moves = game.legal_moves();
        let e = evaluate(&trainer.net, &game, &moves);
        assert_eq!(e.priors.len(), moves.len());
        assert!((e.priors.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(e.value.abs() < 1.0);
    }

    #[test]
    fn gradients_match_finite_differences() {
        let mut rng = Rng::new(3, Stream::Data);
        let width = 1 + POLICY_SIZE;
        let examples = vec![
            Example {
                observation: vec![],
                indices: vec![4, 17, 99],
                policy: vec![0.5, 0.25, 0.25],
                outcome: 1.0,
            },
            Example {
                observation: vec![],
                indices: vec![0, 127],
                policy: vec![0.0, 1.0],
                outcome: -1.0,
            },
        ];
        let outputs: Vec<f64> = (0..2 * width).map(|_| rng.normal()).collect();
        let (_, _, grad) = az_gradients(&outputs, &examples, 0.7);
        let loss = |o: &[f64]| {
            let (v, p, _) = az_gradients(o, &examples, 0.7);
            v + p
        };
        let h = 1e-6;
        for i in [0, 5, 18, 100, width, width + 1, width + 128, width + 50] {
            let (mut up, mut down) = (outputs.clone(), outputs.clone());
            up[i] += h;
            down[i] -= h;
            let numeric = (loss(&up) - loss(&down)) / (2.0 * h);
            assert!((numeric - grad[i]).abs() < 1e-8, "output {i}: {numeric} vs {}", grad[i]);
        }
        // an illegal move's logit gets no gradient
        assert_eq!(grad[1 + 50], 0.0);
    }

    #[test]
    fn search_counts_every_simulation_and_finds_a_winning_move() {
        let trainer = AlphaZero::new(2, &small(), 40, 200).unwrap();
        let game = Checkers::new(40);
        let result = search(&trainer.net, &game, 30, 1.5, None);
        assert_eq!(result.visits.iter().sum::<u32>(), 30);
        // Black's last piece, a man on (0, 1), has one move: to (1, 0). Red's king stepping there leaves Black without
        // a move -- a win -- and Red's other four moves don't. Untrained priors know nothing; the search does, because
        // a finished game is scored exactly.
        let mut end = Checkers::new(40);
        end.board = Board::from_cells(vec![
            ((2, 1), Piece { owner: 0, king: true }),
            ((7, 4), Piece { owner: 0, king: false }),
            ((0, 1), Piece { owner: 1, king: false }),
        ]);
        let moves = end.legal_moves();
        assert_eq!(moves.len(), 5);
        let result = search(&trainer.net, &end, 100, 1.5, None);
        assert_eq!(
            moves[most_visited(&result.visits)],
            vec![(2, 1), (1, 0)],
            "visits {:?}",
            result.visits
        );
        assert!(result.value > 0.5, "{}", result.value);
    }

    #[test]
    fn noise_is_a_distribution_and_changes_the_root() {
        let mut rng = Rng::new(4, Stream::Explore);
        for alpha in [0.3, 1.0, 2.5] {
            let d = dirichlet(&mut rng, alpha, 7);
            assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-12 && d.iter().all(|&x| x >= 0.0));
        }
        let mean: f64 = (0..4000).map(|_| gamma(&mut rng, 0.5)).sum::<f64>() / 4000.0;
        assert!((mean - 0.5).abs() < 0.05, "Gamma(0.5) has mean 0.5: {mean}");
    }

    #[test]
    fn self_play_learns_and_exports_an_evaluator() {
        let mut trainer = AlphaZero::new(1, &small(), 40, 200).unwrap();
        let untrained = trainer.net.clone();
        let stats = trainer.train(60);
        assert_eq!(stats.first_wins + stats.second_wins + stats.draws, 60);
        assert!(stats.updates > 0 && stats.value_loss.is_finite() && stats.policy_loss.is_finite());
        assert!(trainer.net.params != untrained.params);
        // the policy learned something: its loss is below the uniform policy's on the same buffer
        let flat: Vec<f64> = trainer
            .buffer
            .iter()
            .flat_map(|e| e.observation.iter().copied())
            .collect();
        let before = az_gradients(
            untrained.forward_batch(&flat, trainer.buffer.len()).outputs(),
            &trainer.buffer,
            1.0,
        );
        let after = az_gradients(
            trainer.net.forward_batch(&flat, trainer.buffer.len()).outputs(),
            &trainer.buffer,
            1.0,
        );
        assert!(after.1 < before.1, "policy loss {} -> {}", before.1, after.1);
        assert!(after.0 < before.0, "value loss {} -> {}", before.0, after.0);

        // the exported value network is the trunk + value unit: the games crate's evaluator scores positions with it
        let (weights, sizes) = trainer.value_network();
        assert_eq!(sizes, vec![32, 16, 1]);
        let evaluator = redqueen_games::nets::Network::new(weights.clone(), sizes.clone()).unwrap();
        let game = Checkers::new(40);
        let raw = trainer.net.forward(&game.observation())[0];
        assert!((evaluator.score(&game.observation()) - libm::tanh(raw)).abs() < 1e-12);
        let json = trainer.snapshot();
        assert!(json.ends_with(r#""layer_sizes": [32, 16, 1]}"#));
        assert!(trainer.network_json().ends_with(r#""heads": "value1+policy128.v1"}"#));

        // and a value network loads back as the trunk, with a uniform policy
        let mut other = AlphaZero::new(7, &small(), 40, 200).unwrap();
        other.set_value_network(&weights, &sizes).unwrap();
        assert_eq!(other.snapshot(), json);
        let e = evaluate(&other.net, &game, &game.legal_moves());
        assert!(e.priors.iter().all(|&p| (p - 1.0 / 7.0).abs() < 1e-12));
        assert!(other.set_value_network(&weights, &[32, 8, 1]).is_err());

        let points = trainer.points_against("random", 12, 6, 0).unwrap();
        assert!((0.0..=1.0).contains(&points));
    }

    #[test]
    fn bad_params_are_rejected() {
        assert!(AlphaZero::new(0, &params(&[("simulations", 0.0)]), 40, 200).is_err());
        assert!(AlphaZero::new(0, &params(&[("dirichlet_weight", 1.5)]), 40, 200).is_err());
        assert!(AlphaZero::new(0, &params(&[("c_puct", -1.0)]), 40, 200).is_err());
        assert!(AlphaZero::new(0, &params(&[("lambda", 0.5)]), 40, 200).is_err());
    }
}

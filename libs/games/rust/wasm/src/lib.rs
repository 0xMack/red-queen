//! The game core for the browser (docs/design/0009 Decision 2): the same Rust the training jobs run
//! through PyO3, compiled to WebAssembly for apps/frontend's session worker. Plain numbers and typed
//! arrays across the boundary -- observations go straight from here into ONNX Runtime.
//!
//! Built by `libs/games/build-wasm.py` into `apps/frontend/app/wasm/games/`.

use redqueen_games::baselines::{snake_greedy, SnakeRandom};
use redqueen_games::snake::{decode_relative3, Observer, Snake};
use wasm_bindgen::prelude::*;

/// One Snake game plus the observer its model reads.
#[wasm_bindgen]
pub struct SnakeGame {
    inner: Snake,
    observer: Observer,
}

#[wasm_bindgen]
impl SnakeGame {
    /// `seed` is a u32 (JS numbers carry 53 bits; every seed the site generates fits in 32).
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32, seed: u32, observer_id: &str) -> Result<SnakeGame, JsError> {
        let observer = Observer::from_id(observer_id)
            .ok_or_else(|| JsError::new(&format!("unknown snake observer: {observer_id}")))?;
        if !Snake::fits(width, height) {
            return Err(JsError::new("board too small for the starting snake"));
        }
        Ok(SnakeGame {
            inner: Snake::new(width, height, seed as u64, None),
            observer,
        })
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Relative action (-1 / 0 / +1). Returns the reward; `done` says whether the game ended.
    pub fn step(&mut self, action: i32) -> f64 {
        self.inner.step(action as i64).0
    }

    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        !self.inner.alive
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> u32 {
        self.inner.score
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> i32 {
        self.inner.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> i32 {
        self.inner.height
    }

    /// What the model sees right now, under this game's observer.
    pub fn observation(&self) -> Vec<f64> {
        self.inner.encode(self.observer)
    }

    /// Render cells flattened as [x, y, label, x, y, label, ...] with label 0 body / 1 head / 2 food,
    /// in render order (body behind the head first, then head, then food).
    pub fn cells(&self) -> Vec<i32> {
        use redqueen_games::snake::Label;
        let mut out = Vec::new();
        for (x, y, label) in self.inner.cells() {
            out.extend([
                x,
                y,
                match label {
                    Label::Body => 0,
                    Label::Head => 1,
                    Label::Food => 2,
                },
            ]);
        }
        out
    }
}

/// `relative3.v1`: raw model outputs -> -1 / 0 / +1.
#[wasm_bindgen(js_name = decodeRelative3)]
pub fn decode_relative3_js(outputs: &[f64]) -> i32 {
    if outputs.is_empty() {
        0
    } else {
        decode_relative3(outputs)
    }
}

#[wasm_bindgen]
pub struct RandomPolicy {
    inner: SnakeRandom,
}

#[wasm_bindgen]
impl RandomPolicy {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> RandomPolicy {
        RandomPolicy {
            inner: SnakeRandom::new(seed as u64),
        }
    }

    pub fn decide(&mut self, observation: &[f64]) -> i32 {
        self.inner.decide(observation)
    }
}

/// The greedy baseline; reads snake/features.v1.
#[wasm_bindgen(js_name = greedyDecide)]
pub fn greedy_decide(observation: &[f64]) -> i32 {
    if observation.len() < 11 {
        0
    } else {
        snake_greedy(observation)
    }
}

/// Checkers for the browser. Moves cross as indexes into the current `legalMoves()` list -- the same
/// list, in the same order, the Python side and every strategy see.
#[wasm_bindgen]
pub struct CheckersGame {
    inner: redqueen_games::checkers::Checkers,
    moves: Vec<redqueen_games::checkers::Move>,
}

#[wasm_bindgen]
impl CheckersGame {
    #[wasm_bindgen(constructor)]
    pub fn new(max_moves_without_capture: u32) -> CheckersGame {
        let inner = redqueen_games::checkers::Checkers::new(max_moves_without_capture);
        let moves = inner.legal_moves();
        CheckersGame { inner, moves }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.moves = self.inner.legal_moves();
    }

    /// An independent copy of this game in its current position -- what a lookahead strategy steps
    /// through to see a move's consequences without touching the real game.
    pub fn duplicate(&self) -> CheckersGame {
        CheckersGame {
            inner: self.inner.clone(),
            moves: self.moves.clone(),
        }
    }

    /// Flattened: for each move, its square count n, then n (x, y) pairs.
    #[wasm_bindgen(js_name = legalMoves)]
    pub fn legal_moves(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for mv in &self.moves {
            out.push(mv.len() as i32);
            for &(x, y) in mv {
                out.extend([x, y]);
            }
        }
        out
    }

    /// Play the `index`-th legal move; returns whether the game is over.
    pub fn step(&mut self, index: usize) -> Result<bool, JsError> {
        let mv = self
            .moves
            .get(index)
            .ok_or_else(|| JsError::new("no such legal move"))?
            .clone();
        let done = self.inner.step(&mv).map_err(|e| JsError::new(&e))?;
        self.moves = if done { Vec::new() } else { self.inner.legal_moves() };
        Ok(done)
    }

    pub fn observation(&self) -> Vec<f64> {
        self.inner.observation()
    }

    pub fn simulate(&self, index: usize) -> Result<Vec<f64>, JsError> {
        let mv = self
            .moves
            .get(index)
            .ok_or_else(|| JsError::new("no such legal move"))?;
        self.inner.simulate(mv).map_err(|e| JsError::new(&e))
    }

    /// Flattened [x, y, piece] with piece 0 red man / 1 red king / 2 black man / 3 black king.
    pub fn cells(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for &((x, y), p) in self.inner.board.iter() {
            out.extend([x, y, (p.owner as i32) * 2 + p.king as i32]);
        }
        out
    }

    #[wasm_bindgen(getter, js_name = currentPlayer)]
    pub fn current_player(&self) -> u8 {
        self.inner.current_player
    }

    /// -1 while undecided or for a draw; check `done` to tell them apart.
    #[wasm_bindgen(getter)]
    pub fn winner(&self) -> i32 {
        self.inner.winner.map_or(-1, |w| w as i32)
    }

    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        self.inner.done
    }
}

/// A Checkers strategy for the browser -- the same players training and evaluation use
/// (rust/core/src/checkers_strategies.rs). `pick` returns an index into the game's `legalMoves()`.
#[wasm_bindgen]
pub struct CheckersStrategy {
    inner: redqueen_games::checkers_strategies::Strategy,
}

#[wasm_bindgen]
impl CheckersStrategy {
    /// `name`: random | first-legal | material-N | evaluator. Only the evaluator takes a network: `weights` +
    /// `layerSizes` (an evolve.WeightVector's) or `graph` (a compiled NEAT genome), searched `depth` plies
    /// deep (1 = one ply; omit for the default).
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: &str,
        seed: u32,
        weights: Option<Vec<f64>>,
        layer_sizes: Option<Vec<u32>>,
        depth: Option<u32>,
        graph: Option<Vec<f64>>,
    ) -> Result<CheckersStrategy, JsError> {
        let layers = layer_sizes.map(|l| l.into_iter().map(|n| n as usize).collect());
        let inner = redqueen_games::checkers_strategies::Strategy::build(
            name,
            seed as u64,
            weights,
            layers,
            depth.unwrap_or(1),
            graph,
        )
        .map_err(|e| JsError::new(&e))?;
        Ok(CheckersStrategy { inner })
    }

    /// One score per legal move (higher = better), or undefined for a strategy that doesn't score moves.
    pub fn scores(&self, game: &CheckersGame) -> Option<Vec<f64>> {
        self.inner.scores(&game.inner)
    }

    /// Every layer's values, concatenated (input layer first; split by the network's layer sizes), when the
    /// network evaluated the position legal move `index` leads to -- or undefined for a strategy without one.
    pub fn activations(&self, game: &CheckersGame, index: usize) -> Option<Vec<f64>> {
        self.inner
            .activations(&game.inner, index)
            .map(|layers| layers.into_iter().flatten().collect())
    }

    pub fn pick(&mut self, game: &CheckersGame) -> Result<usize, JsError> {
        if game.moves.is_empty() {
            return Err(JsError::new("no legal moves to pick from"));
        }
        Ok(self.inner.pick(&game.inner))
    }
}

#[wasm_bindgen]
pub struct Reach1DGame {
    inner: redqueen_games::reach1d::Reach1D,
}

#[wasm_bindgen]
impl Reach1DGame {
    #[wasm_bindgen(constructor)]
    pub fn new(
        target: f64,
        start_position: f64,
        start_velocity: f64,
        dt: f64,
        max_acceleration: f64,
        damping: f64,
    ) -> Reach1DGame {
        Reach1DGame {
            inner: redqueen_games::reach1d::Reach1D::new(
                target,
                start_position,
                start_velocity,
                dt,
                max_acceleration,
                damping,
            ),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Returns the reward.
    pub fn step(&mut self, action: f64) -> f64 {
        self.inner.step(action).1
    }

    /// [position - target, velocity]
    pub fn observation(&self) -> Vec<f64> {
        let (a, b) = self.inner.observation();
        vec![a, b]
    }

    #[wasm_bindgen(getter)]
    pub fn position(&self) -> f64 {
        self.inner.position
    }
}

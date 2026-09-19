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
        let observer = Observer::from_id(observer_id).ok_or_else(|| JsError::new(&format!("unknown snake observer: {observer_id}")))?;
        if !Snake::fits(width, height) {
            return Err(JsError::new("board too small for the starting snake"));
        }
        Ok(SnakeGame { inner: Snake::new(width, height, seed as u64, None), observer })
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
            out.extend([x, y, match label { Label::Body => 0, Label::Head => 1, Label::Food => 2 }]);
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
        RandomPolicy { inner: SnakeRandom::new(seed as u64) }
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

//! The RL core for the browser (docs/design/0010): the same Rust the training jobs run through PyO3, compiled to
//! WebAssembly for Learn's live-training demos. Phase 0 exposes what the device check needs -- the determinism
//! digests, the benchmark kernels, and a trainer to measure environment steps per second -- timed from JS
//! (`performance.now()`), since `std::time` doesn't exist here.
//!
//! Built by `libs/rl/build-wasm.py` into `apps/frontend/app/wasm/rl/`.

use redqueen_rl::agent::{Params, Trainer as CoreTrainer, TrainerConfig};
use redqueen_rl::digest;
use redqueen_rl::nn::{Activation, Shape};
use redqueen_rl_envs as envs;
use wasm_bindgen::prelude::*;

fn shape(layer_sizes: &[u32], activations: &str) -> Result<Shape, JsError> {
    let activations = activations
        .split(',')
        .map(|a| Activation::parse(a.trim()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| JsError::new(&e))?;
    Shape::new(layer_sizes.iter().map(|&n| n as usize).collect(), activations).map_err(|e| JsError::new(&e))
}

/// `training_digest(seed, updates)` of `redqueen_rl::digest`: must equal the native build's (determinism.json).
#[wasm_bindgen(js_name = trainingDigest)]
pub fn training_digest(seed: u32, updates: u32) -> String {
    digest::training_digest(seed as u64, updates)
}

/// `rollout_digest(seed)`: a random agent trained and evaluated on Snake and Reach1D, hashed.
#[wasm_bindgen(js_name = rolloutDigest)]
pub fn rollout_digest(seed: u32) -> String {
    envs::rollout_digest(seed as u64)
}

/// `learning_digest(seed)`: Q-learning and SARSA trained on Snake, hashed.
#[wasm_bindgen(js_name = learningDigest)]
pub fn learning_digest(seed: u32) -> String {
    envs::learning_digest(seed as u64)
}

/// `iterations` training updates (forward + backward + Adam) on a batch. `activations`: comma-separated, one per
/// layer after the input (`"relu,relu,linear"`). Returns a checksum; time the call.
#[wasm_bindgen(js_name = benchUpdates)]
pub fn bench_updates(layer_sizes: &[u32], activations: &str, batch: u32, iterations: u32) -> Result<f64, JsError> {
    Ok(digest::bench_updates(
        &shape(layer_sizes, activations)?,
        batch as usize,
        iterations,
        0,
    ))
}

/// `iterations` single-observation forward passes (choosing an action). Returns a checksum; time the call.
#[wasm_bindgen(js_name = benchForwards)]
pub fn bench_forwards(layer_sizes: &[u32], activations: &str, iterations: u32) -> Result<f64, JsError> {
    Ok(digest::bench_forwards(&shape(layer_sizes, activations)?, iterations, 0))
}

/// One agent learning in one environment (`snake/<observer>+relative3.v1` on 10x10, or `reach1d`).
#[wasm_bindgen]
pub struct Trainer {
    inner: CoreTrainer,
}

#[wasm_bindgen]
impl Trainer {
    #[wasm_bindgen(constructor)]
    pub fn new(algorithm: &str, env_id: &str, seed: u32) -> Result<Trainer, JsError> {
        let factory = envs::factory(env_id, 10, 10).map_err(|e| JsError::new(&e))?;
        let config = TrainerConfig {
            seed: seed as u64,
            seed_pool: (100_000, 1_000_000),
            max_episode_steps: 1000,
        };
        let inner = CoreTrainer::build(factory, algorithm, &Params::default(), config).map_err(|e| JsError::new(&e))?;
        Ok(Trainer { inner })
    }

    /// Advance by `steps` environment steps; returns how many episodes ended.
    pub fn train(&mut self, steps: u32) -> u32 {
        self.inner.train(steps as u64).episodes.len() as u32
    }
}

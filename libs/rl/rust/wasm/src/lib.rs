//! The RL core for the browser (docs/design/0010): the same Rust the training jobs run through PyO3, compiled to
//! WebAssembly for Learn's live-training demos: the determinism digests and benchmark kernels (`/dev/rl`, timed
//! from JS with `performance.now()`, since `std::time` doesn't exist here), a `Trainer` a demo drives tick by tick
//! and looks inside (a tabular agent's Q-table and visits), and a `DemoGame` to watch the greedy policy play.
//!
//! Built by `libs/rl/build-wasm.py` into `apps/frontend/app/wasm/rl/`.

use redqueen_games::snake::{Label, Observer, Snake};
use redqueen_rl::agent::{Params, Trainer as CoreTrainer, TrainerConfig};
use redqueen_rl::digest;
use redqueen_rl::env::Action;
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

/// `dqn_digest(seed)`: a DQN with every stability piece trained on Snake's egocentric observer, hashed.
#[wasm_bindgen(js_name = dqnDigest)]
pub fn dqn_digest(seed: u32) -> String {
    envs::dqn_digest(seed as u64)
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

/// One agent learning in one environment (`snake/<observer>+relative3.v1` on 10x10, or `reach1d`) -- the engine of
/// Learn's live demos, the same `Trainer` the training jobs drive through PyO3.
#[wasm_bindgen]
pub struct Trainer {
    inner: CoreTrainer,
}

/// What one call to `Trainer::train` did, as numbers JS can read without a serializer.
#[wasm_bindgen]
pub struct Progress {
    /// Episodes that ended during the call, and their mean return and game score (NaN if none ended).
    pub episodes: u32,
    pub mean_return: f64,
    pub mean_score: f64,
    pub total_steps: f64,
    pub total_episodes: f64,
    pub entropy: f64,
    /// Current epsilon and rows ever updated, for a tabular agent (NaN otherwise).
    pub epsilon: f64,
    pub states_visited: f64,
}

#[wasm_bindgen]
impl Trainer {
    /// `params`: the algorithm's hyperparameters as `name=value` pairs separated by commas (`"alpha=0.1,n_step=3"`,
    /// empty for the defaults). `reward`: `shaped` or `sparse` (Snake).
    #[wasm_bindgen(constructor)]
    pub fn new(algorithm: &str, env_id: &str, seed: u32, params: &str, reward: &str) -> Result<Trainer, JsError> {
        let reward = envs::Reward::parse(reward).map_err(|e| JsError::new(&e))?;
        let factory = envs::factory_with(env_id, 10, 10, reward).map_err(|e| JsError::new(&e))?;
        let config = TrainerConfig {
            seed: seed as u64,
            seed_pool: (100_000, 1_000_000),
            max_episode_steps: 1000,
        };
        let mut pairs = Vec::new();
        for pair in params.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (name, value) = pair
                .split_once('=')
                .ok_or_else(|| JsError::new(&format!("want name=value, got {pair:?}")))?;
            let value: f64 = value
                .trim()
                .parse()
                .map_err(|_| JsError::new(&format!("not a number: {value:?}")))?;
            pairs.push((name.trim().to_string(), value));
        }
        let inner =
            CoreTrainer::build(factory, algorithm, &Params::new(pairs), config).map_err(|e| JsError::new(&e))?;
        Ok(Trainer { inner })
    }

    /// Advance by `steps` environment steps.
    pub fn train(&mut self, steps: u32) -> Progress {
        let stats = self.inner.train(steps as u64);
        let n = stats.episodes.len();
        let mean = |f: fn(&redqueen_rl::env::Episode) -> f64| {
            if n == 0 {
                f64::NAN
            } else {
                stats.episodes.iter().map(f).sum::<f64>() / n as f64
            }
        };
        let extra = |name: &str| {
            stats
                .extras
                .iter()
                .find(|(k, _)| k == name)
                .map_or(f64::NAN, |&(_, v)| v)
        };
        Progress {
            episodes: n as u32,
            mean_return: mean(|e| e.total_reward),
            mean_score: mean(|e| e.score),
            total_steps: stats.total_steps as f64,
            total_episodes: stats.total_episodes as f64,
            entropy: stats.entropy,
            epsilon: extra("epsilon"),
            states_visited: extra("states_visited"),
        }
    }

    /// Environment steps trained so far.
    #[wasm_bindgen(getter, js_name = totalSteps)]
    pub fn total_steps(&self) -> f64 {
        self.inner.total_steps() as f64
    }

    /// The greedy policy's game score on each of `seeds`, games capped at `max_steps`.
    pub fn evaluate(&mut self, seeds: &[u32], max_steps: u32) -> Vec<f64> {
        let seeds: Vec<u64> = seeds.iter().map(|&s| s as u64).collect();
        self.inner.evaluate(&seeds, max_steps).iter().map(|e| e.score).collect()
    }

    /// A tabular agent's values, `row * actions + action` (empty for other agents).
    #[wasm_bindgen(js_name = qValues)]
    pub fn q_values(&self) -> Vec<f64> {
        self.inner
            .agent()
            .table()
            .map(|t| t.values.to_vec())
            .unwrap_or_default()
    }

    /// How many updates each row of the table has had (empty for other agents).
    pub fn visits(&self) -> Vec<u32> {
        self.inner
            .agent()
            .table()
            .map(|t| t.visits.to_vec())
            .unwrap_or_default()
    }

    /// The table row `observation` falls in (-1 for a non-tabular agent).
    pub fn row(&self, observation: &[f64]) -> i32 {
        self.inner
            .agent()
            .table()
            .map_or(-1, |t| t.discretizer.index(observation) as i32)
    }

    /// The greedy action on `observation`, as an index (Snake: 0 left, 1 straight, 2 right).
    #[wasm_bindgen(js_name = greedyAction)]
    pub fn greedy_action(&mut self, observation: &[f64]) -> u32 {
        match self.inner.act_greedy(observation) {
            Action::Discrete(i) => i as u32,
            Action::Continuous(_) => 0,
        }
    }

    /// The current policy as its champion JSON (`modelpack.champions`).
    pub fn snapshot(&self) -> String {
        self.inner.agent().snapshot()
    }
}

/// A Snake game for a demo to play the agent's greedy policy on, move by move, and draw.
#[wasm_bindgen]
pub struct DemoGame {
    game: Snake,
    observer: Observer,
}

#[wasm_bindgen]
impl DemoGame {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, observer_id: &str) -> Result<DemoGame, JsError> {
        let observer =
            Observer::from_id(observer_id).ok_or_else(|| JsError::new(&format!("unknown observer {observer_id}")))?;
        Ok(DemoGame {
            game: Snake::new(10, 10, seed as u64, None),
            observer,
        })
    }

    pub fn observation(&self) -> Vec<f64> {
        self.game.encode(self.observer)
    }

    /// Relative action index (0 left, 1 straight, 2 right); returns the reward.
    pub fn step(&mut self, action: u32) -> f64 {
        self.game.step(action as i64 - 1).0
    }

    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        !self.game.alive
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> u32 {
        self.game.score
    }

    /// Cells as `[x, y, label, ...]`, label 0 body / 1 head / 2 food, in render order (like the games module).
    pub fn cells(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for (x, y, label) in self.game.cells() {
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

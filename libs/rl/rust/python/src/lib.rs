//! `rl._native`: the RL core as a Python extension (docs/design/0010). Training jobs drive a `Trainer` one iteration
//! (thousands of environment steps) per call, so Python is never in the inner loop. Plain types across the boundary.
//! The network, optimizer and DQN-update entry points exist for the `libs/autodiff` oracles (`tests/reference_nn.py`,
//! `tests/reference_dqn.py`) and for benchmarks.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use redqueen_rl::agent::{Params, Trainer as CoreTrainer, TrainerConfig, ALGORITHMS};
use redqueen_rl::digest;
use redqueen_rl::dqn::{td_gradients, Batch, QNet};
use redqueen_rl::env::{ActionSpace, Episode};
use redqueen_rl::nn::{Activation, Adam, Mlp, Shape};
use redqueen_rl::pg::{gae, pg_gradients, PgBatch};
use redqueen_rl::rng::{Rng, Stream};
use redqueen_rl::tabular::{Discretizer, QTableAgent, Step};
use redqueen_rl_envs as envs;

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn episode_dict<'py>(py: Python<'py>, episode: &Episode) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("seed", episode.seed)?;
    d.set_item("total_reward", episode.total_reward)?;
    d.set_item("steps", episode.steps)?;
    d.set_item("score", episode.score)?;
    Ok(d)
}

fn shape(layer_sizes: Vec<usize>, activations: Vec<String>) -> PyResult<Shape> {
    let activations = activations
        .iter()
        .map(|a| Activation::parse(a))
        .collect::<Result<Vec<_>, _>>()
        .map_err(value_error)?;
    Shape::new(layer_sizes, activations).map_err(value_error)
}

/// One agent learning in one environment. `train(steps)` is one iteration of an RL run.
#[pyclass(module = "rl._native", unsendable)]
struct Trainer {
    inner: CoreTrainer,
    algorithm: String,
}

#[pymethods]
impl Trainer {
    /// `env_id`: an interface id (`snake/features.v1+relative3.v1`) or `reach1d`. `params`: the algorithm's
    /// hyperparameters by name (unknown names are an error). Training games come from `seed_pool` = `[lo, hi)`.
    /// `reward`: Snake's `shaped` reward or `sparse` outcomes only (evaluation always reports the game's score).
    #[new]
    #[pyo3(signature = (algorithm, env_id, seed, params=None, width=10, height=10, seed_pool=(100_000, 1_000_000), max_episode_steps=1000, reward="shaped"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        algorithm: &str,
        env_id: &str,
        seed: u64,
        params: Option<HashMap<String, f64>>,
        width: i32,
        height: i32,
        seed_pool: (u64, u64),
        max_episode_steps: u32,
        reward: &str,
    ) -> PyResult<Self> {
        if seed_pool.1 <= seed_pool.0 {
            return Err(value_error(format!("empty seed pool {seed_pool:?}")));
        }
        let reward = envs::Reward::parse(reward).map_err(value_error)?;
        let factory = envs::factory_with(env_id, width, height, reward).map_err(value_error)?;
        let config = TrainerConfig {
            seed,
            seed_pool,
            max_episode_steps,
        };
        let params = Params::new(params.unwrap_or_default());
        let inner = CoreTrainer::build(factory, algorithm, &params, config).map_err(value_error)?;
        Ok(Trainer {
            inner,
            algorithm: algorithm.to_string(),
        })
    }

    /// Advance training by `steps` environment steps: {steps, episodes: [{seed, total_reward, steps, score}], entropy,
    /// extras: {name: value}, total_steps, total_episodes}.
    fn train<'py>(&mut self, py: Python<'py>, steps: u64) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.train(steps);
        let d = PyDict::new(py);
        d.set_item("steps", stats.steps)?;
        let episodes = stats
            .episodes
            .iter()
            .map(|e| episode_dict(py, e))
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("episodes", episodes)?;
        d.set_item("entropy", stats.entropy)?;
        let extras = PyDict::new(py);
        for (name, value) in &stats.extras {
            extras.set_item(name, value)?;
        }
        d.set_item("extras", extras)?;
        d.set_item("total_steps", stats.total_steps)?;
        d.set_item("total_episodes", stats.total_episodes)?;
        Ok(d)
    }

    /// Advance training by `steps` without building per-episode statistics: returns how many episodes ended. What a
    /// benchmark times (the same call the WASM build's `train` makes), since `train`'s dicts cost more than short
    /// episodes do.
    fn advance(&mut self, steps: u64) -> usize {
        self.inner.train(steps).episodes.len()
    }

    /// The greedy policy on games `seeds`, each capped at `max_steps`: one {seed, total_reward, steps, score} each.
    fn evaluate<'py>(&mut self, py: Python<'py>, seeds: Vec<u64>, max_steps: u32) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .evaluate(&seeds, max_steps)
            .iter()
            .map(|e| episode_dict(py, e))
            .collect()
    }

    /// The current policy in its wire format (JSON).
    fn snapshot(&self) -> String {
        self.inner.agent().snapshot()
    }

    #[getter]
    fn algorithm(&self) -> &str {
        &self.algorithm
    }

    #[getter]
    fn env_id(&self) -> String {
        self.inner.env_id()
    }
}

/// Checkers by self-play (`envs::selfplay`, docs/design/0010 Phase 4): TD(λ) on a position-value network, trained
/// `games` games per call. `snapshot()` is the network as `evolve.WeightVector` JSON -- an evaluator the Checkers
/// strategies, the versus leaderboard and the page load as they are.
#[pyclass(module = "rl._native", unsendable)]
struct CheckersSelfPlay {
    inner: envs::selfplay::SelfPlay,
}

#[pymethods]
impl CheckersSelfPlay {
    #[new]
    #[pyo3(signature = (seed, params=None, max_moves_without_capture=40, max_plies=200))]
    fn new(
        seed: u64,
        params: Option<HashMap<String, f64>>,
        max_moves_without_capture: u32,
        max_plies: u32,
    ) -> PyResult<Self> {
        let params = Params::new(params.unwrap_or_default());
        let inner =
            envs::selfplay::SelfPlay::new(seed, &params, max_moves_without_capture, max_plies).map_err(value_error)?;
        Ok(CheckersSelfPlay { inner })
    }

    /// Play and learn from `games` games: {games, first_wins, second_wins, draws, mean_plies, loss, epsilon,
    /// pool_games, total_games}.
    fn train<'py>(&mut self, py: Python<'py>, games: u64) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.train(games);
        let d = PyDict::new(py);
        d.set_item("games", s.games)?;
        d.set_item("first_wins", s.first_wins)?;
        d.set_item("second_wins", s.second_wins)?;
        d.set_item("draws", s.draws)?;
        d.set_item("mean_plies", s.mean_plies)?;
        d.set_item("loss", s.loss)?;
        d.set_item("epsilon", s.epsilon)?;
        d.set_item("pool_games", s.pool_games)?;
        d.set_item("total_games", self.inner.games_played())?;
        Ok(d)
    }

    fn snapshot(&self) -> String {
        self.inner.snapshot()
    }

    /// The network's value of the starting position for the side to move, and of the same position a king up.
    fn probe(&self) -> (f64, f64) {
        self.inner.probe()
    }
}

/// A `games.baselines` policy through the Snake adapter, one {seed, total_reward, steps, score} per game.
#[pyfunction]
#[pyo3(signature = (name, env_id, seeds, max_steps, width=10, height=10))]
fn evaluate_baseline<'py>(
    py: Python<'py>,
    name: &str,
    env_id: &str,
    seeds: Vec<u64>,
    max_steps: u32,
    width: i32,
    height: i32,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let factory = envs::factory(env_id, width, height).map_err(value_error)?;
    envs::evaluate_baseline(name, factory.as_ref(), &seeds, max_steps)
        .map_err(value_error)?
        .iter()
        .map(|e| episode_dict(py, e))
        .collect()
}

/// (observation size, action-space numbers, kind): `[n]` for "discrete", `[low, high]` for "continuous".
#[pyfunction]
#[pyo3(signature = (env_id, width=10, height=10))]
fn env_info(env_id: &str, width: i32, height: i32) -> PyResult<(usize, Vec<f64>, String)> {
    let env = envs::factory(env_id, width, height).map_err(value_error)?.make();
    Ok(match env.action_space() {
        ActionSpace::Discrete(n) => (env.observation_size(), vec![n as f64], "discrete".into()),
        ActionSpace::Continuous { low, high } => (env.observation_size(), vec![low, high], "continuous".into()),
    })
}

#[pyfunction]
fn training_digest(seed: u64, updates: u32) -> String {
    digest::training_digest(seed, updates)
}

#[pyfunction]
fn rollout_digest(seed: u64) -> String {
    envs::rollout_digest(seed)
}

#[pyfunction]
fn learning_digest(seed: u64) -> String {
    envs::learning_digest(seed)
}

#[pyfunction]
fn dqn_digest(seed: u64) -> String {
    envs::dqn_digest(seed)
}

#[pyfunction]
fn pg_digest(seed: u64) -> String {
    envs::pg_digest(seed)
}

#[pyfunction]
fn selfplay_digest(seed: u64) -> String {
    envs::selfplay_digest(seed)
}

/// Initial parameters (He-uniform weights, zero biases) for a network, from the run's `Init` stream.
#[pyfunction]
fn mlp_init(layer_sizes: Vec<usize>, activations: Vec<String>, seed: u64) -> PyResult<Vec<f64>> {
    Ok(Mlp::init(shape(layer_sizes, activations)?, &mut Rng::new(seed, Stream::Init)).params)
}

/// Forward `batch` rows of `inputs`, then backpropagate `output_grad`: (outputs, parameter gradients).
#[pyfunction]
fn mlp_forward_backward(
    layer_sizes: Vec<usize>,
    activations: Vec<String>,
    params: Vec<f64>,
    inputs: Vec<f64>,
    batch: usize,
    output_grad: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let mlp = Mlp::new(shape(layer_sizes, activations)?, params).map_err(value_error)?;
    if inputs.len() != batch * mlp.shape.inputs() || output_grad.len() != batch * mlp.shape.outputs() {
        return Err(value_error(
            "inputs/output_grad don't match batch x the network's shape".into(),
        ));
    }
    let cache = mlp.forward_batch(&inputs, batch);
    let grads = mlp.backward(&cache, &output_grad);
    Ok((cache.outputs().to_vec(), grads))
}

/// Apply Adam for each gradient in `gradients`, in order; returns the final parameters.
#[pyfunction]
#[pyo3(signature = (params, gradients, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8))]
fn adam_steps(
    mut params: Vec<f64>,
    gradients: Vec<Vec<f64>>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
) -> PyResult<Vec<f64>> {
    let mut adam = Adam::with(params.len(), learning_rate, beta1, beta2, epsilon);
    for grads in &gradients {
        if grads.len() != params.len() {
            return Err(value_error("one gradient per parameter".into()));
        }
        adam.step(&mut params, grads);
    }
    Ok(params)
}

/// `iterations` training updates (forward + backward + Adam) on a batch; time it from Python.
#[pyfunction]
fn bench_updates(
    layer_sizes: Vec<usize>,
    activations: Vec<String>,
    batch: usize,
    iterations: u32,
    seed: u64,
) -> PyResult<f64> {
    Ok(digest::bench_updates(
        &shape(layer_sizes, activations)?,
        batch,
        iterations,
        seed,
    ))
}

/// `iterations` single-observation forward passes; time it from Python.
#[pyfunction]
fn bench_forwards(layer_sizes: Vec<usize>, activations: Vec<String>, iterations: u32, seed: u64) -> PyResult<f64> {
    Ok(digest::bench_forwards(
        &shape(layer_sizes, activations)?,
        iterations,
        seed,
    ))
}

/// A recorded transition for `tabular_replay`: (state, action, reward, next_state, done, truncated, next_action).
type ReplayStep = (usize, usize, f64, usize, bool, bool, Option<usize>);

/// Run `steps` through a fresh `q_learning`/`sarsa` table's update rule (no exploration, no environment) and return
/// the table, row-major -- what `tests/reference_tabular.py` recomputes independently.
#[pyfunction]
fn tabular_replay(
    algorithm: &str,
    bits: usize,
    actions: usize,
    params: HashMap<String, f64>,
    steps: Vec<ReplayStep>,
) -> PyResult<Vec<f64>> {
    let sarsa = match algorithm {
        "q_learning" => false,
        "sarsa" => true,
        other => return Err(value_error(format!("not a tabular algorithm: {other:?}"))),
    };
    let mut agent = QTableAgent::new(
        sarsa,
        Discretizer::Binary { bits },
        ActionSpace::Discrete(actions),
        &Params::new(params),
    )
    .map_err(value_error)?;
    let states = 1usize << bits;
    for (state, action, reward, next_state, done, truncated, next_action) in steps {
        if state >= states || next_state >= states || action >= actions || next_action.is_some_and(|a| a >= actions) {
            return Err(value_error(format!(
                "step out of range for {states} states x {actions} actions"
            )));
        }
        agent.learn(Step {
            state,
            action,
            reward,
            next_state,
            done,
            truncated,
            next_action,
        });
    }
    Ok(agent.table().to_vec())
}

/// A Q-network's initial parameters, one vector per sub-network (plain: 1; dueling: trunk, value, advantage).
#[pyfunction]
fn dqn_init(inputs: usize, hidden: Vec<usize>, actions: usize, dueling: bool, seed: u64) -> Vec<Vec<f64>> {
    let net = QNet::init(inputs, &hidden, actions, dueling, &mut Rng::new(seed, Stream::Init));
    net.parts().into_iter().cloned().collect()
}

/// (observations, actions, returns, discounts, next observations, weights), flattened row by row.
type BatchArgs = (Vec<f64>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
/// (loss, gradients per sub-network, td errors, mean Q(s, a)).
type TdOutput = (f64, Vec<Vec<f64>>, Vec<f64>, f64);

/// One DQN update's loss and gradients -- `dqn::td_gradients`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn dqn_td_gradients(
    inputs: usize,
    hidden: Vec<usize>,
    actions: usize,
    dueling: bool,
    double: bool,
    online: Vec<Vec<f64>>,
    target: Vec<Vec<f64>>,
    batch: BatchArgs,
) -> PyResult<TdOutput> {
    let like = QNet::init(inputs, &hidden, actions, dueling, &mut Rng::new(0, Stream::Init));
    let online = QNet::with_params(&like, &online).map_err(value_error)?;
    let target = QNet::with_params(&like, &target).map_err(value_error)?;
    let (observations, taken, returns, discounts, next_observations, weights) = batch;
    let n = taken.len();
    if observations.len() != n * inputs || next_observations.len() != n * inputs || taken.iter().any(|&a| a >= actions)
    {
        return Err(value_error("batch doesn't match the network's shape".into()));
    }
    let result = td_gradients(
        &online,
        &target,
        &Batch {
            observations,
            actions: taken,
            returns,
            discounts,
            next_observations,
            weights,
        },
        double,
    );
    Ok((result.loss, result.grads, result.td, result.q_mean))
}

/// The policy-gradient update's (loss, gradients -- the network's, then a Gaussian's log std --, entropy, kl, clip
/// fraction) -- `pg::pg_gradients`. `log_std` None: a softmax over the outputs; `clip` None: no PPO clipping.
#[pyfunction]
#[pyo3(signature = (layer_sizes, params, log_std, observations, actions, old_log_probs, advantages, clip, entropy_coef))]
#[allow(clippy::too_many_arguments)]
fn policy_gradients(
    layer_sizes: Vec<usize>,
    params: Vec<f64>,
    log_std: Option<f64>,
    observations: Vec<f64>,
    actions: Vec<f64>,
    old_log_probs: Vec<f64>,
    advantages: Vec<f64>,
    clip: Option<f64>,
    entropy_coef: f64,
) -> PyResult<(f64, Vec<f64>, f64, f64, f64)> {
    let hidden = layer_sizes.len().saturating_sub(2);
    let mut activations = vec!["tanh".to_string(); hidden];
    activations.push("linear".into());
    let policy = Mlp::new(shape(layer_sizes, activations)?, params).map_err(value_error)?;
    let n = actions.len();
    if observations.len() != n * policy.shape.inputs() || old_log_probs.len() != n || advantages.len() != n {
        return Err(value_error("batch doesn't match the network's shape".into()));
    }
    let batch = PgBatch {
        observations,
        actions,
        old_log_probs,
        advantages,
    };
    let r = pg_gradients(&policy, log_std, &batch, clip, entropy_coef);
    Ok((r.loss, r.grads, r.entropy, r.kl, r.clip_fraction))
}

/// GAE(λ): (advantages, return targets) -- `pg::gae`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generalized_advantages(
    rewards: Vec<f64>,
    values: Vec<f64>,
    next_values: Vec<f64>,
    done: Vec<bool>,
    cut: Vec<bool>,
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    gae(&rewards, &values, &next_values, &done, &cut, gamma, lambda)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ALGORITHMS", ALGORITHMS.to_vec())?;
    m.add_function(wrap_pyfunction!(tabular_replay, m)?)?;
    m.add_class::<Trainer>()?;
    m.add_class::<CheckersSelfPlay>()?;
    m.add_function(wrap_pyfunction!(evaluate_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(env_info, m)?)?;
    m.add_function(wrap_pyfunction!(training_digest, m)?)?;
    m.add_function(wrap_pyfunction!(rollout_digest, m)?)?;
    m.add_function(wrap_pyfunction!(learning_digest, m)?)?;
    m.add_function(wrap_pyfunction!(dqn_digest, m)?)?;
    m.add_function(wrap_pyfunction!(pg_digest, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_digest, m)?)?;
    m.add_function(wrap_pyfunction!(mlp_init, m)?)?;
    m.add_function(wrap_pyfunction!(mlp_forward_backward, m)?)?;
    m.add_function(wrap_pyfunction!(adam_steps, m)?)?;
    m.add_function(wrap_pyfunction!(dqn_init, m)?)?;
    m.add_function(wrap_pyfunction!(dqn_td_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(policy_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(generalized_advantages, m)?)?;
    m.add_function(wrap_pyfunction!(bench_updates, m)?)?;
    m.add_function(wrap_pyfunction!(bench_forwards, m)?)?;
    Ok(())
}

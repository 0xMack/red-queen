//! `rl._native`: the RL core as a Python extension (docs/design/0010). Training jobs drive a `Trainer` one iteration
//! (thousands of environment steps) per call, so Python is never in the inner loop. Plain types across the boundary.
//! The network and optimizer entry points exist for `tests/reference_nn.py` (the `libs/autodiff` oracle) and for
//! benchmarks.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use redqueen_rl::agent::{Params, Trainer as CoreTrainer, TrainerConfig, ALGORITHMS};
use redqueen_rl::digest;
use redqueen_rl::env::{ActionSpace, Episode};
use redqueen_rl::nn::{Activation, Adam, Mlp, Shape};
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ALGORITHMS", ALGORITHMS.to_vec())?;
    m.add_function(wrap_pyfunction!(tabular_replay, m)?)?;
    m.add_class::<Trainer>()?;
    m.add_function(wrap_pyfunction!(evaluate_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(env_info, m)?)?;
    m.add_function(wrap_pyfunction!(training_digest, m)?)?;
    m.add_function(wrap_pyfunction!(rollout_digest, m)?)?;
    m.add_function(wrap_pyfunction!(learning_digest, m)?)?;
    m.add_function(wrap_pyfunction!(mlp_init, m)?)?;
    m.add_function(wrap_pyfunction!(mlp_forward_backward, m)?)?;
    m.add_function(wrap_pyfunction!(adam_steps, m)?)?;
    m.add_function(wrap_pyfunction!(bench_updates, m)?)?;
    m.add_function(wrap_pyfunction!(bench_forwards, m)?)?;
    Ok(())
}

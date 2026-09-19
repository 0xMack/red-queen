//! `games._native`: the Rust game core as a Python extension (docs/design/0009 Decision 2). Kept to
//! plain types (ints, floats, lists, tuples) -- `games/snake.py` wraps it in the same Python API every
//! caller already uses (`Snake(observer=...)`, `render_state()`, ...).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use redqueen_games::baselines::{snake_greedy, SnakeRandom};
use redqueen_games::pcg::Pcg32 as CorePcg32;
use redqueen_games::snake::{decode_relative3, Observer, Snake};

fn observer(id: &str) -> PyResult<Observer> {
    Observer::from_id(id).ok_or_else(|| PyValueError::new_err(format!("unknown snake observer: {id}")))
}

#[pyclass(module = "games._native")]
struct SnakeCore {
    inner: Snake,
}

#[pymethods]
impl SnakeCore {
    #[new]
    #[pyo3(signature = (width, height, seed, max_steps_without_food=None))]
    fn new(width: i32, height: i32, seed: u64, max_steps_without_food: Option<u32>) -> PyResult<Self> {
        if !Snake::fits(width, height) {
            return Err(PyValueError::new_err(format!("a {width}x{height} board can't fit the starting snake (need width >= 4)")));
        }
        Ok(SnakeCore { inner: Snake::new(width, height, seed, max_steps_without_food) })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// (reward, done)
    fn step(&mut self, action: i64) -> (f64, bool) {
        self.inner.step(action)
    }

    fn encode(&self, observer_id: &str) -> PyResult<Vec<f64>> {
        Ok(self.inner.encode(observer(observer_id)?))
    }

    /// step + encode in one call -- the training hot path.
    fn step_encode(&mut self, action: i64, observer_id: &str) -> PyResult<(Vec<f64>, f64, bool)> {
        let obs = observer(observer_id)?;
        let (reward, done) = self.inner.step(action);
        Ok((self.inner.encode(obs), reward, done))
    }

    fn observation_size(&self, observer_id: &str) -> PyResult<usize> {
        Ok(self.inner.observation_size(observer(observer_id)?))
    }

    /// [(x, y, label)] -- body behind the head first, then head, then food.
    fn cells(&self) -> Vec<(i32, i32, &'static str)> {
        self.inner.cells().into_iter().map(|(x, y, l)| (x, y, l.as_str())).collect()
    }

    fn set_state(&mut self, body: Vec<(i32, i32)>, direction: usize, food: Option<(i32, i32)>) -> PyResult<()> {
        if body.is_empty() {
            return Err(PyValueError::new_err("a snake needs a head"));
        }
        self.inner.set_state(body, direction, food);
        Ok(())
    }

    #[getter]
    fn body(&self) -> Vec<(i32, i32)> {
        self.inner.body.iter().copied().collect()
    }
    #[getter]
    fn food(&self) -> Option<(i32, i32)> {
        self.inner.food
    }
    #[getter]
    fn direction(&self) -> usize {
        self.inner.direction
    }
    #[getter]
    fn score(&self) -> u32 {
        self.inner.score
    }
    #[getter]
    fn alive(&self) -> bool {
        self.inner.alive
    }
    #[getter]
    fn steps_without_food(&self) -> u32 {
        self.inner.steps_without_food
    }
    #[getter]
    fn max_steps_without_food(&self) -> u32 {
        self.inner.max_steps_without_food
    }
}

#[pyclass(module = "games._native")]
struct Pcg32 {
    inner: CorePcg32,
}

#[pymethods]
impl Pcg32 {
    #[new]
    fn new(seed: u64) -> Self {
        Pcg32 { inner: CorePcg32::new(seed) }
    }
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }
    fn bounded(&mut self, n: u32) -> PyResult<u32> {
        if n == 0 {
            return Err(PyValueError::new_err("bounded(0)"));
        }
        Ok(self.inner.bounded(n))
    }
}

#[pyclass(module = "games._native")]
struct SnakeRandomPolicy {
    inner: SnakeRandom,
}

#[pymethods]
impl SnakeRandomPolicy {
    #[new]
    fn new(seed: u64) -> Self {
        SnakeRandomPolicy { inner: SnakeRandom::new(seed) }
    }
    fn __call__(&mut self, observation: Vec<f64>) -> i32 {
        self.inner.decide(&observation)
    }
}

#[pyfunction]
fn snake_greedy_decide(observation: Vec<f64>) -> PyResult<i32> {
    if observation.len() < 11 {
        return Err(PyValueError::new_err("greedy reads snake/features.v1: 11 values"));
    }
    Ok(snake_greedy(&observation))
}

#[pyfunction]
fn relative3_decode(outputs: Vec<f64>) -> PyResult<i32> {
    if outputs.is_empty() {
        return Err(PyValueError::new_err("no outputs to decode"));
    }
    Ok(decode_relative3(&outputs))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnakeCore>()?;
    m.add_class::<Pcg32>()?;
    m.add_class::<SnakeRandomPolicy>()?;
    m.add_function(wrap_pyfunction!(snake_greedy_decide, m)?)?;
    m.add_function(wrap_pyfunction!(relative3_decode, m)?)?;
    Ok(())
}

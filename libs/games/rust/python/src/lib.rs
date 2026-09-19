//! `games._native`: the Rust game core as a Python extension (docs/design/0009 Decision 2). Kept to
//! plain types (ints, floats, lists, tuples) -- `games/snake.py` wraps it in the same Python API every
//! caller already uses (`Snake(observer=...)`, `render_state()`, ...).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use redqueen_games::baselines::{snake_greedy, SnakeRandom};
use redqueen_games::checkers::{Board, Checkers, Piece, Square};
use redqueen_games::pcg::Pcg32 as CorePcg32;
use redqueen_games::reach1d::Reach1D;
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

    /// Games are plain values: copying one (e.g. a lookahead strategy's `copy.deepcopy(env)`) clones
    /// the whole state, PRNG included.
    fn __copy__(&self) -> Self {
        SnakeCore { inner: self.inner.clone() }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        SnakeCore { inner: self.inner.clone() }
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

/// Checkers state: the board crosses as an ordered list of (x, y, owner, king) -- order matters
/// (checkers.rs: legal-move order follows it).
#[pyclass(module = "games._native")]
struct CheckersCore {
    inner: Checkers,
}

fn to_move(mv: Vec<(i32, i32)>) -> Vec<Square> {
    mv
}

#[pymethods]
impl CheckersCore {
    #[new]
    fn new(max_moves_without_capture: u32) -> Self {
        CheckersCore { inner: Checkers::new(max_moves_without_capture) }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Games are plain values: copying one (e.g. a lookahead strategy's `copy.deepcopy(env)`) clones
    /// the whole state, PRNG included.
    fn __copy__(&self) -> Self {
        CheckersCore { inner: self.inner.clone() }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        CheckersCore { inner: self.inner.clone() }
    }

    fn legal_moves(&self) -> Vec<Vec<(i32, i32)>> {
        self.inner.legal_moves()
    }

    /// done (the winner, None for a draw, is `winner`)
    fn step(&mut self, mv: Vec<(i32, i32)>) -> PyResult<bool> {
        self.inner.step(&to_move(mv)).map_err(PyValueError::new_err)
    }

    fn simulate(&self, mv: Vec<(i32, i32)>) -> PyResult<Vec<f64>> {
        self.inner.simulate(&to_move(mv)).map_err(PyValueError::new_err)
    }

    fn observation(&self) -> Vec<f64> {
        self.inner.observation()
    }

    fn cells(&self) -> Vec<(i32, i32, &'static str)> {
        self.inner.cells()
    }

    #[getter]
    fn board(&self) -> Vec<(i32, i32, u8, bool)> {
        self.inner.board.iter().map(|&((x, y), p)| (x, y, p.owner, p.king)).collect()
    }

    #[setter]
    fn set_board(&mut self, cells: Vec<(i32, i32, u8, bool)>) {
        self.inner.board = Board::from_cells(cells.into_iter().map(|(x, y, owner, king)| ((x, y), Piece { owner, king })).collect());
    }

    #[getter]
    fn current_player(&self) -> u8 {
        self.inner.current_player
    }

    #[setter]
    fn set_current_player(&mut self, player: u8) -> PyResult<()> {
        if player > 1 {
            return Err(PyValueError::new_err("player is 0 or 1"));
        }
        self.inner.current_player = player;
        Ok(())
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.winner
    }

    #[getter]
    fn moves_without_capture(&self) -> u32 {
        self.inner.moves_without_capture
    }
}

#[pyclass(module = "games._native")]
struct Reach1DCore {
    inner: Reach1D,
}

#[pymethods]
impl Reach1DCore {
    #[new]
    fn new(target: f64, start_position: f64, start_velocity: f64, dt: f64, max_acceleration: f64, damping: f64) -> Self {
        Reach1DCore { inner: Reach1D::new(target, start_position, start_velocity, dt, max_acceleration, damping) }
    }

    /// Games are plain values: copying one (e.g. a lookahead strategy's `copy.deepcopy(env)`) clones
    /// the whole state, PRNG included.
    fn __copy__(&self) -> Self {
        Reach1DCore { inner: self.inner.clone() }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        Reach1DCore { inner: self.inner.clone() }
    }

    fn reset(&mut self) -> (f64, f64) {
        self.inner.reset()
    }

    /// (observation, reward)
    fn step(&mut self, action: f64) -> ((f64, f64), f64) {
        self.inner.step(action)
    }

    #[getter]
    fn position(&self) -> f64 {
        self.inner.position
    }
    #[setter]
    fn set_position(&mut self, value: f64) {
        self.inner.position = value;
    }
    #[getter]
    fn velocity(&self) -> f64 {
        self.inner.velocity
    }
    #[setter]
    fn set_velocity(&mut self, value: f64) {
        self.inner.velocity = value;
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnakeCore>()?;
    m.add_class::<CheckersCore>()?;
    m.add_class::<Reach1DCore>()?;
    m.add_class::<Pcg32>()?;
    m.add_class::<SnakeRandomPolicy>()?;
    m.add_function(wrap_pyfunction!(snake_greedy_decide, m)?)?;
    m.add_function(wrap_pyfunction!(relative3_decode, m)?)?;
    Ok(())
}

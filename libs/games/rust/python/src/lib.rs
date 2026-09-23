//! `games._native`: the Rust game core as a Python extension (docs/design/0009 Decision 2). Kept to
//! plain types (ints, floats, lists, tuples) -- `games/snake.py` wraps it in the same Python API every
//! caller already uses (`Snake(observer=...)`, `render_state()`, ...).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use redqueen_games::baselines::{snake_greedy, SnakeRandom};
use redqueen_games::checkers::{Board, Checkers, Piece, Square};
use redqueen_games::checkers_strategies::Strategy;
use redqueen_games::nets::{GraphNet, Net, Network};
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
            return Err(PyValueError::new_err(format!(
                "a {width}x{height} board can't fit the starting snake (need width >= 4)"
            )));
        }
        Ok(SnakeCore {
            inner: Snake::new(width, height, seed, max_steps_without_food),
        })
    }

    /// Games are plain values: copying one (e.g. a lookahead strategy's `copy.deepcopy(env)`) clones
    /// the whole state, PRNG included.
    fn __copy__(&self) -> Self {
        SnakeCore {
            inner: self.inner.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        SnakeCore {
            inner: self.inner.clone(),
        }
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
        self.inner
            .cells()
            .into_iter()
            .map(|(x, y, l)| (x, y, l.as_str()))
            .collect()
    }

    fn set_state(&mut self, body: Vec<(i32, i32)>, direction: usize, food: Option<(i32, i32)>) -> PyResult<()> {
        if body.is_empty() {
            return Err(PyValueError::new_err("a snake needs a head"));
        }
        self.inner.set_state(body, direction, food);
        Ok(())
    }

    /// One whole episode from a fresh reset, `policy` choosing every move: (total reward, steps). See
    /// `Snake::play` -- the training hot path, with no Python in the loop.
    fn play(&mut self, policy: PyRef<'_, Policy>, observer_id: &str, max_steps: u32) -> PyResult<(f64, u32)> {
        self.inner
            .play(observer(observer_id)?, &policy.inner, max_steps)
            .map_err(PyValueError::new_err)
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

/// A trained network the core runs itself (rust/core/src/nets.rs), built from `evolve.networks.compiled()`'s
/// numbers: `Policy.layered(weights, layer_sizes)` or `Policy.graph(encoding)`.
#[pyclass(module = "games._native", frozen)]
struct Policy {
    inner: Net,
}

#[pymethods]
impl Policy {
    #[staticmethod]
    fn layered(weights: Vec<f64>, layer_sizes: Vec<usize>) -> PyResult<Self> {
        Ok(Policy {
            inner: Net::Layered(Network::new(weights, layer_sizes).map_err(PyValueError::new_err)?),
        })
    }

    #[staticmethod]
    fn graph(encoding: Vec<f64>) -> PyResult<Self> {
        Ok(Policy {
            inner: Net::Graph(GraphNet::from_flat(&encoding).map_err(PyValueError::new_err)?),
        })
    }

    fn forward(&self, observation: Vec<f64>) -> PyResult<Vec<f64>> {
        if observation.len() != self.inner.inputs() {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.inner.inputs(),
                observation.len()
            )));
        }
        Ok(self.inner.forward(&observation))
    }

    #[getter]
    fn inputs(&self) -> usize {
        self.inner.inputs()
    }

    #[getter]
    fn outputs(&self) -> usize {
        self.inner.outputs()
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
        Pcg32 {
            inner: CorePcg32::new(seed),
        }
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
        SnakeRandomPolicy {
            inner: SnakeRandom::new(seed),
        }
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
        CheckersCore {
            inner: Checkers::new(max_moves_without_capture),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Games are plain values: copying one (e.g. a lookahead strategy's `copy.deepcopy(env)`) clones
    /// the whole state, PRNG included.
    fn __copy__(&self) -> Self {
        CheckersCore {
            inner: self.inner.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        CheckersCore {
            inner: self.inner.clone(),
        }
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
        self.inner
            .board
            .iter()
            .map(|&((x, y), p)| (x, y, p.owner, p.king))
            .collect()
    }

    #[setter]
    fn set_board(&mut self, cells: Vec<(i32, i32, u8, bool)>) {
        self.inner.board = Board::from_cells(
            cells
                .into_iter()
                .map(|(x, y, owner, king)| ((x, y), Piece { owner, king }))
                .collect(),
        );
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

/// A Checkers strategy (rust/core/src/checkers_strategies.rs): owns its PRNG, picks an index into the
/// game's current `legal_moves()`.
#[pyclass(module = "games._native")]
struct CheckersStrategy {
    inner: Strategy,
}

#[pymethods]
impl CheckersStrategy {
    /// Only the "evaluator" strategy takes a network: `weights` + `layer_sizes` (an `evolve.WeightVector`'s) or
    /// `graph` (a compiled NEAT genome, see `GraphNet::from_flat`), searched to `depth` plies (1 = one ply).
    #[new]
    #[pyo3(signature = (name, seed, weights=None, layer_sizes=None, depth=1, graph=None))]
    fn new(
        name: &str,
        seed: u64,
        weights: Option<Vec<f64>>,
        layer_sizes: Option<Vec<usize>>,
        depth: u32,
        graph: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(CheckersStrategy {
            inner: Strategy::build(name, seed, weights, layer_sizes, depth, graph).map_err(PyValueError::new_err)?,
        })
    }

    /// One score per legal move (higher = better), or None for a strategy that doesn't score moves.
    fn scores(&self, game: PyRef<'_, CheckersCore>) -> Option<Vec<f64>> {
        self.inner.scores(&game.inner)
    }

    /// Each layer's values (input layer first) when the network evaluated the position legal move
    /// `index` leads to; None for a strategy without a network.
    fn activations(&self, game: PyRef<'_, CheckersCore>, index: usize) -> Option<Vec<Vec<f64>>> {
        self.inner.activations(&game.inner, index)
    }

    fn pick(&mut self, game: PyRef<'_, CheckersCore>) -> PyResult<usize> {
        if game.inner.legal_moves().is_empty() {
            return Err(PyValueError::new_err("no legal moves to pick from"));
        }
        Ok(self.inner.pick(&game.inner))
    }
}

#[pyclass(module = "games._native")]
struct Reach1DCore {
    inner: Reach1D,
}

#[pymethods]
impl Reach1DCore {
    #[new]
    fn new(
        target: f64,
        start_position: f64,
        start_velocity: f64,
        dt: f64,
        max_acceleration: f64,
        damping: f64,
    ) -> Self {
        Reach1DCore {
            inner: Reach1D::new(target, start_position, start_velocity, dt, max_acceleration, damping),
        }
    }

    /// Games are plain values: copying one (e.g. a lookahead strategy's `copy.deepcopy(env)`) clones
    /// the whole state, PRNG included.
    fn __copy__(&self) -> Self {
        Reach1DCore {
            inner: self.inner.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        Reach1DCore {
            inner: self.inner.clone(),
        }
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
    m.add_class::<CheckersStrategy>()?;
    m.add_class::<Reach1DCore>()?;
    m.add_class::<Pcg32>()?;
    m.add_class::<Policy>()?;
    m.add_class::<SnakeRandomPolicy>()?;
    m.add_function(wrap_pyfunction!(snake_greedy_decide, m)?)?;
    m.add_function(wrap_pyfunction!(relative3_decode, m)?)?;
    Ok(())
}

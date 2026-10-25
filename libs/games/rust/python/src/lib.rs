//! `games._native`: the Rust game core as a Python extension (docs/design/0009 Decision 2). Kept to
//! plain types (ints, floats, lists, tuples) -- `games/snake.py` wraps it in the same Python API every
//! caller already uses (`Snake(observer=...)`, `render_state()`, ...).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use redqueen_games::bandit::{Bandit, Payout, Scenario};
use redqueen_games::baselines::{snake_greedy, SnakeRandom};
use redqueen_games::checkers::{Board, Checkers, Piece, Square};
use redqueen_games::checkers_strategies::Strategy;
use redqueen_games::copycat::{layer_recovery, Document, Geometry};
use redqueen_games::copycat_climb::{hill_climb, ClimbConfig};
use redqueen_games::copycat_trace::{trace, TraceConfig};
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
    /// `quiescence` > 0 extends a search (`material-N`, `evaluator`) up to that many plies through captures.
    #[new]
    #[pyo3(signature = (name, seed, weights=None, layer_sizes=None, depth=1, graph=None, quiescence=0))]
    fn new(
        name: &str,
        seed: u64,
        weights: Option<Vec<f64>>,
        layer_sizes: Option<Vec<usize>>,
        depth: u32,
        graph: Option<Vec<f64>>,
        quiescence: u32,
    ) -> PyResult<Self> {
        let inner = Strategy::build(name, seed, weights, layer_sizes, depth, graph)
            .and_then(|s| s.with_quiescence(quiescence))
            .map_err(PyValueError::new_err)?;
        Ok(CheckersStrategy { inner })
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

fn payout_dict(payout: &Payout) -> Vec<(&'static str, f64)> {
    let mut out = vec![("mean", payout.mean())];
    match *payout {
        Payout::Bernoulli { p } => out.push(("p", p)),
        Payout::Gaussian { mean: _, sd } => out.push(("sd", sd)),
        Payout::Jackpot { p, prize } => out.extend([("p", p), ("prize", prize)]),
        Payout::Fixed { value } => out.push(("value", value)),
    }
    out
}

/// A multi-armed bandit game (docs/design/0011); `games/bandit.py` is its Python face.
#[pyclass(module = "games._native")]
struct BanditCore {
    inner: Bandit,
}

#[pymethods]
impl BanditCore {
    #[new]
    fn new(scenario: &str, seed: u64) -> PyResult<Self> {
        let scenario = Scenario::from_id(scenario)
            .ok_or_else(|| PyValueError::new_err(format!("unknown bandit scenario: {scenario}")))?;
        Ok(BanditCore {
            inner: Bandit::new(scenario, seed),
        })
    }

    fn __copy__(&self) -> Self {
        BanditCore {
            inner: self.inner.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.__copy__()
    }

    fn reset(&mut self) {
        self.inner.reset()
    }

    /// (payout, done)
    fn pull(&mut self, arm: usize) -> PyResult<(f64, bool)> {
        if self.inner.done() {
            return Err(PyValueError::new_err("the game is over"));
        }
        if arm >= self.inner.arms() {
            return Err(PyValueError::new_err(format!("no arm {arm}")));
        }
        Ok(self.inner.pull(arm))
    }

    fn means(&self) -> Vec<f64> {
        self.inner.means()
    }

    /// Each arm's payout under lamp `context`, before any drift: kind and parameters.
    fn payouts(&self, context: usize) -> Vec<(&'static str, Vec<(&'static str, f64)>)> {
        self.inner
            .payouts(context)
            .iter()
            .map(|p| (p.kind(), payout_dict(p)))
            .collect()
    }

    fn drift(&self) -> Option<(u32, Vec<f64>)> {
        self.inner
            .drift()
            .map(|(at, after)| (at, after.iter().map(Payout::mean).collect()))
    }

    #[getter]
    fn arms(&self) -> usize {
        self.inner.arms()
    }
    #[getter]
    fn budget(&self) -> u32 {
        self.inner.budget
    }
    #[getter]
    fn pulls(&self) -> u32 {
        self.inner.pulls
    }
    #[getter]
    fn lamp(&self) -> usize {
        self.inner.lamp
    }
    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }
    #[getter]
    fn regret(&self) -> f64 {
        self.inner.regret()
    }
    #[getter]
    fn efficiency(&self) -> f64 {
        self.inner.efficiency()
    }
    #[getter]
    fn skill(&self) -> f64 {
        self.inner.skill()
    }
    #[getter]
    fn counts(&self) -> Vec<u32> {
        self.inner.counts.clone()
    }
    #[getter]
    fn best_pulls(&self) -> u32 {
        self.inner.best_pulls
    }
    #[getter]
    fn best_arm(&self) -> usize {
        self.inner.best_arm()
    }
    #[getter]
    fn detour(&self) -> Option<usize> {
        self.inner.detour()
    }
}

/// A Copy Cat SVG, parsed into the subset (docs/design/0022); `games/copycat.py` is its Python face. Images cross as
/// bytes: float64 little-endian or 8-bit RGBA, row-major.
#[pyclass(module = "games._native", frozen)]
struct CopyCatDocument {
    inner: Document,
}

fn f64_bytes<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, PyBytes> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    PyBytes::new(py, &bytes)
}

#[pymethods]
impl CopyCatDocument {
    /// Raises ValueError, with the reason, for anything outside the subset.
    #[staticmethod]
    fn parse(text: &str) -> PyResult<Self> {
        Document::parse(text)
            .map(|inner| CopyCatDocument { inner })
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }
    #[getter]
    fn shapes(&self) -> usize {
        self.inner.shapes.len()
    }
    #[getter]
    fn nodes(&self) -> usize {
        self.inner.nodes()
    }
    #[getter]
    fn palette(&self) -> usize {
        self.inner.palette()
    }

    fn canonical(&self) -> String {
        self.inner.canonical()
    }

    /// Total absolute turning angle along every flattened outline, radians.
    #[getter]
    fn turning(&self) -> f64 {
        self.inner.turning()
    }

    /// (kind, fill, opacity) per shape, in paint order.
    fn describe(&self) -> Vec<(&'static str, (u8, u8, u8), f64)> {
        self.inner
            .shapes
            .iter()
            .map(|s| {
                let kind = match s.geometry {
                    Geometry::Rect { .. } => "rect",
                    Geometry::Circle { .. } => "circle",
                    Geometry::Ellipse { .. } => "ellipse",
                    Geometry::Polygon(_) => "polygon",
                    Geometry::Path(_) => "path",
                };
                (kind, (s.fill[0], s.fill[1], s.fill[2]), s.opacity)
            })
            .collect()
    }

    /// Premultiplied RGBA in 0..1, float64.
    fn render<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        f64_bytes(py, &self.inner.render().data)
    }

    /// Straight-alpha RGBA, 8 bits (a PNG's pixels).
    fn render_rgba8<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.render().to_rgba8())
    }

    /// Shape `index` alone: its coverage over the whole image, float64.
    fn coverage<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyBytes>> {
        let shape = self
            .inner
            .shapes
            .get(index)
            .ok_or_else(|| PyValueError::new_err(format!("no shape {index}")))?;
        let (w, h) = (self.inner.width as usize, self.inner.height as usize);
        let mask = shape.mask(self.inner.width, self.inner.height);
        let full: Vec<f64> = (0..w * h).map(|i| mask.at(i % w, i / w)).collect();
        Ok(f64_bytes(py, &full))
    }

    /// Every shape's flattened outline: closed polylines (what smoothness is measured on).
    fn outlines(&self) -> Vec<Vec<Vec<(f64, f64)>>> {
        self.inner.shapes.iter().map(|s| s.outline()).collect()
    }
}

/// For each shape of `clean`, its best whole-shape IoU with a shape of `output`.
#[pyfunction]
fn copycat_layer_recovery(clean: &CopyCatDocument, output: &CopyCatDocument) -> Vec<f64> {
    layer_recovery(&clean.inner, &output.inner)
}

/// The `primitive`-style hill climb on an 8-bit RGBA image.
#[pyfunction]
#[pyo3(signature = (rgba, width, height, shapes, candidates, patience, seed))]
#[allow(clippy::too_many_arguments)]
fn copycat_hill_climb(
    py: Python<'_>,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    shapes: usize,
    candidates: usize,
    patience: usize,
    seed: u64,
) -> PyResult<CopyCatDocument> {
    if rgba.len() != (width * height * 4) as usize {
        return Err(PyValueError::new_err(format!("rgba is not {width}x{height}")));
    }
    let config = ClimbConfig {
        shapes,
        candidates,
        patience,
        seed,
    };
    let inner = py.detach(|| hill_climb(&rgba, width, height, &config));
    Ok(CopyCatDocument { inner })
}

/// The structured pipeline (docs/design/0022 step 2) on an opaque 8-bit RGBA image.
#[pyfunction]
#[pyo3(signature = (rgba, width, height, colors, merge_delta, background_delta, pixel_bits, edge_bits, min_area, outline_gain, seed))]
#[allow(clippy::too_many_arguments)]
fn copycat_trace(
    py: Python<'_>,
    rgba: Vec<u8>,
    width: usize,
    height: usize,
    colors: usize,
    merge_delta: f64,
    background_delta: f64,
    pixel_bits: f64,
    edge_bits: f64,
    min_area: usize,
    outline_gain: f64,
    seed: u64,
) -> PyResult<CopyCatDocument> {
    let config = TraceConfig {
        colors,
        merge_delta,
        background_delta,
        pixel_bits,
        edge_bits,
        min_area,
        outline_gain,
        seed,
    };
    let inner = py
        .detach(|| trace(&rgba, width, height, &config))
        .map_err(PyValueError::new_err)?;
    Ok(CopyCatDocument { inner })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnakeCore>()?;
    m.add_class::<CheckersCore>()?;
    m.add_class::<CheckersStrategy>()?;
    m.add_class::<Reach1DCore>()?;
    m.add_class::<BanditCore>()?;
    m.add_class::<CopyCatDocument>()?;
    m.add_class::<Pcg32>()?;
    m.add_class::<Policy>()?;
    m.add_class::<SnakeRandomPolicy>()?;
    m.add_function(wrap_pyfunction!(snake_greedy_decide, m)?)?;
    m.add_function(wrap_pyfunction!(relative3_decode, m)?)?;
    m.add_function(wrap_pyfunction!(copycat_layer_recovery, m)?)?;
    m.add_function(wrap_pyfunction!(copycat_hill_climb, m)?)?;
    m.add_function(wrap_pyfunction!(copycat_trace, m)?)?;
    Ok(())
}

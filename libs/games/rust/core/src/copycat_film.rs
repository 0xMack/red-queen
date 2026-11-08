//! A film of an entrant at work (docs/design/0023): what its document looked like after each change it kept, so the game
//! page can play it back. A frame is a patch to the shape list, so a film of thousands stays small.
//!
//! Recording changes nothing: it draws no random numbers and affects no comparison, and a film ends on the exact document
//! the entrant returns without one (`Film::matches_output`, held to it by tests here and in `copycat_climb` and
//! `copycat_trace`). Frames are scored by one evaluator (`Film::fidelity_at`, `arena.copycat`'s clean fidelity), whatever
//! the entrant's own objective is.
//!
//! The fitter's descent is many calls deep and also runs for fits that are only compared, so it reports here through
//! a thread-local log that is off unless a recording pipeline turns it on: `attempt`, `step` and the rest cost one check
//! when off. After a part is fitted, `Log::chain` finds which attempt became its final shape.

use std::cell::RefCell;

use crate::copycat::{write_shape, Document, Geometry, Shape};
use crate::copycat_fit::Fit;
use crate::raster::Canvas;

/// Clean fidelity's ceiling, as `arena.copycat.PSNR_CAP`.
const PSNR_CAP: f64 = 60.0;
/// A shape's way to its final form is thinned to this many frames: its steps are small, and most look the same.
pub const MAX_STEPS: usize = 24;

/// What a frame shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum View {
    /// The image it was given.
    Input,
    /// The pipeline's labels after merging.
    Clustered,
    /// The pipeline's labels after cleaning.
    Labels,
    /// The document so far.
    Vector,
}

/// How much a frame matters, which sets how long it holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// A stage of the process.
    Stage,
    /// A candidate shown and not necessarily kept: off the path the chart follows.
    Try,
    /// A change it kept.
    Step,
    /// A shape's final form.
    Lock,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Patch {
    None,
    /// Sets the shape at this index, or appends it at the end of the list.
    Set(usize, Shape),
    Remove(usize),
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub stage: String,
    pub note: String,
    pub kind: Kind,
    pub view: View,
    pub patch: Patch,
    /// The entrant's own objective at this frame (`Film::objective`), NaN if it has none yet.
    pub objective: f64,
}

/// What `Film::progress` holds.
pub type Listener = Box<dyn FnMut(&Film, f64)>;

pub struct Film {
    pub width: u32,
    pub height: u32,
    /// What `Frame::objective` is, for a label.
    pub objective: &'static str,
    pub frames: Vec<Frame>,
    /// What the entrant returned.
    pub document: Document,
    /// The pipeline's label images and palette, for `View::Clustered` and `View::Labels`; empty for other entrants.
    pub clustered: Vec<usize>,
    pub labels: Vec<usize>,
    pub palette: Vec<[u8; 3]>,
    /// Told how far the run is (0 to 1) as it goes, with the film so far: for a page showing it at work.
    pub progress: Option<Listener>,
}

impl Film {
    pub fn new(width: u32, height: u32, objective: &'static str) -> Self {
        Film {
            width,
            height,
            objective,
            frames: Vec::new(),
            document: Document {
                width,
                height,
                groups: Vec::new(),
                shapes: Vec::new(),
            },
            clustered: Vec::new(),
            labels: Vec::new(),
            palette: Vec::new(),
            progress: None,
        }
    }

    /// Tells `progress`, if anyone is listening, that the run is `done` of the way through.
    pub fn report(&mut self, done: f64) {
        if let Some(mut tell) = self.progress.take() {
            tell(self, done);
            self.progress = Some(tell);
        }
    }

    /// The elements in place after the last frame, in paint order.
    pub fn elements_now(&self) -> String {
        let mut out = String::new();
        if let Some(last) = self.frames.len().checked_sub(1) {
            for shape in self.shapes_at(last) {
                write_shape(&mut out, &shape);
            }
        }
        out
    }

    /// The shapes in place at frame `i`.
    pub fn shapes_at(&self, i: usize) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();
        for frame in &self.frames[..=i] {
            match &frame.patch {
                Patch::None => {}
                Patch::Set(k, shape) if *k < shapes.len() => shapes[*k] = shape.clone(),
                Patch::Set(_, shape) => shapes.push(shape.clone()),
                Patch::Remove(k) => {
                    shapes.remove(*k);
                }
            }
        }
        shapes
    }

    /// Whether the last frame is what the entrant returned: its shapes, in order, whatever groups hold them.
    pub fn matches_output(&self) -> bool {
        let Some(last) = self.frames.len().checked_sub(1) else {
            return self.document.shapes.is_empty();
        };
        let shapes = self.shapes_at(last);
        shapes.len() == self.document.shapes.len()
            && shapes
                .iter()
                .zip(&self.document.shapes)
                .all(|(a, b)| a.geometry == b.geometry && a.fill == b.fill && a.opacity == b.opacity)
    }

    /// The element frame `i` sets, in the canonical form.
    pub fn element(&self, i: usize) -> Option<String> {
        match &self.frames.get(i)?.patch {
            Patch::Set(_, shape) => {
                let mut out = String::new();
                write_shape(&mut out, shape);
                Some(out)
            }
            _ => None,
        }
    }

    /// Clean fidelity of frame `i` against the clean document. Scored as its canonical form, as the leaderboard scores
    /// an SVG: coordinates at 0.1 px, not the fitter's exact ones.
    pub fn fidelity_at(&self, i: usize, clean: &Canvas) -> f64 {
        let document = Document {
            width: self.width,
            height: self.height,
            groups: Vec::new(),
            shapes: self.shapes_at(i),
        };
        let written = Document::parse(&document.canonical()).expect("a canonical document parses");
        fidelity(&written.render(), clean)
    }
}

/// PSNR over black and over white, averaged: `arena.copycat.score`'s rank.
pub fn fidelity(ours: &Canvas, theirs: &Canvas) -> f64 {
    assert_eq!(ours.data.len(), theirs.data.len(), "two canvases of different sizes");
    let (mut black, mut white) = (0.0, 0.0);
    for (a, b) in ours.data.as_chunks::<4>().0.iter().zip(theirs.data.as_chunks::<4>().0) {
        for k in 0..3 {
            let on_black = a[k] - b[k];
            let on_white = on_black - (a[3] - b[3]);
            black += on_black * on_black;
            white += on_white * on_white;
        }
    }
    let values = (ours.data.len() / 4 * 3) as f64;
    (psnr(black / values) + psnr(white / values)) / 2.0
}

/// PSNR of a mean squared error, as `arena.copycat._psnr`.
pub fn psnr(mse: f64) -> f64 {
    if mse <= libm::pow(10.0, -PSNR_CAP / 10.0) {
        PSNR_CAP
    } else {
        (-10.0 * libm::log10(mse)).min(PSNR_CAP)
    }
}

/// About `max` of `items`, evenly spaced, always with the last.
pub fn thinned<T>(items: Vec<T>, max: usize) -> Vec<T> {
    let n = items.len();
    if n <= max {
        return items;
    }
    items
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i == n - 1 || i * max / n != (i + 1) * max / n)
        .map(|(_, item)| item)
        .collect()
}

/// One candidate's way through the fitter's descent.
pub struct Attempt {
    pub kind: &'static str,
    /// Each state it kept and its description's length, in bits.
    pub states: Vec<(Geometry, f64)>,
    /// Cutting holes starts from another attempt's result, not from a fresh candidate.
    from_earlier: bool,
}

#[derive(Default)]
pub struct Log {
    pub attempts: Vec<Attempt>,
    /// How many attempts the part's first whole fit made: what is shown as tried.
    pub first_fit: Option<usize>,
    paused: u32,
}

impl Log {
    /// The states that led to `geometry`, oldest first.
    pub fn chain(&self, geometry: &Geometry) -> Vec<(Geometry, f64)> {
        self.chain_before(geometry, self.attempts.len())
    }

    fn chain_before(&self, geometry: &Geometry, end: usize) -> Vec<(Geometry, f64)> {
        let found = self.attempts[..end]
            .iter()
            .rposition(|a| a.states.last().is_some_and(|(g, _)| g == geometry));
        let Some(i) = found else { return Vec::new() };
        let attempt = &self.attempts[i];
        if !attempt.from_earlier {
            return attempt.states.clone();
        }
        let mut chain = self.chain_before(&attempt.states[0].0, i);
        chain.extend(attempt.states.iter().skip(1).cloned());
        chain
    }
}

thread_local! {
    static LOG: RefCell<Option<Log>> = const { RefCell::new(None) };
}

fn with_log(f: impl FnOnce(&mut Log)) {
    LOG.with_borrow_mut(|log| {
        if let Some(log) = log {
            f(log);
        }
    });
}

fn kind_of(fit: &Fit) -> &'static str {
    match (&fit.knots, &fit.rounded, &fit.geometry) {
        (Some(_), _, _) if fit.tight => "tight spline",
        (Some(_), _, _) => "spline",
        (_, Some(_), _) => "rounded rect",
        (_, _, Geometry::Rect { .. }) => "rect",
        (_, _, Geometry::Circle { .. }) => "circle",
        (_, _, Geometry::Ellipse { .. }) => "ellipse",
        (_, _, Geometry::Polygon(_)) => "polygon",
        (_, _, Geometry::Path(_)) => "path",
    }
}

/// Starts logging, for a recording pipeline.
pub(crate) fn start() {
    LOG.set(Some(Log::default()));
}

/// Stops logging.
pub(crate) fn stop() {
    LOG.set(None);
}

/// What was logged since the last call.
pub(crate) fn take() -> Log {
    LOG.with_borrow_mut(|log| log.as_mut().map(std::mem::take).unwrap_or_default())
}

/// A candidate starts its descent.
pub(crate) fn attempt(fit: &Fit) {
    with_log(|log| {
        if log.paused == 0 {
            log.attempts.push(Attempt {
                kind: kind_of(fit),
                states: vec![(fit.geometry.clone(), fit.bits())],
                from_earlier: false,
            });
        }
    });
}

/// Holes start being cut from a part's fit.
pub(crate) fn cutting(fit: &Fit) {
    with_log(|log| {
        if log.paused == 0 {
            log.attempts.push(Attempt {
                kind: "holes",
                states: vec![(fit.geometry.clone(), fit.bits())],
                from_earlier: true,
            });
        }
    });
}

/// The current candidate kept a change.
pub(crate) fn step(fit: &Fit) {
    with_log(|log| {
        if log.paused == 0 {
            if let Some(attempt) = log.attempts.last_mut() {
                attempt.states.push((fit.geometry.clone(), fit.bits()));
            }
        }
    });
}

/// The current candidate's length, for `rewind`.
pub(crate) fn mark() -> usize {
    let mut at = 0;
    with_log(|log| at = log.attempts.last().map_or(0, |a| a.states.len()));
    at
}

/// Drops the states since `mark`: a branch that wasn't kept.
pub(crate) fn rewind(mark: usize) {
    with_log(|log| {
        if log.paused == 0 {
            if let Some(attempt) = log.attempts.last_mut() {
                attempt.states.truncate(mark);
            }
        }
    });
}

/// A whole fit finished its candidates: the first one for a part is what gets shown as tried.
pub(crate) fn fitted() {
    with_log(|log| {
        if log.paused == 0 && log.first_fit.is_none() {
            log.first_fit = Some(log.attempts.len());
        }
    });
}

/// Fits inside a fit (a hole's own shape) aren't logged.
pub(crate) fn pause() {
    with_log(|log| log.paused += 1);
}

pub(crate) fn resume() {
    with_log(|log| log.paused -= 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinning_keeps_about_max_and_always_the_last() {
        assert_eq!(thinned(vec![1, 2, 3], 5), vec![1, 2, 3]);
        let kept = thinned((0..1000).collect(), 24);
        assert!(kept.len() <= 25 && kept.len() >= 20, "{}", kept.len());
        assert_eq!(kept.last(), Some(&999));
    }

    #[test]
    fn identical_canvases_hit_the_cap_and_a_known_error_is_exact() {
        let mut a = Canvas::new(2, 2);
        assert_eq!(fidelity(&a, &a.clone()), PSNR_CAP);
        // an opaque white image against a transparent one: over black every channel is off by 1, over white by none
        for px in a.data.as_chunks_mut::<4>().0 {
            *px = [1.0, 1.0, 1.0, 1.0];
        }
        assert_eq!(fidelity(&a, &Canvas::new(2, 2)), (0.0 + PSNR_CAP) / 2.0);
        assert!((psnr(0.01) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn replay_applies_sets_appends_and_removes() {
        let shape = |x: f64| Shape {
            geometry: Geometry::Circle { cx: x, cy: 1.0, r: 1.0 },
            fill: [1, 2, 3],
            opacity: 1.0,
            group: None,
        };
        let frame = |patch: Patch| Frame {
            stage: String::new(),
            note: String::new(),
            kind: Kind::Step,
            view: View::Vector,
            patch,
            objective: f64::NAN,
        };
        let mut film = Film::new(8, 8, "bits");
        film.frames = vec![
            frame(Patch::Set(0, shape(1.0))),
            frame(Patch::Set(1, shape(2.0))),
            frame(Patch::Set(0, shape(3.0))),
            frame(Patch::None),
            frame(Patch::Remove(0)),
        ];
        assert_eq!(film.shapes_at(2), vec![shape(3.0), shape(2.0)]);
        assert_eq!(film.shapes_at(4), vec![shape(2.0)]);
        film.document.shapes = vec![shape(2.0)];
        assert!(film.matches_output());
        film.document.shapes = vec![shape(9.0)];
        assert!(!film.matches_output());
        assert!(film.element(0).unwrap().starts_with("<circle"));
        assert_eq!(film.element(3), None);
    }

    #[test]
    fn the_log_is_off_unless_started_and_a_chain_follows_holes_back() {
        let fit = |r: f64| Fit {
            geometry: Geometry::Circle { cx: 1.0, cy: 1.0, r },
            model_bits: r,
            misfit_bits: 0.0,
            knots: None,
            tight: false,
            rounded: None,
        };
        attempt(&fit(1.0));
        assert!(take().attempts.is_empty());

        start();
        attempt(&fit(1.0));
        step(&fit(2.0));
        let at = mark();
        step(&fit(9.0));
        rewind(at);
        attempt(&fit(5.0));
        fitted();
        pause();
        attempt(&fit(7.0));
        resume();
        cutting(&fit(2.0));
        step(&fit(3.0));
        let log = take();
        stop();
        assert_eq!(log.first_fit, Some(2));
        assert_eq!(log.attempts.len(), 3);
        let radii = |chain: Vec<(Geometry, f64)>| chain.into_iter().map(|(_, bits)| bits).collect::<Vec<_>>();
        assert_eq!(radii(log.chain(&fit(3.0).geometry)), vec![1.0, 2.0, 3.0]);
        assert_eq!(radii(log.chain(&fit(5.0).geometry)), vec![5.0]);
    }
}

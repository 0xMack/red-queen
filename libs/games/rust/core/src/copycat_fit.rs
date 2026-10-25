//! Fitting one region with the shortest description (docs/design/0022 Decision 4, the ladder's step 2.3): every
//! candidate shape -- circle, ellipse, rect, polygon, Bézier path -- is scored by its code length, and the shortest
//! wins:
//!
//! ```text
//! bits = model bits (a shape type, then each coordinate at 0.1 px) + misfit bits (pixel_bits per wrongly covered pixel)
//! ```
//!
//! So a near-circle snaps to a circle (3 numbers) rather than the Béziers that trace it, and a straight run is one
//! segment. **Free pixels** -- those covered by a shape above this one -- cost nothing either way, which is how a hidden
//! part completes itself (Decision 5): the cheapest circle behind a square is the whole circle, not a crescent.
//!
//! Contours come from marching squares on the region's indicator, are cut at corners, and are fitted with cubic Béziers by Schneider's algorithm ("An Algorithm for
//! Automatically Fitting Digitized Curves", Graphics Gems, 1990). Only `+ - * /` and `sqrt` (and `log2` for the
//! coordinate code) -- the same numbers natively and in WASM.

use std::collections::HashMap;

use crate::copycat::{Geometry, Segment, Shape};
use crate::raster::{coverage, Point};

/// What a region must cover (`visible`: how much of each pixel is the region's, 0..1 -- an edge pixel's fraction comes
/// from its colour, so edges are sub-pixel), what it may cover for free (`free`: hidden under shapes above it), over a
/// `width x height` image, row-major. Everything else must stay uncovered.
/// How many outline candidates (polygons and paths) are refined point by point: the shortest as first fitted.
const REFINED_OUTLINES: usize = 2;

pub struct Target<'a> {
    pub width: usize,
    pub height: usize,
    pub visible: &'a [f64],
    pub free: &'a [bool],
}

#[derive(Clone, Copy, Debug)]
pub struct Costs {
    /// Bits per coordinate (0.1 px over the image, plus a margin).
    pub coord_bits: f64,
    /// Bits to say which kind of shape, or which kind of path segment.
    pub type_bits: f64,
    /// Bits per pixel the shape covers wrongly (or fails to cover).
    pub pixel_bits: f64,
    /// Bits a move of an outline's point must save to be kept: above 0, an outline stops chasing the noise.
    pub outline_gain: f64,
}

impl Costs {
    pub fn new(width: usize, height: usize, pixel_bits: f64) -> Self {
        Costs {
            coord_bits: libm::log2(10.0 * width.max(height) as f64) + 1.0,
            type_bits: 3.0,
            pixel_bits,
            outline_gain: 0.0,
        }
    }

    pub fn model_bits(&self, geometry: &Geometry) -> f64 {
        let coords = match geometry {
            Geometry::Circle { .. } => 3,
            Geometry::Rect { .. } | Geometry::Ellipse { .. } => 4,
            Geometry::Polygon(points) => 2 * points.len() + 1, // + its count, about a coordinate's worth
            Geometry::Path(segments) => {
                let mut n = 0;
                for segment in segments {
                    n += match segment {
                        Segment::Move(_) | Segment::Line(_) => 2,
                        Segment::Quad(..) => 4,
                        Segment::Cubic(..) => 6,
                        Segment::Close => 0,
                    };
                }
                return self.type_bits * (1 + segments.len()) as f64 + self.coord_bits * n as f64;
            }
        };
        self.type_bits + self.coord_bits * coords as f64
    }

    /// A smooth closed spline through `knots` points: a shape type, a "smooth" flag, the count, and each knot. Its
    /// tangents follow from the neighbouring knots, so they cost nothing.
    pub fn spline_bits(&self, knots: usize) -> f64 {
        2.0 * self.type_bits + self.coord_bits * (2 * knots + 1) as f64
    }
}

#[derive(Clone, Debug)]
pub struct Fit {
    pub geometry: Geometry,
    pub model_bits: f64,
    pub misfit_bits: f64,
    /// For a smooth closed spline, its knots: `geometry` is the cubics they imply, and the description is the knots.
    pub knots: Option<Vec<Point>>,
}

impl Fit {
    pub fn bits(&self) -> f64 {
        self.model_bits + self.misfit_bits
    }
}

impl Target<'_> {
    fn at(&self, x: i64, y: i64) -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height)
            .then(|| y as usize * self.width + x as usize)
    }

    /// Wrongly covered pixels: the squared difference between the shape's coverage and the target, free pixels
    /// skipped (pixels outside the image never count).
    pub fn misfit(&self, geometry: &Geometry) -> f64 {
        self.misfit_from(geometry, self.uncovered())
    }

    /// The misfit if nothing were covered: every visible pixel that isn't free, missed.
    fn uncovered(&self) -> f64 {
        self.visible
            .iter()
            .zip(self.free)
            .filter(|(_, f)| !**f)
            .map(|(v, _)| v * v)
            .sum()
    }

    /// `misfit`, given `uncovered()` (the same for every candidate, so computed once per fit).
    fn misfit_from(&self, geometry: &Geometry, uncovered: f64) -> f64 {
        let shape = Shape {
            geometry: geometry.clone(),
            fill: [0, 0, 0],
            opacity: 1.0,
            group: None,
        };
        let mask = coverage(&shape.outline(), self.width, self.height);
        // everything uncovered, then the difference the shape's own bounding box makes
        let mut wrong = uncovered;
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let i = (mask.y0 + my) * self.width + mask.x0 + mx;
                if self.free[i] {
                    continue;
                }
                let (c, v) = (mask.data[my * mask.width + mx], self.visible[i]);
                wrong += (v - c) * (v - c) - v * v;
            }
        }
        wrong.max(0.0)
    }

    fn inside(&self, i: usize) -> bool {
        self.visible[i] >= 0.5
    }

    fn score(&self, geometry: Geometry, costs: &Costs, uncovered: f64) -> Fit {
        Fit {
            misfit_bits: self.misfit_from(&geometry, uncovered) * costs.pixel_bits,
            model_bits: costs.model_bits(&geometry),
            geometry,
            knots: None,
        }
    }

    /// A smooth closed spline through `knots`: drawn as the cubics they imply, described by the knots alone.
    fn score_spline(&self, knots: Vec<Point>, costs: &Costs, uncovered: f64) -> Fit {
        let geometry = spline(&knots);
        Fit {
            misfit_bits: self.misfit_from(&geometry, uncovered) * costs.pixel_bits,
            model_bits: costs.spline_bits(knots.len()),
            geometry,
            knots: Some(knots),
        }
    }

    /// Points on the visible region's *true* edges: where the fraction crosses 0.5 between an inside pixel and an
    /// uncovered one (not free, inside the image), interpolated between their centres. An edge against a shape above,
    /// or the image border, says nothing about the part's outline.
    fn true_edge(&self) -> Vec<Point> {
        let mut out = Vec::new();
        for y in 0..self.height as i64 {
            for x in 0..self.width as i64 {
                let i = y as usize * self.width + x as usize;
                if !self.inside(i) {
                    continue;
                }
                for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                    if let Some(j) = self.at(x + dx, y + dy) {
                        if !self.inside(j) && !self.free[j] {
                            let (vi, vj) = (self.visible[i], self.visible[j]);
                            let t = if vi > vj {
                                ((vi - 0.5) / (vi - vj)).clamp(0.0, 1.0)
                            } else {
                                0.5
                            };
                            out.push((x as f64 + 0.5 + t * dx as f64, y as f64 + 0.5 + t * dy as f64));
                        }
                    }
                }
            }
        }
        out
    }

    /// The visible fractions, with the free pixels inside the inside pixels' convex hull made whole: the region
    /// completed under whatever covers it, as simply as a mask can say.
    pub fn completed(&self) -> Vec<f64> {
        let mut corners = Vec::new();
        for i in 0..self.visible.len() {
            if self.inside(i) {
                let (x, y) = ((i % self.width) as f64, (i / self.width) as f64);
                corners.extend([(x, y), (x + 1.0, y), (x, y + 1.0), (x + 1.0, y + 1.0)]);
            }
        }
        let hull = convex_hull(corners);
        let mut out = self.visible.to_vec();
        if hull.len() < 3 {
            return out;
        }
        let mask = coverage(&vec![hull], self.width, self.height);
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let i = (mask.y0 + my) * self.width + mask.x0 + mx;
                if self.free[i] && mask.data[my * mask.width + mx] >= 0.5 {
                    out[i] = 1.0;
                }
            }
        }
        out
    }

    /// The shortest description of this region among every candidate. None if the region is empty.
    pub fn fit(&self, costs: &Costs) -> Option<Fit> {
        let mut candidates: Vec<Geometry> = Vec::new();
        let mut knot_sets: Vec<Vec<Point>> = Vec::new();
        let mut fitted_knots: Vec<Vec<Point>> = Vec::new();
        let edge = self.true_edge();
        if edge.len() >= 6 {
            candidates.extend(fit_circle(&edge));
            candidates.extend(fit_ellipse(&edge));
            candidates.extend(edge_rect(&edge));
        }
        let completed = self.completed();
        if let Some(loop_) = largest_contour(&completed, self.width, self.height) {
            candidates.extend(mask_rect(&completed, self.width));
            for tolerance in [0.7, 1.5, 3.0] {
                let polygon = simplify_closed(&loop_, tolerance);
                if polygon.len() >= 3 {
                    candidates.push(Geometry::Polygon(polygon));
                }
            }
            for tolerance in [0.4, 0.8, 1.6] {
                if let Some(path) = bezier_path(&loop_, tolerance) {
                    candidates.push(path);
                }
            }
            knot_sets = SPLINE_KNOTS.iter().filter_map(|&k| knots_along(&loop_, k)).collect();
            // fitted to the true edge where there is enough of it: a hidden part's outline along its occluders and its
            // completed hull is not its own
            let to: &[Point] = if edge.len() >= 12 { &edge } else { &loop_ };
            fitted_knots = FITTED_KNOTS.filter_map(|k| fit_knots(&loop_, to, k)).collect();
        }
        // Every primitive is refined (a few numbers each); of the outlines, only the REFINED_OUTLINES shortest as
        // fitted, since refining one costs two scores per point per step.
        let uncovered = self.uncovered();
        let (mut outlines, primitives): (Vec<Fit>, Vec<Fit>) = candidates
            .into_iter()
            .map(|candidate| self.score(candidate, costs, uncovered))
            .partition(|fit| matches!(fit.geometry, Geometry::Polygon(_) | Geometry::Path(_)));
        outlines.sort_by(|a, b| a.bits().total_cmp(&b.bits()));
        outlines.truncate(REFINED_OUTLINES);
        // and the shortest spline as fitted, of each start: a spline's evenly spaced knots are rough, so it rarely leads
        // before refinement, and the least-squares knots, though closer, can refine to a worse one
        let splines = knot_sets
            .into_iter()
            .map(|knots| self.score_spline(knots, costs, uncovered))
            .min_by(|a, b| a.bits().total_cmp(&b.bits()));
        let fitted = fitted_knots
            .into_iter()
            .map(|knots| self.score_spline(knots, costs, uncovered))
            .min_by(|a, b| a.bits().total_cmp(&b.bits()));
        let others = primitives
            .into_iter()
            .chain(outlines)
            .chain(splines)
            .map(|fit| self.refine(fit, costs, uncovered))
            .min_by(|a, b| a.bits().total_cmp(&b.bits()));
        // a least-squares spline follows a noisy edge's noise too, so on a noisy image it must win by a margin
        let fitted = fitted.map(|fit| self.refine(fit, costs, uncovered));
        let margin = FITTED_GAIN * costs.outline_gain;
        match (others, fitted) {
            (Some(o), Some(f)) => Some(if f.bits() + margin < o.bits() { f } else { o }),
            (o, f) => o.or(f),
        }
    }

    /// Coordinate descent on a shape's numbers -- a primitive's parameters, a polygon's vertices, a path's points, a
    /// spline's knots -- then, for a polygon or a spline, deleting the point that costs least to lose, kept unless the
    /// descent from there is longer by `outline_gain` or more.
    fn refine(&self, fit: Fit, costs: &Costs, uncovered: f64) -> Fit {
        let mut fit = self.descend(fit, costs, uncovered);
        loop {
            let (points, is_spline) = match (&fit.knots, &fit.geometry) {
                (Some(knots), _) => (knots, true),
                (None, Geometry::Polygon(points)) => (points, false),
                _ => break,
            };
            if points.len() <= 3 {
                break;
            }
            let Some(fewer) = (0..points.len())
                .map(|i| {
                    let mut rest = points.clone();
                    rest.remove(i);
                    if is_spline {
                        self.score_spline(rest, costs, uncovered)
                    } else {
                        self.score(Geometry::Polygon(rest), costs, uncovered)
                    }
                })
                .min_by(|a, b| a.bits().total_cmp(&b.bits()))
            else {
                break;
            };
            let fewer = self.descend(fewer, costs, uncovered);
            // a point stays only if it saves what a move of it must: on a noisy image, one that pays for itself by
            // less is following the noise
            if fewer.bits() >= fit.bits() + costs.outline_gain {
                break;
            }
            fit = fewer;
        }
        fit
    }

    /// Each number nudged by 0.5, 0.25 and 0.1 px while that shortens the description (by `outline_gain` bits, for a
    /// polygon's or path's points).
    fn descend(&self, mut fit: Fit, costs: &Costs, uncovered: f64) -> Fit {
        let mut p = match &fit.knots {
            Some(knots) => knots.iter().flat_map(|&(x, y)| [x, y]).collect(),
            None => params(&fit.geometry),
        };
        let outline = matches!(fit.geometry, Geometry::Polygon(_) | Geometry::Path(_));
        let min_gain = if outline { costs.outline_gain.max(1e-9) } else { 1e-9 };
        for step in [0.5, 0.25, 0.1] {
            for _round in 0..4 {
                let mut improved = false;
                for k in 0..p.len() {
                    for sign in [1.0, -1.0] {
                        let mut q = p.clone();
                        q[k] += sign * step;
                        let candidate = match &fit.knots {
                            Some(knots) => self.score_spline(
                                (0..knots.len()).map(|i| (q[2 * i], q[2 * i + 1])).collect(),
                                costs,
                                uncovered,
                            ),
                            None => self.score(with_params(&fit.geometry, &q), costs, uncovered),
                        };
                        if candidate.bits() < fit.bits() - min_gain {
                            fit = candidate;
                            p = q;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        fit
    }
}

/// How many knots a spline candidate starts with, spaced evenly along the contour.
const SPLINE_KNOTS: [usize; 4] = [4, 5, 6, 8];

/// The weights a point of a closed spline's segment at `t` puts on the segment's four knots (the one before, its two
/// ends, the one after): the Bernstein weights of `spline`'s cubic, its control points written in the knots.
fn spline_weights(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    let b = [s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t];
    [
        -b[1] / 6.0,
        b[0] + b[1] + b[2] / 6.0,
        b[1] / 6.0 + b[2] + b[3],
        -b[2] / 6.0,
    ]
}

/// The `k` knots whose spline is closest to the points `to` (in any order), by least squares: each point is given the
/// nearest point of the current spline (a segment and a `t`), which makes it a fixed mix of four knots, and its
/// distance from there is measured mostly across the curve (tangent-distance minimization), so the knots solve a
/// linear system; repeated from knots spaced evenly along `loop_`. None if the loop is too short or the system
/// singular.
fn fit_knots(loop_: &[Point], to: &[Point], k: usize) -> Option<Vec<Point>> {
    // where on the curve each point starts its search; a ternary search then places it between samples
    const SAMPLES: usize = 12;
    let mut knots = knots_along(loop_, k)?;
    let step = to.len().div_ceil(240).max(1);
    let points: Vec<Point> = to.iter().step_by(step).copied().collect();
    let position = |knots: &[Point], i: usize, t: f64| {
        let w = spline_weights(t);
        (0..4).fold((0.0, 0.0), |acc, j| add(acc, scale(knots[(i + k + j - 1) % k], w[j])))
    };
    for _ in 0..SPLINE_ROUNDS {
        let curve: Vec<(usize, f64, Point)> = (0..k)
            .flat_map(|i| (0..SAMPLES).map(move |s| (i, s as f64 / SAMPLES as f64)))
            .map(|(i, t)| (i, t, position(&knots, i, t)))
            .collect();
        // unknowns: x0, y0, x1, y1, ...
        let mut ata = vec![vec![0.0; 2 * k]; 2 * k];
        let mut atb = vec![0.0; 2 * k];
        for &p in &points {
            let &(i, t, _) = curve
                .iter()
                .min_by(|a, b| length(sub(a.2, p)).total_cmp(&length(sub(b.2, p))))?;
            // then to a fraction of a sample, by ternary search between its neighbours
            let (mut lo, mut hi) = ((t - 1.0 / SAMPLES as f64).max(0.0), (t + 1.0 / SAMPLES as f64).min(1.0));
            for _ in 0..12 {
                let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
                if length(sub(position(&knots, i, a), p)) < length(sub(position(&knots, i, b), p)) {
                    hi = b;
                } else {
                    lo = a;
                }
            }
            let t = (lo + hi) / 2.0;
            let tangent = unit(sub(
                position(&knots, i, (t + 1e-3).min(1.0)),
                position(&knots, i, (t - 1e-3).max(0.0)),
            ));
            let normal = (-tangent.1, tangent.0);
            let w = spline_weights(t);
            for (u, weight) in [(normal, 1.0), (tangent, TANGENT_WEIGHT)] {
                let row: Vec<(usize, f64)> = (0..4)
                    .flat_map(|j| {
                        let kj = (i + k + j - 1) % k;
                        [(2 * kj, w[j] * u.0), (2 * kj + 1, w[j] * u.1)]
                    })
                    .collect();
                let target = dot(u, p);
                for &(a, va) in &row {
                    atb[a] += weight * va * target;
                    for &(b, vb) in &row {
                        ata[a][b] += weight * va * vb;
                    }
                }
            }
        }
        let x = solve_dense(ata, atb)?;
        knots = (0..k).map(|j| (x[2 * j], x[2 * j + 1])).collect();
    }
    Some(knots)
}

/// Gaussian elimination with partial pivoting. None if singular.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let pivot = (c..n).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))?;
        if a[pivot][c].abs() < 1e-9 {
            return None;
        }
        a.swap(c, pivot);
        b.swap(c, pivot);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            let (above, below) = a.split_at_mut(r);
            for (x, &y) in below[0][c..].iter_mut().zip(&above[c][c..]) {
                *x -= f * y;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let v: f64 = b[r] - (r + 1..n).map(|j| a[r][j] * x[j]).sum::<f64>();
        x[r] = v / a[r][r];
    }
    Some(x)
}

/// How a point's distance along the spline's tangent counts against its distance across it, in `fit_knots`: a point
/// sliding along the curve is mostly the assignment's error, not the knots', and counting it fully (plain projection)
/// is what makes the knots creep there slowly. Some weight keeps the system from going singular.
const TANGENT_WEIGHT: f64 = 0.05;

/// How many times `fit_knots` re-assigns the points and solves again.
const SPLINE_ROUNDS: usize = 6;
/// A least-squares spline must be shorter than every other candidate by this many outline moves' worth of bits
/// (`Costs::outline_gain`, zero on a clean image).
const FITTED_GAIN: f64 = 3.0;
/// The knot counts a least-squares spline candidate is fitted with.
const FITTED_KNOTS: std::ops::RangeInclusive<usize> = 4..=8;

/// `k` points spaced evenly along a closed loop (by length). None if the loop is too short to hold them.
fn knots_along(loop_: &[Point], k: usize) -> Option<Vec<Point>> {
    if loop_.len() < 2 * k {
        return None;
    }
    let n = loop_.len();
    let mut cumulative = Vec::with_capacity(n + 1);
    cumulative.push(0.0);
    for i in 0..n {
        let last = cumulative[i];
        cumulative.push(last + length(sub(loop_[(i + 1) % n], loop_[i])));
    }
    let total = cumulative[n];
    let mut out = Vec::with_capacity(k);
    let mut i = 0;
    for j in 0..k {
        let at = total * j as f64 / k as f64;
        while cumulative[i + 1] < at {
            i += 1;
        }
        let span = cumulative[i + 1] - cumulative[i];
        let t = if span > 0.0 { (at - cumulative[i]) / span } else { 0.0 };
        out.push(add(loop_[i], scale(sub(loop_[(i + 1) % n], loop_[i]), t)));
    }
    Some(out)
}

/// The closed Catmull-Rom spline through `knots`, as cubics: each knot's tangent is a sixth of the chord between its
/// neighbours, so the curve is smooth everywhere and says nothing the knots don't.
pub fn spline(knots: &[Point]) -> Geometry {
    let k = knots.len();
    let mut segments = vec![Segment::Move(knots[0])];
    for i in 0..k {
        let (p0, p1, p2, p3) = (knots[(i + k - 1) % k], knots[i], knots[(i + 1) % k], knots[(i + 2) % k]);
        segments.push(Segment::Cubic(
            add(p1, scale(sub(p2, p0), 1.0 / 6.0)),
            sub(p2, scale(sub(p3, p1), 1.0 / 6.0)),
            p2,
        ));
    }
    segments.push(Segment::Close);
    Geometry::Path(segments)
}

/// A shape's numbers, in a fixed order: a primitive's parameters, or every point of a polygon or path.
fn params(geometry: &Geometry) -> Vec<f64> {
    match geometry {
        Geometry::Circle { cx, cy, r } => vec![*cx, *cy, *r],
        Geometry::Ellipse { cx, cy, rx, ry } => vec![*cx, *cy, *rx, *ry],
        Geometry::Rect { x, y, width, height } => vec![*x, *y, *width, *height],
        Geometry::Polygon(points) => points.iter().flat_map(|&(x, y)| [x, y]).collect(),
        Geometry::Path(segments) => {
            let mut out = Vec::new();
            for segment in segments {
                match *segment {
                    Segment::Move(a) | Segment::Line(a) => out.extend([a.0, a.1]),
                    Segment::Quad(a, b) => out.extend([a.0, a.1, b.0, b.1]),
                    Segment::Cubic(a, b, c) => out.extend([a.0, a.1, b.0, b.1, c.0, c.1]),
                    Segment::Close => {}
                }
            }
            out
        }
    }
}

/// `geometry` with its numbers replaced by `p` (same order as `params`); sizes stay at least half a pixel.
fn with_params(geometry: &Geometry, p: &[f64]) -> Geometry {
    match geometry {
        Geometry::Circle { .. } => Geometry::Circle {
            cx: p[0],
            cy: p[1],
            r: p[2].max(0.5),
        },
        Geometry::Ellipse { .. } => Geometry::Ellipse {
            cx: p[0],
            cy: p[1],
            rx: p[2].max(0.5),
            ry: p[3].max(0.5),
        },
        Geometry::Rect { .. } => Geometry::Rect {
            x: p[0],
            y: p[1],
            width: p[2].max(0.5),
            height: p[3].max(0.5),
        },
        Geometry::Polygon(points) => Geometry::Polygon((0..points.len()).map(|i| (p[2 * i], p[2 * i + 1])).collect()),
        Geometry::Path(segments) => {
            let mut k = 0;
            let mut next = || {
                k += 2;
                (p[k - 2], p[k - 1])
            };
            Geometry::Path(
                segments
                    .iter()
                    .map(|segment| match segment {
                        Segment::Move(_) => Segment::Move(next()),
                        Segment::Line(_) => Segment::Line(next()),
                        Segment::Quad(..) => Segment::Quad(next(), next()),
                        Segment::Cubic(..) => Segment::Cubic(next(), next(), next()),
                        Segment::Close => Segment::Close,
                    })
                    .collect(),
            )
        }
    }
}

// --- primitives ---------------------------------------------------------------------------------------------------

/// Solves a small dense system by Gaussian elimination with partial pivoting; None if it is singular.
pub fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col];
        for row in col + 1..N {
            let f = a[row][col] / pivot_row[col];
            for (v, p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *v -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let mut s = b[row];
        for k in row + 1..N {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

fn centroid(points: &[Point]) -> Point {
    let n = points.len() as f64;
    (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    )
}

/// The algebraic (Kåsa) circle through edge points: x^2 + y^2 + Dx + Ey + F = 0 by least squares.
fn fit_circle(points: &[Point]) -> Option<Geometry> {
    let (mx, my) = centroid(points);
    let (mut a, mut b) = ([[0.0; 3]; 3], [0.0; 3]);
    for &(x, y) in points {
        let (x, y) = (x - mx, y - my);
        let row = [x, y, 1.0];
        let z = -(x * x + y * y);
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += row[i] * row[j];
            }
            b[i] += row[i] * z;
        }
    }
    let [d, e, f] = solve(a, b)?;
    let (cx, cy) = (-d / 2.0, -e / 2.0);
    let r2 = cx * cx + cy * cy - f;
    (r2 > 0.25).then(|| Geometry::Circle {
        cx: cx + mx,
        cy: cy + my,
        r: r2.sqrt(),
    })
}

/// The axis-aligned ellipse through edge points: Ax^2 + Cy^2 + Dx + Ey = 1 by least squares (about their centroid).
fn fit_ellipse(points: &[Point]) -> Option<Geometry> {
    let (mx, my) = centroid(points);
    let (mut a, mut b) = ([[0.0; 4]; 4], [0.0; 4]);
    for &(x, y) in points {
        let (x, y) = (x - mx, y - my);
        let row = [x * x, y * y, x, y];
        for i in 0..4 {
            for j in 0..4 {
                a[i][j] += row[i] * row[j];
            }
            b[i] += row[i];
        }
    }
    let [ca, cc, d, e] = solve(a, b)?;
    if ca <= 0.0 || cc <= 0.0 {
        return None;
    }
    let g = 1.0 + d * d / (4.0 * ca) + e * e / (4.0 * cc);
    (g > 0.0).then(|| Geometry::Ellipse {
        cx: mx - d / (2.0 * ca),
        cy: my - e / (2.0 * cc),
        rx: (g / ca).sqrt(),
        ry: (g / cc).sqrt(),
    })
}

fn edge_rect(points: &[Point]) -> Option<Geometry> {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in points {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    (x1 > x0 && y1 > y0).then_some(Geometry::Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

fn mask_rect(mask: &[f64], width: usize) -> Option<Geometry> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m >= 0.5) {
        let (x, y) = (i % width, i / width);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x + 1);
        y1 = y1.max(y + 1);
    }
    (x1 > x0).then_some(Geometry::Rect {
        x: x0 as f64,
        y: y0 as f64,
        width: (x1 - x0) as f64,
        height: (y1 - y0) as f64,
    })
}

/// Andrew's monotone chain, counter-clockwise; collinear points dropped.
pub fn convex_hull(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let cross = |o: Point, a: Point, b: Point| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut hull: Vec<Point> = Vec::with_capacity(points.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &Point>> = if pass == 0 {
            Box::new(points.iter())
        } else {
            Box::new(points.iter().rev())
        };
        for &p in iter {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

// --- contours -----------------------------------------------------------------------------------------------------

/// The largest closed contour (by enclosed area) of `mask` at 0.5, traced by marching squares between pixel centres:
/// a soft mask's edges land between pixels where the fractions say, a binary one's on the pixel boundary (a diagonal
/// as a half-pixel staircase, which the fits smooth); points in image coordinates.
pub fn largest_contour(mask: &[f64], width: usize, height: usize) -> Option<Vec<Point>> {
    // the grid is the image padded by one empty pixel on every side
    let (gw, gh) = (width + 2, height + 2);
    let value = |x: i64, y: i64| -> f64 {
        if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
            0.0
        } else {
            mask[y as usize * width + x as usize]
        }
    };
    let mut grid = vec![0.0; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            let (x, y) = (gx as i64 - 1, gy as i64 - 1);
            grid[gy * gw + gx] = value(x, y);
        }
    }
    let g = |x: usize, y: usize| grid[y * gw + x];
    let inside = |v: f64| v > 0.5;
    // edge ids: the horizontal edge from (x, y) to (x+1, y) is 2 (y gw + x); the vertical one down from it, + 1
    let h_edge = |x: usize, y: usize| 2 * (y * gw + x);
    let v_edge = |x: usize, y: usize| 2 * (y * gw + x) + 1;
    let mut points: HashMap<usize, Point> = HashMap::new();
    let mut point = |id: usize, (ax, ay): (usize, usize), (bx, by): (usize, usize)| {
        points.entry(id).or_insert_with(|| {
            let (va, vb) = (g(ax, ay), g(bx, by));
            let t = (0.5 - va) / (vb - va);
            // grid (gx, gy) is pixel (gx - 1, gy - 1), whose centre is at gx - 0.5
            (
                ax as f64 + t * (bx as f64 - ax as f64) - 0.5,
                ay as f64 + t * (by as f64 - ay as f64) - 0.5,
            )
        });
        id
    };
    let mut segments: Vec<(usize, usize)> = Vec::new();
    for y in 0..gh - 1 {
        for x in 0..gw - 1 {
            let (v00, v10, v11, v01) = (g(x, y), g(x + 1, y), g(x + 1, y + 1), g(x, y + 1));
            let case =
                inside(v00) as u8 | (inside(v10) as u8) << 1 | (inside(v11) as u8) << 2 | (inside(v01) as u8) << 3;
            if case == 0 || case == 15 {
                continue;
            }
            let top = point(h_edge(x, y), (x, y), (x + 1, y));
            let bottom = point(h_edge(x, y + 1), (x, y + 1), (x + 1, y + 1));
            let left = point(v_edge(x, y), (x, y), (x, y + 1));
            let right = point(v_edge(x + 1, y), (x + 1, y), (x + 1, y + 1));
            let centre = inside((v00 + v10 + v11 + v01) / 4.0);
            match case {
                1 | 14 => segments.push((left, top)),
                2 | 13 => segments.push((top, right)),
                3 | 12 => segments.push((left, right)),
                4 | 11 => segments.push((right, bottom)),
                6 | 9 => segments.push((top, bottom)),
                7 | 8 => segments.push((left, bottom)),
                5 => {
                    if centre {
                        segments.extend([(top, right), (bottom, left)]);
                    } else {
                        segments.extend([(left, top), (right, bottom)]);
                    }
                }
                _ => {
                    // 10
                    if centre {
                        segments.extend([(left, top), (right, bottom)]);
                    } else {
                        segments.extend([(top, right), (bottom, left)]);
                    }
                }
            }
        }
    }
    // link segments through their shared edge points into closed loops
    let mut by_point: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &(a, b)) in segments.iter().enumerate() {
        by_point.entry(a).or_default().push(i);
        by_point.entry(b).or_default().push(i);
    }
    let mut used = vec![false; segments.len()];
    let mut best: Option<(f64, Vec<Point>)> = None;
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let (first, mut at) = segments[start];
        let mut ids = vec![first];
        while at != first {
            ids.push(at);
            let next = by_point[&at].iter().copied().find(|&s| !used[s]);
            let Some(next) = next else { break };
            used[next] = true;
            let (a, b) = segments[next];
            at = if a == at { b } else { a };
        }
        if ids.len() < 3 {
            continue;
        }
        let loop_: Vec<Point> = ids.iter().map(|id| points[id]).collect();
        let area = signed_area(&loop_).abs();
        if best.as_ref().is_none_or(|b| area > b.0) {
            best = Some((area, loop_));
        }
    }
    best.map(|b| b.1)
}

/// A closed loop's notches, deepest first: where it is furthest inside its convex hull, one point (the deepest) per run
/// of points more than `depth` inside. Two overlapping convex shapes' union has a notch at each crossing of their
/// outlines.
pub fn notches(loop_: &[Point], depth: f64) -> Vec<usize> {
    let hull = convex_hull(loop_.to_vec());
    if hull.len() < 3 {
        return Vec::new();
    }
    let inset: Vec<f64> = loop_
        .iter()
        .map(|&p| {
            (0..hull.len())
                .map(|k| distance_to_segment(p, hull[k], hull[(k + 1) % hull.len()]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    // start the walk at a point on the hull, so no run wraps round the end
    let Some(start) = (0..loop_.len()).min_by(|&a, &b| inset[a].total_cmp(&inset[b])) else {
        return Vec::new();
    };
    let mut out: Vec<usize> = Vec::new();
    let mut run: Option<usize> = None;
    for step in 0..=loop_.len() {
        let i = (start + step) % loop_.len();
        if inset[i] > depth {
            run = Some(run.map_or(i, |r| if inset[i] > inset[r] { i } else { r }));
        } else if let Some(r) = run.take() {
            out.push(r);
        }
    }
    out.sort_by(|&a, &b| inset[b].total_cmp(&inset[a]));
    out
}

/// Whether `p` is inside the closed polygon `points` (even-odd).
pub fn inside_polygon(p: Point, points: &[Point]) -> bool {
    let n = points.len();
    let mut inside = false;
    for k in 0..n {
        let (a, b) = (points[k], points[(k + 1) % n]);
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < a.0 + (p.1 - a.1) / (b.1 - a.1) * (b.0 - a.0) {
            inside = !inside;
        }
    }
    inside
}

fn signed_area(points: &[Point]) -> f64 {
    let n = points.len();
    (0..n)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum::<f64>()
        / 2.0
}

fn distance_to_segment(p: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (a.0 + t * dx - p.0, a.1 + t * dy - p.1);
    (qx * qx + qy * qy).sqrt()
}

fn douglas_peucker(points: &[Point], tolerance: f64, keep: &mut [bool], first: usize, last: usize) {
    if last <= first + 1 {
        return;
    }
    let (mut worst, mut at) = (0.0, first);
    for i in first + 1..last {
        let d = distance_to_segment(points[i], points[first], points[last]);
        if d > worst {
            worst = d;
            at = i;
        }
    }
    if worst > tolerance {
        keep[at] = true;
        douglas_peucker(points, tolerance, keep, first, at);
        douglas_peucker(points, tolerance, keep, at, last);
    }
}

/// A closed loop simplified to within `tolerance` px (Douglas-Peucker, split at its two farthest-apart points).
pub fn simplify_closed(loop_: &[Point], tolerance: f64) -> Vec<Point> {
    let n = loop_.len();
    let far = (0..n)
        .max_by(|&i, &j| {
            let d = |k: usize| (loop_[k].0 - loop_[0].0).powi(2) + (loop_[k].1 - loop_[0].1).powi(2);
            d(i).total_cmp(&d(j))
        })
        .unwrap_or(0);
    let mut points = loop_.to_vec();
    points.push(loop_[0]);
    let mut keep = vec![false; n + 1];
    keep[0] = true;
    keep[far] = true;
    douglas_peucker(&points, tolerance, &mut keep, 0, far);
    douglas_peucker(&points, tolerance, &mut keep, far, n);
    (0..n).filter(|&i| keep[i]).map(|i| loop_[i]).collect()
}

// --- Bézier fitting -----------------------------------------------------------------------------------------------

type Cubic = [Point; 4];

fn sub(a: Point, b: Point) -> Point {
    (a.0 - b.0, a.1 - b.1)
}
fn add(a: Point, b: Point) -> Point {
    (a.0 + b.0, a.1 + b.1)
}
fn scale(a: Point, s: f64) -> Point {
    (a.0 * s, a.1 * s)
}
fn dot(a: Point, b: Point) -> f64 {
    a.0 * b.0 + a.1 * b.1
}
fn length(a: Point) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: Point) -> Point {
    let l = length(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        (0.0, 0.0)
    }
}

fn bezier(c: &Cubic, t: f64) -> Point {
    let u = 1.0 - t;
    let (a, b, cc, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    (
        a * c[0].0 + b * c[1].0 + cc * c[2].0 + d * c[3].0,
        a * c[0].1 + b * c[1].1 + cc * c[2].1 + d * c[3].1,
    )
}

fn bezier_d1(c: &Cubic, t: f64) -> Point {
    let u = 1.0 - t;
    let p = scale(sub(c[1], c[0]), 3.0 * u * u);
    let q = scale(sub(c[2], c[1]), 6.0 * u * t);
    let r = scale(sub(c[3], c[2]), 3.0 * t * t);
    add(add(p, q), r)
}

fn bezier_d2(c: &Cubic, t: f64) -> Point {
    let p = scale(add(sub(c[2], scale(c[1], 2.0)), c[0]), 6.0 * (1.0 - t));
    let q = scale(add(sub(c[3], scale(c[2], 2.0)), c[1]), 6.0 * t);
    add(p, q)
}

fn chord_lengths(d: &[Point]) -> Vec<f64> {
    let mut u = vec![0.0; d.len()];
    for i in 1..d.len() {
        u[i] = u[i - 1] + length(sub(d[i], d[i - 1]));
    }
    let total = u[d.len() - 1];
    if total > 0.0 {
        u.iter_mut().for_each(|v| *v /= total);
    }
    u
}

fn generate(d: &[Point], u: &[f64], t1: Point, t2: Point) -> Cubic {
    let (first, last) = (d[0], d[d.len() - 1]);
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (p, &t) in d.iter().zip(u) {
        let s = 1.0 - t;
        let (b0, b1, b2, b3) = (s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t);
        let (a1, a2) = (scale(t1, b1), scale(t2, b2));
        c00 += dot(a1, a1);
        c01 += dot(a1, a2);
        c11 += dot(a2, a2);
        let tmp = sub(*p, add(scale(first, b0 + b1), scale(last, b2 + b3)));
        x0 += dot(a1, tmp);
        x1 += dot(a2, tmp);
    }
    let det = c00 * c11 - c01 * c01;
    let seg = length(sub(last, first));
    let eps = 1e-6 * seg;
    let (mut alpha1, mut alpha2) = if det.abs() > 1e-12 {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (0.0, 0.0)
    };
    if alpha1 < eps || alpha2 < eps {
        alpha1 = seg / 3.0;
        alpha2 = seg / 3.0;
    }
    [first, add(first, scale(t1, alpha1)), add(last, scale(t2, alpha2)), last]
}

/// The largest squared distance from a point to its parameter's place on the curve, and where.
fn max_error(d: &[Point], c: &Cubic, u: &[f64]) -> (f64, usize) {
    let (mut worst, mut at) = (0.0, d.len() / 2);
    for i in 1..d.len() - 1 {
        let e = sub(bezier(c, u[i]), d[i]);
        let e2 = dot(e, e);
        if e2 >= worst {
            worst = e2;
            at = i;
        }
    }
    (worst, at)
}

fn reparameterize(d: &[Point], u: &[f64], c: &Cubic) -> Vec<f64> {
    d.iter()
        .zip(u)
        .map(|(&p, &t)| {
            let q = sub(bezier(c, t), p);
            let (q1, q2) = (bezier_d1(c, t), bezier_d2(c, t));
            let den = dot(q1, q1) + dot(q, q2);
            if den.abs() < 1e-12 {
                t
            } else {
                (t - dot(q, q1) / den).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Schneider's recursive fit of `d` (at least two points) with end tangents `t1` (leaving the first point) and `t2`
/// (leaving the last point, backwards), every point within `error` px.
fn fit_cubics(d: &[Point], t1: Point, t2: Point, error: f64, out: &mut Vec<Cubic>) {
    let (first, last) = (d[0], d[d.len() - 1]);
    // a run that is straight to within the error is a line, whatever the tangents say
    if d.len() == 2 || d.iter().all(|&p| distance_to_segment(p, first, last) < error) {
        let (a, b) = (
            add(first, scale(sub(last, first), 1.0 / 3.0)),
            add(first, scale(sub(last, first), 2.0 / 3.0)),
        );
        if d.len() > 2 {
            out.push([first, a, b, last]);
            return;
        }
    }
    if d.len() == 2 {
        let dist = length(sub(last, first)) / 3.0;
        out.push([first, add(first, scale(t1, dist)), add(last, scale(t2, dist)), last]);
        return;
    }
    let mut u = chord_lengths(d);
    let mut c = generate(d, &u, t1, t2);
    let (mut worst, mut split) = max_error(d, &c, &u);
    let e2 = error * error;
    if worst < e2 {
        out.push(c);
        return;
    }
    if worst < 4.0 * e2 {
        for _ in 0..4 {
            u = reparameterize(d, &u, &c);
            c = generate(d, &u, t1, t2);
            (worst, split) = max_error(d, &c, &u);
            if worst < e2 {
                out.push(c);
                return;
            }
        }
    }
    let split = split.clamp(1, d.len() - 2);
    let centre = unit(sub(d[split - 1], d[split + 1]));
    fit_cubics(&d[..=split], t1, centre, error, out);
    fit_cubics(&d[split..], scale(centre, -1.0), t2, error, out);
}

/// Corners of a dense closed loop: points where the direction turns by more than ~50 degrees over `k` points either
/// side, and turns most there.
fn corners(loop_: &[Point], k: usize) -> Vec<usize> {
    let n = loop_.len();
    if n < 2 * k + 3 {
        return Vec::new();
    }
    let cosine = |i: usize| {
        let a = sub(loop_[i], loop_[(i + n - k) % n]);
        let b = sub(loop_[(i + k) % n], loop_[i]);
        let l = length(a) * length(b);
        if l > 0.0 {
            dot(a, b) / l
        } else {
            1.0
        }
    };
    let cos: Vec<f64> = (0..n).map(cosine).collect();
    (0..n)
        .filter(|&i| cos[i] < 0.64 && (1..=k).all(|j| cos[i] <= cos[(i + j) % n] && cos[i] < cos[(i + n - j) % n]))
        .collect()
}

/// The loop as a closed path of cubics (straight ones written as lines), cut at its corners, every point within
/// `error` px. None for a degenerate loop.
pub fn bezier_path(loop_: &[Point], error: f64) -> Option<Geometry> {
    let n = loop_.len();
    if n < 4 {
        return None;
    }
    const K: usize = 3;
    let mut cuts = corners(loop_, K);
    let smooth_start = cuts.is_empty();
    if smooth_start {
        cuts = vec![0, n / 2];
    }
    let tangent_at = |i: usize, forward: bool| -> Point {
        let is_corner = !smooth_start && cuts.contains(&i);
        let span = K.min(n / 4).max(1);
        if is_corner {
            if forward {
                unit(sub(loop_[(i + span) % n], loop_[i]))
            } else {
                unit(sub(loop_[(i + n - span) % n], loop_[i]))
            }
        } else {
            let t = unit(sub(loop_[(i + span) % n], loop_[(i + n - span) % n]));
            if forward {
                t
            } else {
                scale(t, -1.0)
            }
        }
    };
    let mut cubics: Vec<Cubic> = Vec::new();
    for (j, &start) in cuts.iter().enumerate() {
        let end = cuts[(j + 1) % cuts.len()];
        let mut run: Vec<Point> = Vec::new();
        let mut i = start;
        loop {
            run.push(loop_[i]);
            if i == end && run.len() > 1 {
                break;
            }
            i = (i + 1) % n;
        }
        if run.len() < 2 {
            continue;
        }
        fit_cubics(
            &run,
            tangent_at(start, true),
            tangent_at(end, false),
            error,
            &mut cubics,
        );
    }
    if cubics.is_empty() {
        return None;
    }
    let mut segments = vec![Segment::Move(cubics[0][0])];
    for c in &cubics {
        let chord = (c[0], c[3]);
        let straight = distance_to_segment(c[1], chord.0, chord.1) < 0.25 * error
            && distance_to_segment(c[2], chord.0, chord.1) < 0.25 * error;
        segments.push(if straight {
            Segment::Line(c[3])
        } else {
            Segment::Cubic(c[1], c[2], c[3])
        });
    }
    segments.push(Segment::Close);
    Some(Geometry::Path(segments))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk(width: usize, height: usize, cx: f64, cy: f64, r: f64) -> Vec<f64> {
        (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f64 + 0.5 - cx, (i / width) as f64 + 0.5 - cy);
                if x * x + y * y <= r * r {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn a_disk_is_described_as_a_circle() {
        let visible = disk(48, 48, 20.3, 25.1, 13.2);
        let free = vec![false; 48 * 48];
        let target = Target {
            width: 48,
            height: 48,
            visible: &visible,
            free: &free,
        };
        let fit = target.fit(&Costs::new(48, 48, 2.0)).unwrap();
        let Geometry::Circle { cx, cy, r } = fit.geometry else {
            panic!("{:?}", fit.geometry)
        };
        assert!(
            (cx - 20.3).abs() < 0.3 && (cy - 25.1).abs() < 0.3 && (r - 13.2).abs() < 0.3,
            "{cx} {cy} {r}"
        );
    }

    #[test]
    fn a_hidden_part_completes_itself() {
        // a disk with its lower right covered by a square above it: the free pixels let the whole circle win
        let full = disk(48, 48, 20.0, 20.0, 12.0);
        let free: Vec<bool> = (0..48 * 48).map(|i| i % 48 >= 22 && i / 48 >= 22).collect();
        let visible: Vec<f64> = full.iter().zip(&free).map(|(v, f)| if *f { 0.0 } else { *v }).collect();
        let target = Target {
            width: 48,
            height: 48,
            visible: &visible,
            free: &free,
        };
        let fit = target.fit(&Costs::new(48, 48, 2.0)).unwrap();
        let Geometry::Circle { cx, cy, r } = fit.geometry else {
            panic!("{:?}", fit.geometry)
        };
        assert!(
            (cx - 20.0).abs() < 0.4 && (cy - 20.0).abs() < 0.4 && (r - 12.0).abs() < 0.4,
            "{cx} {cy} {r}"
        );
    }

    #[test]
    fn contours_and_paths_follow_the_mask() {
        let mask = disk(40, 40, 20.0, 20.0, 10.0);
        let loop_ = largest_contour(&mask, 40, 40).unwrap();
        let area = signed_area(&loop_).abs();
        assert!((area - std::f64::consts::PI * 100.0).abs() < 8.0, "{area}");
        let Some(Geometry::Path(segments)) = bezier_path(&loop_, 0.5) else {
            panic!()
        };
        assert!(segments.len() <= 10, "{} segments", segments.len());
        // a square's corners are kept, its sides are lines
        let square: Vec<f64> = (0..1600)
            .map(|i| ((8..30).contains(&(i % 40)) && (5..25).contains(&(i / 40))) as u8 as f64)
            .collect();
        let loop_ = largest_contour(&square, 40, 40).unwrap();
        assert_eq!(corners(&loop_, 3).len(), 4);
        let Some(Geometry::Path(segments)) = bezier_path(&loop_, 0.5) else {
            panic!()
        };
        assert!(
            segments.iter().filter(|s| matches!(s, Segment::Line(_))).count() >= 4,
            "{segments:?}"
        );
    }

    #[test]
    fn an_outline_is_refined_onto_its_edges() {
        // an anti-aliased triangle with sub-pixel corners: the traced contour lands near them, refinement onto them
        let corners = [(8.3, 9.6), (40.7, 14.2), (19.4, 39.1)];
        let mask = coverage(&vec![corners.to_vec()], 48, 48);
        let mut visible = vec![0.0; 48 * 48];
        for my in 0..mask.height {
            for mx in 0..mask.width {
                visible[(mask.y0 + my) * 48 + mask.x0 + mx] = mask.data[my * mask.width + mx];
            }
        }
        let free = vec![false; 48 * 48];
        let target = Target {
            width: 48,
            height: 48,
            visible: &visible,
            free: &free,
        };
        let fit = target.fit(&Costs::new(48, 48, 12.0)).unwrap();
        let Geometry::Polygon(points) = &fit.geometry else {
            panic!("{:?}", fit.geometry)
        };
        assert_eq!(points.len(), 3, "{points:?}");
        for corner in corners {
            let nearest = points
                .iter()
                .map(|&p| length(sub(p, corner)))
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 0.3, "{corner:?} {points:?}");
        }
    }

    #[test]
    fn least_squares_knots_recover_a_spline() {
        let knots = vec![(30.0, 10.0), (40.0, 28.0), (26.0, 40.0), (10.0, 30.0), (12.0, 14.0)];
        let Geometry::Path(segments) = spline(&knots) else {
            unreachable!()
        };
        let mut samples = Vec::new();
        let mut at = knots[0];
        for segment in &segments {
            if let Segment::Cubic(c1, c2, end) = *segment {
                let cubic = [at, c1, c2, end];
                samples.extend((0..40).map(|s| bezier(&cubic, s as f64 / 40.0)));
                at = end;
            }
        }
        let fitted = fit_knots(&samples, &samples, 5).unwrap();
        // every sample within a tenth of a pixel of the fitted spline
        let Geometry::Path(fitted_segments) = spline(&fitted) else {
            unreachable!()
        };
        let mut curve = Vec::new();
        let mut at = fitted[0];
        for segment in &fitted_segments {
            if let Segment::Cubic(c1, c2, end) = *segment {
                let cubic = [at, c1, c2, end];
                curve.extend((0..400).map(|s| bezier(&cubic, s as f64 / 400.0)));
                at = end;
            }
        }
        let worst = samples
            .iter()
            .map(|p| curve.iter().map(|c| length(sub(*c, *p))).fold(f64::INFINITY, f64::min))
            .fold(0.0, f64::max);
        assert!(worst < 0.1, "{worst} px: {knots:?} vs {fitted:?}");
    }

    #[test]
    fn a_smooth_blob_is_described_as_a_spline() {
        // the closed spline through five knots, anti-aliased: its knots say it more briefly than cubics or a polygon
        let knots = vec![(24.0, 6.3), (41.2, 19.0), (35.5, 40.1), (12.4, 38.2), (7.1, 17.6)];
        let mask = coverage(
            &Shape {
                geometry: spline(&knots),
                fill: [0, 0, 0],
                opacity: 1.0,
                group: None,
            }
            .outline(),
            48,
            48,
        );
        let mut visible = vec![0.0; 48 * 48];
        for my in 0..mask.height {
            for mx in 0..mask.width {
                visible[(mask.y0 + my) * 48 + mask.x0 + mx] = mask.data[my * mask.width + mx];
            }
        }
        let free = vec![false; 48 * 48];
        let target = Target {
            width: 48,
            height: 48,
            visible: &visible,
            free: &free,
        };
        let fit = target.fit(&Costs::new(48, 48, 12.0)).unwrap();
        let fitted = fit.knots.as_ref().unwrap_or_else(|| panic!("{:?}", fit.geometry));
        assert_eq!(fitted.len(), 5, "{fitted:?}");
        // knots found elsewhere on the curve can't trace it exactly: ~0.01 of a pixel per edge pixel is left over
        assert!(target.misfit(&fit.geometry) < 2.0, "{}", target.misfit(&fit.geometry));
    }

    #[test]
    fn hulls_and_solves() {
        let hull = convex_hull(vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (1.0, 0.0),
        ]);
        assert_eq!(hull.len(), 4);
        assert_eq!(
            solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0]).map(|x| x.map(|v| (v * 1e9).round() / 1e9)),
            Some([0.8, 1.4])
        );
    }
}

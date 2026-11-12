//! Polish (docs/design/0022, the ladder's refinement rung): the finished shapes' numbers nudged against the image
//! itself, with the structure held fixed. Each part was fitted to its own soft mask, one at a time; here every shape is
//! judged as it is drawn -- under the shapes above it, over the ones below and the background -- by the squared error of
//! the whole render against the input. The error is the rasterizer's exact coverage, so what is polished is what the
//! SVG draws.

use crate::copycat::{Geometry, Segment, Shape};
use crate::copycat_fit::{crosses_itself, drawn_spline, hermite_path, params, rounded_rect, with_params, Fit, Hermite};
use crate::raster::{Mask, Point};

/// Step sizes (px), coarse to fine: the coarse ones catch a shape fitted a pixel off, the fine ones settle its edge.
const STEPS: [f64; 6] = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02];
/// Passes over every number at one step size, at most.
const ROUNDS: usize = 6;
/// The steps a new knot is settled in (px).
const GROW_STEPS: [f64; 3] = [0.5, 0.25, 0.1];
/// Points a spline or a polygon may grow to.
const MAX_POINTS: usize = 32;

/// OKLab to straight sRGB in 0..1 (the inverse of `copycat_trace::oklab`, unrounded).
pub fn srgb(lab: [f64; 3]) -> [f64; 3] {
    let [l, a, b] = lab;
    let l_ = l + 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let m_ = l - 0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let s_ = l - 0.089_484_177_5 * a - 1.291_485_548_0 * b;
    let (l3, m3, s3) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    let linear = [
        4.076_741_662_1 * l3 - 3.307_711_591_3 * m3 + 0.230_969_929_2 * s3,
        -1.268_438_004_6 * l3 + 2.609_757_401_1 * m3 - 0.341_319_396_5 * s3,
        -0.004_196_086_3 * l3 - 0.703_418_614_7 * m3 + 1.707_614_701_0 * s3,
    ];
    linear.map(|c| {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * libm::pow(c, 1.0 / 2.4) - 0.055
        }
    })
}

/// A grown shape: the error it saves, its points and their corners, and how it draws.
type Grown = (f64, Vec<Point>, Vec<bool>, Shape, Mask);

/// The numbers a shape was fitted in, so polishing moves what the fit chose: a spline's knots (its curve stays as
/// smooth as the spline is), a rounded rectangle's five, or the geometry's own numbers.
#[derive(Clone, Debug)]
pub enum Form {
    Plain,
    Knots {
        knots: Vec<Point>,
        tight: bool,
        corners: Vec<bool>,
    },
    Rounded([f64; 5]),
    /// A smooth path whose knots carry their own tangents: polished by its knots' five numbers, never crossing itself.
    Hermite(Vec<Hermite>),
}

impl Form {
    pub fn of(fit: &Fit) -> Form {
        match (&fit.knots, fit.rounded) {
            (Some(knots), _) => Form::Knots {
                knots: knots.clone(),
                tight: fit.tight,
                corners: fit.corners.clone(),
            },
            (_, Some(p)) => Form::Rounded(p),
            _ => match &fit.hermite {
                Some(knots) => Form::Hermite(knots.clone()),
                None => Form::Plain,
            },
        }
    }

    fn numbers(&self, geometry: &Geometry) -> Vec<f64> {
        match self {
            Form::Knots { knots, .. } => knots.iter().flat_map(|&(x, y)| [x, y]).collect(),
            Form::Rounded(p) => p.to_vec(),
            Form::Hermite(knots) => knots.iter().flatten().copied().collect(),
            Form::Plain => params(geometry),
        }
    }

    fn moves(&self, geometry: &Geometry) -> Vec<Vec<usize>> {
        match self {
            Form::Knots { knots, .. } => (0..2 * knots.len()).map(|k| vec![k]).collect(),
            Form::Rounded(_) => (0..5).map(|k| vec![k]).collect(),
            Form::Hermite(knots) => (0..5 * knots.len()).map(|k| vec![k]).collect(),
            Form::Plain => moves(geometry),
        }
    }

    /// The form and geometry with `p` for its numbers.
    fn with(&self, geometry: &Geometry, p: &[f64]) -> (Form, Geometry) {
        match self {
            Form::Knots { tight, corners, .. } => {
                let knots: Vec<Point> = (0..p.len() / 2).map(|i| (p[2 * i], p[2 * i + 1])).collect();
                let g = drawn_spline(&knots, *tight, corners);
                let form = Form::Knots {
                    knots,
                    tight: *tight,
                    corners: corners.clone(),
                };
                (form, g)
            }
            Form::Rounded(_) => {
                let q = [p[0], p[1], p[2], p[3], p[4]];
                (Form::Rounded(q), rounded_rect(q))
            }
            Form::Hermite(_) => {
                let knots: Vec<Hermite> = p.chunks(5).map(|c| [c[0], c[1], c[2], c[3], c[4]]).collect();
                let g = hermite_path(&knots);
                (Form::Hermite(knots), g)
            }
            Form::Plain => (Form::Plain, with_params(geometry, p)),
        }
    }
}

/// The numbers that move together: a primitive's each, a polygon's points, and a path's anchors, each carried with the
/// handles either side of it (so a smooth join stays smooth). Each move is a list of (index, axis) into `params`.
fn moves(geometry: &Geometry) -> Vec<Vec<usize>> {
    match geometry {
        Geometry::Circle { .. } => (0..3).map(|k| vec![k]).collect(),
        Geometry::Ellipse { .. } | Geometry::Rect { .. } => (0..4).map(|k| vec![k]).collect(),
        Geometry::Polygon(points) => (0..2 * points.len()).map(|k| vec![k]).collect(),
        Geometry::Path(segments) => {
            // each segment's points' offsets into params, and which is its end (the anchor)
            let mut offsets: Vec<(Vec<usize>, Option<usize>)> = Vec::new();
            let mut k = 0;
            for segment in segments {
                let n = match segment {
                    Segment::Move(_) | Segment::Line(_) => 1,
                    Segment::Quad(..) => 2,
                    Segment::Cubic(..) => 3,
                    Segment::Close => 0,
                };
                let points: Vec<usize> = (0..n).map(|j| k + 2 * j).collect();
                offsets.push((points.clone(), points.last().copied()));
                k += 2 * n;
            }
            let mut out = Vec::new();
            for (s, (_, anchor)) in offsets.iter().enumerate() {
                let Some(anchor) = *anchor else { continue };
                let mut group = vec![anchor];
                // the handle before it (this segment's second-to-last point) and after it (the next one's first)
                if let (Segment::Cubic(..), points) = (&segments[s], &offsets[s].0) {
                    group.push(points[1]);
                }
                if let Some((Segment::Cubic(..), (points, _))) = segments.get(s + 1).zip(offsets.get(s + 1)) {
                    group.push(points[0]);
                }
                for axis in 0..2 {
                    out.push(group.iter().map(|&p| p + axis).collect());
                }
            }
            out
        }
    }
}

struct Box_ {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

fn bounds(mask: &Mask) -> Box_ {
    Box_ {
        x0: mask.x0,
        y0: mask.y0,
        x1: mask.x0 + mask.width,
        y1: mask.y0 + mask.height,
    }
}

fn at(mask: &Mask, x: usize, y: usize) -> f64 {
    if x < mask.x0 || y < mask.y0 || x >= mask.x0 + mask.width || y >= mask.y0 + mask.height {
        return 0.0;
    }
    mask.data[(y - mask.y0) * mask.width + (x - mask.x0)]
}

/// The image and background the shapes are judged against, straight sRGB in 0..1.
pub struct Judge<'a> {
    pub width: usize,
    pub height: usize,
    pub image: &'a [[f64; 3]],
    pub background: &'a [[f64; 3]],
    /// What a knot costs to say, in wrongly covered pixels; 0 never adds one.
    pub knot_price: f64,
    /// What a radian of turning a new point adds to its outline costs, in wrongly covered pixels. A convex outline
    /// turns 2 pi however many points it has, so this charges only for dents and zigzags: a point that follows a real
    /// bend pays for its coordinates, one that chases pixel stair-steps pays for every turn too.
    pub turn_price: f64,
}

impl Judge<'_> {
    /// The squared error over `region` of `masks` (in paint order, with `fills`) composited over the background.
    fn error(&self, masks: &[&Mask], fills: &[[f64; 3]], region: &Box_) -> f64 {
        let near: Vec<usize> = (0..masks.len())
            .filter(|&j| {
                let b = bounds(masks[j]);
                b.x0 < region.x1 && region.x0 < b.x1 && b.y0 < region.y1 && region.y0 < b.y1
            })
            .collect();
        let mut sum = 0.0;
        for y in region.y0..region.y1 {
            for x in region.x0..region.x1 {
                let (mut c, mut a) = ([0.0; 3], 0.0);
                for &j in &near {
                    let cover = at(masks[j], x, y);
                    if cover == 0.0 {
                        continue;
                    }
                    for (k, v) in c.iter_mut().enumerate() {
                        *v = fills[j][k] * cover + *v * (1.0 - cover);
                    }
                    a = cover + a * (1.0 - cover);
                }
                let i = y * self.width + x;
                for (k, v) in c.iter().enumerate() {
                    let d = v + (1.0 - a) * self.background[i][k] - self.image[i][k];
                    sum += d * d;
                }
            }
        }
        sum
    }

    /// How much the error changes over the region either mask covers when shape `k`'s mask becomes `mask`.
    fn delta(&self, masks: &[Mask], fills: &[[f64; 3]], k: usize, mask: &Mask) -> f64 {
        let (a, b) = (bounds(&masks[k]), bounds(mask));
        let region = Box_ {
            x0: a.x0.min(b.x0),
            y0: a.y0.min(b.y0),
            x1: a.x1.max(b.x1).min(self.width),
            y1: a.y1.max(b.y1).min(self.height),
        };
        let all: Vec<&Mask> = masks.iter().collect();
        let before = self.error(&all, fills, &region);
        let moved: Vec<&Mask> = masks
            .iter()
            .enumerate()
            .map(|(j, m)| if j == k { mask } else { m })
            .collect();
        self.error(&moved, fills, &region) - before
    }

    /// The squared error one wrongly covered pixel of shape `k` costs, on average along its outline: its fill against
    /// what is drawn under it, where nothing above hides the difference.
    fn contrast(&self, masks: &[Mask], fills: &[[f64; 3]], k: usize) -> f64 {
        let b = bounds(&masks[k]);
        let (mut sum, mut n) = (0.0, 0.0);
        for y in b.y0..b.y1.min(self.height) {
            for x in b.x0..b.x1.min(self.width) {
                let cover = at(&masks[k], x, y);
                if cover <= 0.0 || cover >= 1.0 {
                    continue;
                }
                let mut under = self.background[y * self.width + x];
                for j in 0..k {
                    let c = at(&masks[j], x, y);
                    for (v, f) in under.iter_mut().zip(fills[j]) {
                        *v = f * c + *v * (1.0 - c);
                    }
                }
                let seen: f64 = masks[k + 1..].iter().map(|m| 1.0 - at(m, x, y)).product();
                sum += seen * (0..3).map(|c| (fills[k][c] - under[c]).powi(2)).sum::<f64>();
                n += 1.0;
            }
        }
        if n > 0.0 {
            sum / n
        } else {
            0.0
        }
    }

    /// Every shape's numbers (in its `Form`) nudged while that lowers the error; opaque shapes only (all the pipeline
    /// writes). With a `knot_price`, a spline then gains knots where its outline can't follow the image, each kept
    /// only if it saves more than it costs to say, and the shapes are settled again.
    pub fn polish(&self, shapes: &mut [Shape], forms: &mut [Form]) {
        self.polish_with(shapes, forms, &mut |_| {});
    }

    /// `polish`, telling `progress` how far through it is (0 to 1) after each pass over the shapes.
    pub fn polish_with(&self, shapes: &mut [Shape], forms: &mut [Form], progress: &mut dyn FnMut(f64)) {
        let (w, h) = (self.width as u32, self.height as u32);
        let fills: Vec<[f64; 3]> = shapes.iter().map(|s| s.fill.map(|c| c as f64 / 255.0)).collect();
        let mut masks: Vec<Mask> = shapes.iter().map(|s| s.mask(w, h)).collect();
        if self.knot_price <= 0.0 {
            self.descend(shapes, forms, &mut masks, &fills, &STEPS, progress);
            progress(1.0);
            return;
        }
        // measured in WebAssembly on `icons` and `broad`: growing takes about as long as the first descent, the final
        // settle two thirds of that
        let (descended, grown) = (0.38, 0.76);
        self.descend(shapes, forms, &mut masks, &fills, &STEPS, &mut |done| {
            progress(descended * done)
        });
        let grew = self.grow(shapes, forms, &mut masks, &fills, &mut |done| {
            progress(descended + (grown - descended) * done)
        });
        if grew {
            self.descend(shapes, forms, &mut masks, &fills, &STEPS[2..], &mut |done| {
                progress(grown + (1.0 - grown) * done)
            });
        }
        progress(1.0);
    }

    fn descend(
        &self,
        shapes: &mut [Shape],
        forms: &mut [Form],
        masks: &mut [Mask],
        fills: &[[f64; 3]],
        steps: &[f64],
        progress: &mut dyn FnMut(f64),
    ) {
        let (w, h) = (self.width as u32, self.height as u32);
        for (s, &step) in steps.iter().enumerate() {
            for round in 0..ROUNDS {
                progress((s as f64 + round as f64 / ROUNDS as f64) / steps.len() as f64);
                let mut improved = false;
                for k in 0..shapes.len() {
                    for group in forms[k].moves(&shapes[k].geometry) {
                        for sign in [1.0, -1.0] {
                            let mut p = forms[k].numbers(&shapes[k].geometry);
                            for &i in &group {
                                p[i] += sign * step;
                            }
                            let (form, geometry) = forms[k].with(&shapes[k].geometry, &p);
                            if matches!(form, Form::Hermite(_)) && crosses_itself(&geometry) {
                                continue;
                            }
                            let moved = Shape {
                                geometry,
                                ..shapes[k].clone()
                            };
                            let mask = moved.mask(w, h);
                            if self.delta(masks, fills, k, &mask) < -1e-9 {
                                shapes[k] = moved;
                                forms[k] = form;
                                masks[k] = mask;
                                improved = true;
                                break;
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
    }

    /// Points added to splines' knots and polygons' corners, one at a time, at whichever gap saves the most: a new one
    /// starts on the outline, midway along its segment, and is settled there. It is kept if the error it saves is worth
    /// more than it costs: `knot_price` wrongly covered pixels, and `turn_price` more for every radian of turning it
    /// adds; it is settled on that same total, so it finds where it bends least. Whether any shape grew.
    fn grow(
        &self,
        shapes: &mut [Shape],
        forms: &mut [Form],
        masks: &mut [Mask],
        fills: &[[f64; 3]],
        progress: &mut dyn FnMut(f64),
    ) -> bool {
        let (w, h) = (self.width as u32, self.height as u32);
        let mut grown = false;
        for k in 0..shapes.len() {
            progress(k as f64 / shapes.len() as f64);
            // the points, how they're drawn (a spline's tightness, or None for a polygon), and each gap's midpoint
            while let Some((points, tight, corners)) = match (&forms[k], &shapes[k].geometry) {
                (Form::Knots { knots, tight, corners }, _) => Some((knots.clone(), Some(*tight), corners.clone())),
                (Form::Plain, Geometry::Polygon(corners)) => Some((corners.clone(), None, Vec::new())),
                _ => None,
            } {
                if points.len() >= MAX_POINTS {
                    break;
                }
                let n = points.len();
                let mids: Vec<Point> = match &shapes[k].geometry {
                    Geometry::Path(segments) if tight.is_some() => (0..n)
                        .map(|i| match segments.get(i + 1) {
                            Some(&Segment::Cubic(c1, c2, p3)) => {
                                let p0 = points[i];
                                (
                                    (p0.0 + 3.0 * c1.0 + 3.0 * c2.0 + p3.0) / 8.0,
                                    (p0.1 + 3.0 * c1.1 + 3.0 * c2.1 + p3.1) / 8.0,
                                )
                            }
                            _ => points[i],
                        })
                        .collect(),
                    _ => (0..n)
                        .map(|i| {
                            let (a, b) = (points[i], points[(i + 1) % n]);
                            ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0)
                        })
                        .collect(),
                };
                let contrast = self.contrast(masks, fills, k);
                let price = self.knot_price * contrast;
                let turned = shapes[k].turning();
                // the error a candidate saves (negative), with what its added turning costs
                let total = |delta: f64, shape: &Shape| delta + self.turn_price * contrast * (shape.turning() - turned);
                // a new point is smooth
                let flags = |i: usize| -> Vec<bool> {
                    let mut f = corners.clone();
                    if !f.is_empty() {
                        f.insert(i + 1, false);
                    }
                    f
                };
                let draw = |points: &[Point], flags: &[bool]| {
                    let geometry = match tight {
                        Some(tight) => drawn_spline(points, tight, flags),
                        None => Geometry::Polygon(points.to_vec()),
                    };
                    let shape = Shape {
                        geometry,
                        ..shapes[k].clone()
                    };
                    let mask = shape.mask(w, h);
                    (shape, mask)
                };
                let mut best: Option<Grown> = None;
                for (i, &mid) in mids.iter().enumerate() {
                    let mut more = points.clone();
                    more.insert(i + 1, mid);
                    let more_flags = flags(i);
                    let (mut shape, mut mask) = draw(&more, &more_flags);
                    let mut saved = total(self.delta(masks, fills, k, &mask), &shape);
                    for step in GROW_STEPS {
                        for _round in 0..ROUNDS {
                            let mut improved = false;
                            for axis in 0..2 {
                                for sign in [1.0, -1.0] {
                                    let mut tried = more.clone();
                                    if axis == 0 {
                                        tried[i + 1].0 += sign * step;
                                    } else {
                                        tried[i + 1].1 += sign * step;
                                    }
                                    let (s, m) = draw(&tried, &more_flags);
                                    let d = total(self.delta(masks, fills, k, &m), &s);
                                    if d < saved - 1e-9 {
                                        (more, shape, mask, saved) = (tried, s, m, d);
                                        improved = true;
                                        break;
                                    }
                                }
                            }
                            if !improved {
                                break;
                            }
                        }
                    }
                    if saved < -price && best.as_ref().is_none_or(|b| saved < b.0) {
                        best = Some((saved, more, more_flags, shape, mask));
                    }
                }
                let Some((_, points, corners, shape, mask)) = best else {
                    break;
                };
                forms[k] = match tight {
                    Some(tight) => Form::Knots {
                        knots: points,
                        tight,
                        corners,
                    },
                    None => Form::Plain,
                };
                shapes[k] = shape;
                masks[k] = mask;
                grown = true;
            }
        }
        grown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copycat::Document;
    use crate::copycat_fit::spline;

    const ORANGE: [u8; 3] = [230, 120, 30];

    /// What `shapes` look like drawn over white, as the input a tracer is given.
    fn drawn(shapes: &[Shape]) -> Vec<[f64; 3]> {
        let canvas = Document {
            width: 64,
            height: 64,
            groups: Vec::new(),
            shapes: shapes.to_vec(),
        }
        .render();
        canvas
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| [0, 1, 2].map(|c| p[c] + 1.0 - p[3]))
            .collect()
    }

    fn polished(truth: &[Shape], start: &mut [Shape], forms: &mut [Form]) {
        polished_at(0.0, truth, start, forms);
    }

    fn polished_at(knot_price: f64, truth: &[Shape], start: &mut [Shape], forms: &mut [Form]) {
        polished_priced(knot_price, 0.0, truth, start, forms);
    }

    fn polished_priced(knot_price: f64, turn_price: f64, truth: &[Shape], start: &mut [Shape], forms: &mut [Form]) {
        let image = drawn(truth);
        let white = vec![[1.0; 3]; 64 * 64];
        Judge {
            width: 64,
            height: 64,
            image: &image,
            background: &white,
            knot_price,
            turn_price,
        }
        .polish(start, forms);
    }

    fn error(truth: &[Shape], shapes: &[Shape]) -> f64 {
        drawn(shapes)
            .iter()
            .zip(&drawn(truth))
            .map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>())
            .sum()
    }

    fn shape(geometry: Geometry) -> Shape {
        Shape {
            geometry,
            fill: ORANGE,
            opacity: 1.0,
            group: None,
        }
    }

    #[test]
    fn a_circle_fitted_off_its_edge_is_moved_onto_it() {
        let truth = [shape(Geometry::Circle {
            cx: 30.3,
            cy: 33.6,
            r: 14.2,
        })];
        let mut start = [shape(Geometry::Circle {
            cx: 31.0,
            cy: 33.0,
            r: 13.5,
        })];
        polished(&truth, &mut start, &mut [Form::Plain]);
        let Geometry::Circle { cx, cy, r } = start[0].geometry else {
            panic!()
        };
        assert!(
            (cx - 30.3).abs() < 0.03 && (cy - 33.6).abs() < 0.03 && (r - 14.2).abs() < 0.03,
            "{cx} {cy} {r}"
        );
    }

    #[test]
    fn a_spline_is_polished_by_its_knots_and_stays_one() {
        let knots = vec![(20.0, 20.0), (44.0, 18.0), (46.0, 42.0), (22.0, 46.0), (14.0, 32.0)];
        let truth = [shape(spline(&knots))];
        let off: Vec<Point> = knots.iter().map(|&(x, y)| (x + 0.6, y - 0.4)).collect();
        let mut start = [shape(spline(&off))];
        let mut forms = [Form::Knots {
            knots: off,
            tight: false,
            corners: Vec::new(),
        }];
        let before = error(&truth, &start);
        polished(&truth, &mut start, &mut forms);
        let Form::Knots { knots: found, .. } = &forms[0] else {
            panic!()
        };
        assert_eq!(found.len(), knots.len());
        assert_eq!(start[0].geometry, spline(found));
        // a knot can slide along the curve for almost nothing, so it is the drawing that is compared, not the knots
        let after = error(&truth, &start);
        assert!(after < before / 50.0, "error {before} -> {after}");
    }

    #[test]
    fn a_spline_with_too_few_knots_grows_the_ones_its_outline_needs() {
        // a five-pointed star's soft outline, fitted with only five knots (one per point)
        let star: Vec<Point> = (0..10)
            .map(|i| {
                let a = std::f64::consts::PI * i as f64 / 5.0;
                let r = if i % 2 == 0 { 24.0 } else { 12.0 };
                (32.0 + r * a.sin(), 32.0 - r * a.cos())
            })
            .collect();
        let truth = [shape(spline(&star))];
        let few: Vec<Point> = star.iter().step_by(2).copied().collect();
        let start = shape(spline(&few));
        let form = Form::Knots {
            knots: few,
            tight: false,
            corners: Vec::new(),
        };
        let mut fixed = ([start.clone()], [form.clone()]);
        polished(&truth, &mut fixed.0, &mut fixed.1);
        let mut grown = ([start], [form]);
        polished_at(2.0, &truth, &mut grown.0, &mut grown.1);
        let Form::Knots { knots, .. } = &grown.1[0] else {
            panic!()
        };
        assert!(knots.len() >= 10, "{} knots", knots.len());
        let (fixed, grown) = (error(&truth, &fixed.0), error(&truth, &grown.0));
        assert!(grown < fixed / 5.0, "error {fixed} with five knots, {grown} grown");
    }

    #[test]
    fn a_point_that_only_follows_a_notch_pays_for_its_turning() {
        // a square with a notch one pixel deep in its top edge: points can follow the notch, but only by turning
        let notched = Geometry::Polygon(vec![
            (16.0, 16.0),
            (30.0, 16.0),
            (30.0, 17.0),
            (34.0, 17.0),
            (34.0, 16.0),
            (48.0, 16.0),
            (48.0, 48.0),
            (16.0, 48.0),
        ]);
        let truth = [shape(notched)];
        let square = shape(Geometry::Polygon(vec![
            (16.0, 16.0),
            (48.0, 16.0),
            (48.0, 48.0),
            (16.0, 48.0),
        ]));
        let corners = |shapes: &[Shape]| match &shapes[0].geometry {
            Geometry::Polygon(points) => points.len(),
            _ => panic!(),
        };
        let mut free = ([square.clone()], [Form::Plain]);
        polished_priced(0.1, 0.0, &truth, &mut free.0, &mut free.1);
        assert!(corners(&free.0) > 4, "with turning free, the notch is followed");
        let mut priced = ([square.clone()], [Form::Plain]);
        polished_priced(0.1, 20.0, &truth, &mut priced.0, &mut priced.1);
        assert_eq!(corners(&priced.0), 4, "a turn that saves under a pixel isn't worth it");
        assert!(priced.0[0].turning() < free.0[0].turning());
        // a real bend still pays: the star of the test above grows its knots with turning priced (at 2 px a radian)
        let star: Vec<Point> = (0..10)
            .map(|i| {
                let a = std::f64::consts::PI * i as f64 / 5.0;
                let r = if i % 2 == 0 { 24.0 } else { 12.0 };
                (32.0 + r * a.sin(), 32.0 - r * a.cos())
            })
            .collect();
        let few: Vec<Point> = star.iter().step_by(2).copied().collect();
        let mut grown = (
            [shape(spline(&few))],
            [Form::Knots {
                knots: few,
                tight: false,
                corners: Vec::new(),
            }],
        );
        polished_priced(2.0, 2.0, &[shape(spline(&star))], &mut grown.0, &mut grown.1);
        let Form::Knots { knots, .. } = &grown.1[0] else {
            panic!()
        };
        assert!(knots.len() >= 10, "{} knots", knots.len());
    }
}

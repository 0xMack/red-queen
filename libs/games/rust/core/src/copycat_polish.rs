//! Polish (docs/design/0022, the ladder's refinement rung): the finished shapes' numbers nudged against the image
//! itself, with the structure held fixed. Each part was fitted to its own soft mask, one at a time; here every shape is
//! judged as it is drawn -- under the shapes above it, over the ones below and the background -- by the squared error of
//! the whole render against the input. The error is the rasterizer's exact coverage, so what is polished is what the
//! SVG draws.

use crate::copycat::{Geometry, Segment, Shape};
use crate::copycat_fit::{params, rounded_rect, spline, tight_spline, with_params, Fit};
use crate::raster::{Mask, Point};

/// Step sizes (px), coarse to fine: the coarse ones catch a shape fitted a pixel off, the fine ones settle its edge.
const STEPS: [f64; 6] = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02];
/// Passes over every number at one step size, at most.
const ROUNDS: usize = 6;

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

/// The numbers a shape was fitted in, so polishing moves what the fit chose: a spline's knots (its curve stays as
/// smooth as the spline is), a rounded rectangle's five, or the geometry's own numbers.
#[derive(Clone, Debug)]
pub enum Form {
    Plain,
    Knots { knots: Vec<Point>, tight: bool },
    Rounded([f64; 5]),
}

impl Form {
    pub fn of(fit: &Fit) -> Form {
        match (&fit.knots, fit.rounded) {
            (Some(knots), _) => Form::Knots {
                knots: knots.clone(),
                tight: fit.tight,
            },
            (_, Some(p)) => Form::Rounded(p),
            _ => Form::Plain,
        }
    }

    fn numbers(&self, geometry: &Geometry) -> Vec<f64> {
        match self {
            Form::Knots { knots, .. } => knots.iter().flat_map(|&(x, y)| [x, y]).collect(),
            Form::Rounded(p) => p.to_vec(),
            Form::Plain => params(geometry),
        }
    }

    fn moves(&self, geometry: &Geometry) -> Vec<Vec<usize>> {
        match self {
            Form::Knots { knots, .. } => (0..2 * knots.len()).map(|k| vec![k]).collect(),
            Form::Rounded(_) => (0..5).map(|k| vec![k]).collect(),
            Form::Plain => moves(geometry),
        }
    }

    /// The form and geometry with `p` for its numbers.
    fn with(&self, geometry: &Geometry, p: &[f64]) -> (Form, Geometry) {
        match self {
            Form::Knots { tight, .. } => {
                let knots: Vec<Point> = (0..p.len() / 2).map(|i| (p[2 * i], p[2 * i + 1])).collect();
                let g = if *tight { tight_spline(&knots) } else { spline(&knots) };
                (Form::Knots { knots, tight: *tight }, g)
            }
            Form::Rounded(_) => {
                let q = [p[0], p[1], p[2], p[3], p[4]];
                (Form::Rounded(q), rounded_rect(q))
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

    /// Every shape's numbers (in its `Form`) nudged while that lowers the error; opaque shapes only (all the pipeline
    /// writes).
    pub fn polish(&self, shapes: &mut [Shape], forms: &mut [Form]) {
        let (w, h) = (self.width as u32, self.height as u32);
        let fills: Vec<[f64; 3]> = shapes.iter().map(|s| s.fill.map(|c| c as f64 / 255.0)).collect();
        let mut masks: Vec<Mask> = shapes.iter().map(|s| s.mask(w, h)).collect();
        for step in STEPS {
            for _round in 0..ROUNDS {
                let mut improved = false;
                for k in 0..shapes.len() {
                    for group in forms[k].moves(&shapes[k].geometry) {
                        for sign in [1.0, -1.0] {
                            let mut p = forms[k].numbers(&shapes[k].geometry);
                            for &i in &group {
                                p[i] += sign * step;
                            }
                            let (form, geometry) = forms[k].with(&shapes[k].geometry, &p);
                            let moved = Shape {
                                geometry,
                                ..shapes[k].clone()
                            };
                            let mask = moved.mask(w, h);
                            let (a, b) = (bounds(&masks[k]), bounds(&mask));
                            let region = Box_ {
                                x0: a.x0.min(b.x0),
                                y0: a.y0.min(b.y0),
                                x1: a.x1.max(b.x1).min(self.width),
                                y1: a.y1.max(b.y1).min(self.height),
                            };
                            let before = {
                                let all: Vec<&Mask> = masks.iter().collect();
                                self.error(&all, &fills, &region)
                            };
                            let after = {
                                let all: Vec<&Mask> = masks
                                    .iter()
                                    .enumerate()
                                    .map(|(j, m)| if j == k { &mask } else { m })
                                    .collect();
                                self.error(&all, &fills, &region)
                            };
                            if after < before - 1e-9 {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copycat::Document;

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
        let image = drawn(truth);
        let white = vec![[1.0; 3]; 64 * 64];
        Judge {
            width: 64,
            height: 64,
            image: &image,
            background: &white,
        }
        .polish(start, forms);
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
        }];
        let image = drawn(&truth);
        let error = |shapes: &[Shape]| -> f64 {
            drawn(shapes)
                .iter()
                .zip(&image)
                .map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>())
                .sum()
        };
        let before = error(&start);
        polished(&truth, &mut start, &mut forms);
        let Form::Knots { knots: found, .. } = &forms[0] else {
            panic!()
        };
        assert_eq!(found.len(), knots.len());
        assert_eq!(start[0].geometry, spline(found));
        // a knot can slide along the curve for almost nothing, so it is the drawing that is compared, not the knots
        let after = error(&start);
        assert!(after < before / 50.0, "error {before} -> {after}");
    }
}

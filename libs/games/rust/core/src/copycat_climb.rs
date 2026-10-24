//! A `primitive`-style hill climb (after Michael Fogleman's `primitive`), one of Copy Cat's baselines (docs/design/0022,
//! the ladder's first rung). It reproduces the *input*, noise and background included, one shape at a time:
//!
//! 1. the canvas starts as the input's mean colour, drawn as one full-size rect (primitive's background);
//! 2. each new shape is the best of `candidates` random ones, then hill-climbed: a random mutation is kept if it lowers
//!    the squared error, until `patience` mutations in a row fail;
//! 3. a shape's colour is never searched: it is the least-squares colour for its coverage, given the canvas under it.
//!
//! Shapes are opaque triangles (polygons), axis-aligned rects, ellipses and circles: the subset has no rotation.
//! Seeded through `Pcg32`, with Gaussian steps from the Irwin-Hall sum of four uniforms (no transcendentals).

use crate::copycat::{Document, Geometry, Shape};
use crate::pcg::Pcg32;
use crate::raster::{coverage, Mask};

#[derive(Clone, Debug)]
pub struct ClimbConfig {
    /// Shapes after the background.
    pub shapes: usize,
    /// Random shapes tried before climbing from the best of them.
    pub candidates: usize,
    /// Failed mutations in a row that end a climb.
    pub patience: usize,
    pub seed: u64,
}

struct Climber<'a> {
    width: usize,
    height: usize,
    /// The input, straight RGB in 0..1.
    target: &'a [f64],
    /// What the shapes so far draw (opaque, so straight RGB too).
    canvas: Vec<f64>,
    rng: Pcg32,
    /// The scale of a random shape and of a mutation step, in px.
    scale: f64,
}

impl Climber<'_> {
    fn uniform(&mut self) -> f64 {
        self.rng.next_u32() as f64 / 4_294_967_296.0
    }

    fn normal(&mut self) -> f64 {
        // Irwin-Hall: the sum of four uniforms has mean 2 and variance 1/3
        (self.uniform() + self.uniform() + self.uniform() + self.uniform() - 2.0) * 1.732_050_807_568_877_2
    }

    fn point(&mut self) -> (f64, f64) {
        (self.uniform() * self.width as f64, self.uniform() * self.height as f64)
    }

    fn random_shape(&mut self) -> Geometry {
        let s = self.scale;
        match self.rng.bounded(4) {
            0 => {
                let (x, y) = self.point();
                let mut vertex = || {
                    (
                        x + (self.uniform() - 0.5) * 2.0 * s,
                        y + (self.uniform() - 0.5) * 2.0 * s,
                    )
                };
                Geometry::Polygon(vec![(x, y), vertex(), vertex()])
            }
            1 => {
                let (x, y) = self.point();
                Geometry::Rect {
                    x,
                    y,
                    width: 1.0 + self.uniform() * s,
                    height: 1.0 + self.uniform() * s,
                }
            }
            2 => {
                let (cx, cy) = self.point();
                Geometry::Ellipse {
                    cx,
                    cy,
                    rx: 1.0 + self.uniform() * s,
                    ry: 1.0 + self.uniform() * s,
                }
            }
            _ => {
                let (cx, cy) = self.point();
                Geometry::Circle {
                    cx,
                    cy,
                    r: 1.0 + self.uniform() * s,
                }
            }
        }
    }

    fn mutate(&mut self, geometry: &Geometry) -> Geometry {
        let step = self.scale;
        let (w, h) = (self.width as f64, self.height as f64);
        let mut out = geometry.clone();
        match &mut out {
            Geometry::Polygon(points) => {
                let i = self.rng.bounded(points.len() as u32) as usize;
                let (dx, dy) = (self.normal() * step, self.normal() * step);
                points[i].0 = (points[i].0 + dx).clamp(-step, w + step);
                points[i].1 = (points[i].1 + dy).clamp(-step, h + step);
            }
            Geometry::Rect { x, y, width, height } => {
                let (dx, dy) = (self.normal() * step, self.normal() * step);
                if self.rng.bounded(2) == 0 {
                    *x = (*x + dx).clamp(0.0, w - 1.0);
                    *y = (*y + dy).clamp(0.0, h - 1.0);
                } else {
                    *width = (*width + dx).clamp(1.0, w);
                    *height = (*height + dy).clamp(1.0, h);
                }
            }
            Geometry::Ellipse { cx, cy, rx, ry } => {
                let (dx, dy) = (self.normal() * step, self.normal() * step);
                if self.rng.bounded(2) == 0 {
                    *cx = (*cx + dx).clamp(0.0, w);
                    *cy = (*cy + dy).clamp(0.0, h);
                } else {
                    *rx = (*rx + dx).clamp(1.0, w);
                    *ry = (*ry + dy).clamp(1.0, h);
                }
            }
            Geometry::Circle { cx, cy, r } => {
                let (dx, dy) = (self.normal() * step, self.normal() * step);
                if self.rng.bounded(2) == 0 {
                    *cx = (*cx + dx).clamp(0.0, w);
                    *cy = (*cy + dy).clamp(0.0, h);
                } else {
                    *r = (*r + dx).clamp(1.0, w.max(h));
                }
            }
            Geometry::Path(_) => {}
        }
        out
    }

    fn mask(&self, geometry: &Geometry) -> Mask {
        let shape = Shape {
            geometry: geometry.clone(),
            fill: [0, 0, 0],
            opacity: 1.0,
            group: None,
        };
        coverage(&shape.outline(), self.width, self.height)
    }

    /// The least-squares colour through `mask` (quantized to 8 bits, as the SVG will hold it) and the change in
    /// squared error drawing it would make.
    fn score(&self, mask: &Mask) -> ([u8; 3], f64) {
        let (mut num, mut den) = ([0.0; 3], 0.0);
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let c = mask.data[my * mask.width + mx];
                if c == 0.0 {
                    continue;
                }
                let i = ((mask.y0 + my) * self.width + mask.x0 + mx) * 3;
                let (t, cur) = (&self.target[i..i + 3], &self.canvas[i..i + 3]);
                for ((n, t), cur) in num.iter_mut().zip(t).zip(cur) {
                    *n += c * (t - (1.0 - c) * cur);
                }
                den += c * c;
            }
        }
        if den == 0.0 {
            return ([0, 0, 0], 0.0);
        }
        let fill = num.map(|n| ((n / den).clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8);
        let rgb = fill.map(|v| v as f64 / 255.0);
        let mut delta = 0.0;
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let c = mask.data[my * mask.width + mx];
                if c == 0.0 {
                    continue;
                }
                let i = ((mask.y0 + my) * self.width + mask.x0 + mx) * 3;
                let (t, cur) = (&self.target[i..i + 3], &self.canvas[i..i + 3]);
                for ((v, t), cur) in rgb.iter().zip(t).zip(cur) {
                    let before = cur - t;
                    let after = c * v + (1.0 - c) * cur - t;
                    delta += after * after - before * before;
                }
            }
        }
        (fill, delta)
    }

    fn draw(&mut self, mask: &Mask, fill: [u8; 3]) {
        let rgb = fill.map(|v| v as f64 / 255.0);
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let c = mask.data[my * mask.width + mx];
                let i = ((mask.y0 + my) * self.width + mask.x0 + mx) * 3;
                for (cur, v) in self.canvas[i..i + 3].iter_mut().zip(rgb) {
                    *cur = c * v + (1.0 - c) * *cur;
                }
            }
        }
    }

    /// The next shape: the best random candidate, climbed. None if nothing it could draw would help.
    fn next_shape(&mut self, candidates: usize, patience: usize) -> Option<Shape> {
        let mut best: Option<(Geometry, Mask, [u8; 3], f64)> = None;
        for _ in 0..candidates.max(1) {
            let geometry = self.random_shape();
            let mask = self.mask(&geometry);
            let (fill, delta) = self.score(&mask);
            if best.as_ref().is_none_or(|b| delta < b.3) {
                best = Some((geometry, mask, fill, delta));
            }
        }
        let (mut geometry, mut mask, mut fill, mut delta) = best?;
        let mut failures = 0;
        while failures < patience {
            let candidate = self.mutate(&geometry);
            let candidate_mask = self.mask(&candidate);
            let (candidate_fill, candidate_delta) = self.score(&candidate_mask);
            if candidate_delta < delta {
                (geometry, mask, fill, delta) = (candidate, candidate_mask, candidate_fill, candidate_delta);
                failures = 0;
            } else {
                failures += 1;
            }
        }
        if delta >= 0.0 {
            return None;
        }
        self.draw(&mask, fill);
        Some(Shape {
            geometry,
            fill,
            opacity: 1.0,
            group: None,
        })
    }
}

/// The hill climb's SVG for an image: `rgba` is 8-bit RGBA, `width x height`, alpha ignored (inputs are opaque).
pub fn hill_climb(rgba: &[u8], width: u32, height: u32, config: &ClimbConfig) -> Document {
    let (w, h) = (width as usize, height as usize);
    assert_eq!(rgba.len(), w * h * 4, "rgba is not {width}x{height}");
    let target: Vec<f64> = rgba
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]].map(|v| v as f64 / 255.0))
        .collect();
    let pixels = (w * h) as f64;
    let mut mean = [0.0; 3];
    for px in target.chunks_exact(3) {
        for k in 0..3 {
            mean[k] += px[k] / pixels;
        }
    }
    let background = mean.map(|m| (m * 255.0 + 0.5).floor() as u8);
    let canvas = target
        .chunks_exact(3)
        .flat_map(|_| background.map(|v| v as f64 / 255.0))
        .collect();
    let mut climber = Climber {
        width: w,
        height: h,
        target: &target,
        canvas,
        rng: Pcg32::new(config.seed),
        scale: (w.max(h) as f64 / 8.0).max(2.0),
    };
    let mut shapes = vec![Shape {
        geometry: Geometry::Rect {
            x: 0.0,
            y: 0.0,
            width: width as f64,
            height: height as f64,
        },
        fill: background,
        opacity: 1.0,
        group: None,
    }];
    for _ in 0..config.shapes {
        match climber.next_shape(config.candidates, config.patience) {
            Some(shape) => shapes.push(shape),
            None => break,
        }
    }
    Document {
        width,
        height,
        groups: Vec::new(),
        shapes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn climbs_toward_the_input_and_is_seeded() {
        let target = Document::parse(
            "<svg viewBox=\"0 0 32 32\"><rect width=\"32\" height=\"32\" fill=\"#ffffff\"/>\
             <circle cx=\"12\" cy=\"14\" r=\"8\" fill=\"#cc2020\"/>\
             <rect x=\"16\" y=\"6\" width=\"12\" height=\"20\" fill=\"#2040cc\"/></svg>",
        )
        .unwrap();
        let rgba = target.render().to_rgba8();
        let config = ClimbConfig {
            shapes: 12,
            candidates: 40,
            patience: 40,
            seed: 3,
        };
        let error = |doc: &Document| {
            let ours = doc.render().to_rgba8();
            ours.iter()
                .zip(&rgba)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
        };
        let background_only = hill_climb(
            &rgba,
            32,
            32,
            &ClimbConfig {
                shapes: 0,
                ..config.clone()
            },
        );
        let climbed = hill_climb(&rgba, 32, 32, &config);
        assert!(climbed.shapes.len() > 3);
        assert!(error(&climbed) < 0.25 * error(&background_only));
        assert_eq!(climbed, hill_climb(&rgba, 32, 32, &config));
        // the output is in the subset: its canonical form parses back
        assert_eq!(
            Document::parse(&climbed.canonical()).unwrap().shapes.len(),
            climbed.shapes.len()
        );
    }
}

//! The rasterizer (docs/design/0022 Decision 6): filled outlines to anti-aliased RGBA, written once for the Copy Cat
//! protocol, its entrants and the browser, so a score is the same number everywhere.
//!
//! - **Outlines** are closed polylines (subpaths); curves are flattened to lines first (`PathBuilder`), within
//!   `TOLERANCE` px of the true curve.
//! - **Coverage** is analytic, not sampled: every line deposits the exact signed area it sweeps into an accumulation
//!   buffer, one row at a time, and a running sum along each row turns that into each pixel's coverage (the
//!   accumulation scheme of font-rs and stb_truetype). The sum is the area-weighted winding number; coverage is its
//!   magnitude clamped to 1, which is the non-zero rule wherever a pixel's winding keeps one sign (overlapping
//!   subpaths that wind the same way, holes that wind the other way).
//! - **Compositing** is `over`, in premultiplied RGBA with channels in 0..1 (sRGB-encoded, as browsers composite).
//!
//! Arithmetic is `+ - * /`, `floor`/`ceil` and `sqrt`, all exactly rounded under IEEE 754, so a render is bit for bit
//! the same natively and in WASM.

pub type Point = (f64, f64);

/// A filled shape's outline: closed polylines, filled by the non-zero rule.
pub type Outline = Vec<Vec<Point>>;

/// Maximum distance between a curve and the lines it is flattened to, in px.
pub const TOLERANCE: f64 = 0.05;
/// No curve segment is cut into more lines than this, however large it is.
const MAX_PIECES: usize = 256;

/// Builds an `Outline` from path commands, flattening curves as they come.
#[derive(Default)]
pub struct PathBuilder {
    outline: Outline,
    current: Vec<Point>,
}

impl PathBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn last(&self) -> Point {
        *self.current.last().unwrap_or(&(0.0, 0.0))
    }

    pub fn move_to(&mut self, p: Point) {
        self.flush();
        self.current.push(p);
    }

    pub fn line_to(&mut self, p: Point) {
        if self.current.is_empty() {
            self.current.push((0.0, 0.0));
        }
        self.current.push(p);
    }

    pub fn quad_to(&mut self, c: Point, p: Point) {
        let p0 = self.last();
        let dd = norm(p0.0 - 2.0 * c.0 + p.0, p0.1 - 2.0 * c.1 + p.1);
        // uniform steps: the deviation is at most |B''| / (8 n^2), and |B''| = 2 |p0 - 2c + p|
        let n = pieces(0.25 * dd);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let u = 1.0 - t;
            let (a, b, d) = (u * u, 2.0 * u * t, t * t);
            self.line_to((a * p0.0 + b * c.0 + d * p.0, a * p0.1 + b * c.1 + d * p.1));
        }
    }

    pub fn cubic_to(&mut self, c1: Point, c2: Point, p: Point) {
        let p0 = self.last();
        let dd1 = norm(p0.0 - 2.0 * c1.0 + c2.0, p0.1 - 2.0 * c1.1 + c2.1);
        let dd2 = norm(c1.0 - 2.0 * c2.0 + p.0, c1.1 - 2.0 * c2.1 + p.1);
        // |B''| <= 6 max(dd1, dd2)
        let n = pieces(0.75 * dd1.max(dd2));
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let u = 1.0 - t;
            let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            self.line_to((
                a * p0.0 + b * c1.0 + c * c2.0 + d * p.0,
                a * p0.1 + b * c1.1 + c * c2.1 + d * p.1,
            ));
        }
    }

    /// Ends the subpath; the next command starts from its first point, as in SVG.
    pub fn close(&mut self) {
        let start = self.current.first().copied();
        self.flush();
        if let Some(start) = start {
            self.current.push(start);
        }
    }

    fn flush(&mut self) {
        let points = std::mem::take(&mut self.current);
        if points.len() > 2 {
            self.outline.push(points);
        }
    }

    pub fn finish(mut self) -> Outline {
        self.flush();
        self.outline
    }
}

fn norm(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

/// Lines needed for a curve whose deviation bound is `k / n^2`.
fn pieces(k: f64) -> usize {
    let n = (k / TOLERANCE).sqrt().ceil();
    if n.is_nan() || n < 1.0 {
        1
    } else if n > MAX_PIECES as f64 {
        MAX_PIECES
    } else {
        n as usize
    }
}

/// The ellipse centred on `(cx, cy)` as four cubic Béziers (the standard 0.5523 handles: within 0.03% of the radius).
pub fn ellipse(cx: f64, cy: f64, rx: f64, ry: f64) -> Outline {
    const K: f64 = 0.552_284_749_830_793_4; // 4 (sqrt 2 - 1) / 3
    let (kx, ky) = (K * rx, K * ry);
    let mut path = PathBuilder::new();
    path.move_to((cx + rx, cy));
    path.cubic_to((cx + rx, cy + ky), (cx + kx, cy + ry), (cx, cy + ry));
    path.cubic_to((cx - kx, cy + ry), (cx - rx, cy + ky), (cx - rx, cy));
    path.cubic_to((cx - rx, cy - ky), (cx - kx, cy - ry), (cx, cy - ry));
    path.cubic_to((cx + kx, cy - ry), (cx + rx, cy - ky), (cx + rx, cy));
    path.finish()
}

/// Coverage of one outline over the part of a `width x height` image it touches.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub x0: usize,
    pub y0: usize,
    pub width: usize,
    pub height: usize,
    /// Row-major coverage in 0..1.
    pub data: Vec<f64>,
}

impl Mask {
    pub fn empty() -> Self {
        Mask {
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            data: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Coverage at image pixel (x, y), 0 outside the mask.
    pub fn at(&self, x: usize, y: usize) -> f64 {
        if x < self.x0 || y < self.y0 || x >= self.x0 + self.width || y >= self.y0 + self.height {
            0.0
        } else {
            self.data[(y - self.y0) * self.width + (x - self.x0)]
        }
    }

    pub fn area(&self) -> f64 {
        self.data.iter().sum()
    }
}

/// The analytic coverage of `outline` (non-zero rule) on a `width x height` pixel grid, cropped to its bounding box.
pub fn coverage(outline: &Outline, width: usize, height: usize) -> Mask {
    let (mut min_x, mut min_y, mut max_x, mut max_y) =
        (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in outline.iter().flatten() {
        if !(x.is_finite() && y.is_finite()) {
            return Mask::empty();
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let x0 = min_x.max(0.0).floor();
    let y0 = min_y.max(0.0).floor();
    let x1 = max_x.min(width as f64).ceil();
    let y1 = max_y.min(height as f64).ceil();
    if !(x1 > x0 && y1 > y0) {
        return Mask::empty();
    }
    let (x0, y0) = (x0 as usize, y0 as usize);
    let (mw, mh) = (x1 as usize - x0, y1 as usize - y0);
    let mut acc = Accumulator::new(mw, mh);
    for sub in outline {
        for i in 0..sub.len() {
            let a = sub[i];
            let b = sub[(i + 1) % sub.len()]; // every subpath is closed
            acc.clipped_line((a.0 - x0 as f64, a.1 - y0 as f64), (b.0 - x0 as f64, b.1 - y0 as f64));
        }
    }
    Mask {
        x0,
        y0,
        width: mw,
        height: mh,
        data: acc.coverage(),
    }
}

/// Signed area per cell, two spare columns wide: a line at the right edge (x = width) deposits there, out of sight.
struct Accumulator {
    width: usize,
    height: usize,
    stride: usize,
    cells: Vec<f64>,
}

impl Accumulator {
    fn new(width: usize, height: usize) -> Self {
        Accumulator {
            width,
            height,
            stride: width + 2,
            cells: vec![0.0; (width + 2) * height],
        }
    }

    /// Clips a line to 0 <= x <= width. A part outside is replaced by its projection onto the edge -- a vertical line
    /// with the same rows and direction -- which changes the winding of nothing inside.
    fn clipped_line(&mut self, a: Point, b: Point) {
        let w = self.width as f64;
        let inside = |x: f64| (0.0..=w).contains(&x);
        if inside(a.0) && inside(b.0) {
            self.line(a, b);
            return;
        }
        let mut cuts = [0.0, 1.0, 1.0, 1.0];
        let mut n = 1;
        if a.0 != b.0 {
            for edge in [0.0, w] {
                let t = (edge - a.0) / (b.0 - a.0);
                if t > 0.0 && t < 1.0 {
                    cuts[n] = t;
                    n += 1;
                }
            }
        }
        cuts[n] = 1.0;
        cuts[1..n].sort_by(|p, q| p.total_cmp(q));
        let at = |t: f64| {
            if t == 0.0 {
                a
            } else if t == 1.0 {
                b
            } else {
                (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
            }
        };
        for i in 0..n {
            let (p, q) = (at(cuts[i]), at(cuts[i + 1]));
            self.line((p.0.clamp(0.0, w), p.1), (q.0.clamp(0.0, w), q.1));
        }
    }

    /// Deposits the area a line sweeps (0 <= x <= width; any y), row by row.
    fn line(&mut self, p0: Point, p1: Point) {
        if p0.1 == p1.1 {
            return;
        }
        let (dir, p0, p1) = if p0.1 < p1.1 { (1.0, p0, p1) } else { (-1.0, p1, p0) };
        let h = self.height as f64;
        if p1.1 <= 0.0 || p0.1 >= h {
            return;
        }
        let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
        let y_start = p0.1.max(0.0);
        let mut x = p0.0 + (y_start - p0.1) * dxdy;
        let row_end = p1.1.min(h).ceil() as usize;
        for row in (y_start.floor() as usize)..row_end {
            let top = (row as f64).max(p0.1);
            let bottom = ((row + 1) as f64).min(p1.1);
            let dy = bottom - top;
            let x_next = if bottom == p1.1 { p1.0 } else { x + dxdy * dy };
            let d = dy * dir;
            let line = row * self.stride;
            let (xa, xb) = if x < x_next { (x, x_next) } else { (x_next, x) };
            let xa_floor = xa.floor();
            let xa_cell = xa_floor as usize;
            let xb_ceil = xb.ceil();
            let xb_cell = xb_ceil as usize;
            if xb_cell <= xa_cell + 1 {
                // within one column: the trapezoid's area right of the line, split between it and the next cell
                let mid = 0.5 * (x + x_next) - xa_floor;
                self.cells[line + xa_cell] += d - d * mid;
                self.cells[line + xa_cell + 1] += d * mid;
            } else {
                let s = 1.0 / (xb - xa);
                let fa = xa - xa_floor;
                let a0 = 0.5 * s * (1.0 - fa) * (1.0 - fa);
                let fb = xb - xb_ceil + 1.0;
                let am = 0.5 * s * fb * fb;
                self.cells[line + xa_cell] += d * a0;
                if xb_cell == xa_cell + 2 {
                    self.cells[line + xa_cell + 1] += d * (1.0 - a0 - am);
                } else {
                    let a1 = s * (1.5 - fa);
                    self.cells[line + xa_cell + 1] += d * (a1 - a0);
                    for cell in (xa_cell + 2)..(xb_cell - 1) {
                        self.cells[line + cell] += d * s;
                    }
                    let a2 = a1 + (xb_cell - xa_cell - 3) as f64 * s;
                    self.cells[line + xb_cell - 1] += d * (1.0 - a2 - am);
                }
                self.cells[line + xb_cell] += d * am;
            }
            x = x_next;
        }
    }

    fn coverage(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.width * self.height);
        for row in 0..self.height {
            let mut winding = 0.0;
            for cell in &self.cells[row * self.stride..row * self.stride + self.width] {
                winding += cell;
                out.push(winding.abs().min(1.0));
            }
        }
        out
    }
}

/// A premultiplied RGBA image, channels in 0..1, starting transparent.
#[derive(Clone, Debug, PartialEq)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    /// Row-major, 4 channels per pixel: r, g, b premultiplied by a, then a.
    pub data: Vec<f64>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Canvas {
            width,
            height,
            data: vec![0.0; width * height * 4],
        }
    }

    /// `rgb` (0..1, straight) at `opacity`, through `mask`, composited over what is there.
    pub fn fill(&mut self, mask: &Mask, rgb: [f64; 3], opacity: f64) {
        for my in 0..mask.height {
            let row = (mask.y0 + my) * self.width + mask.x0;
            for mx in 0..mask.width {
                let a = mask.data[my * mask.width + mx] * opacity;
                if a == 0.0 {
                    continue;
                }
                let px = &mut self.data[(row + mx) * 4..(row + mx) * 4 + 4];
                let keep = 1.0 - a;
                px[0] = rgb[0] * a + px[0] * keep;
                px[1] = rgb[1] * a + px[1] * keep;
                px[2] = rgb[2] * a + px[2] * keep;
                px[3] = a + px[3] * keep;
            }
        }
    }

    /// Straight-alpha RGBA, 8 bits per channel (what a PNG holds).
    pub fn to_rgba8(&self) -> Vec<u8> {
        let byte = |v: f64| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8;
        let mut out = Vec::with_capacity(self.data.len());
        for px in self.data.chunks_exact(4) {
            let a = px[3];
            if a <= 0.0 {
                out.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                out.extend_from_slice(&[byte(px[0] / a), byte(px[1] / a), byte(px[2] / a), byte(a)]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcg::Pcg32;

    /// The oracle: the non-zero rule at n x n sample points per pixel, by explicit winding numbers.
    fn supersampled(outline: &Outline, width: usize, height: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; width * height];
        for py in 0..height {
            for px in 0..width {
                let mut inside = 0;
                for sy in 0..n {
                    for sx in 0..n {
                        let x = px as f64 + (sx as f64 + 0.5) / n as f64;
                        let y = py as f64 + (sy as f64 + 0.5) / n as f64;
                        if winding(outline, x, y) != 0 {
                            inside += 1;
                        }
                    }
                }
                out[py * width + px] = inside as f64 / (n * n) as f64;
            }
        }
        out
    }

    fn winding(outline: &Outline, x: f64, y: f64) -> i32 {
        let mut w = 0;
        for sub in outline {
            for i in 0..sub.len() {
                let (a, b) = (sub[i], sub[(i + 1) % sub.len()]);
                if (a.1 <= y) != (b.1 <= y) {
                    let t = (y - a.1) / (b.1 - a.1);
                    if a.0 + t * (b.0 - a.0) > x {
                        w += if b.1 > a.1 { 1 } else { -1 };
                    }
                }
            }
        }
        w
    }

    fn full(mask: &Mask, width: usize, height: usize) -> Vec<f64> {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| mask.at(x, y))
            .collect()
    }

    fn assert_close_to_oracle(outline: &Outline, width: usize, height: usize) {
        let ours = full(&coverage(outline, width, height), width, height);
        let oracle = supersampled(outline, width, height, 32);
        let mut worst = 0.0f64;
        let mut total = 0.0;
        for (a, b) in ours.iter().zip(&oracle) {
            worst = worst.max((a - b).abs());
            total += (a - b).abs();
        }
        // 32 x 32 samples resolve coverage to ~1/32 at an edge; the mean error is far smaller
        assert!(worst < 0.04, "worst pixel off by {worst}");
        let mean = total / ours.len() as f64;
        assert!(mean < 0.002, "mean error {mean}");
    }

    /// A simple (non-self-intersecting) polygon: random points sorted by angle around their centre.
    fn random_polygon(rng: &mut Pcg32, vertices: usize, span: f64) -> Vec<Point> {
        let mut uniform = || rng.next_u32() as f64 / 4_294_967_296.0;
        let mut points: Vec<Point> = (0..vertices)
            .map(|_| (uniform() * span - 4.0, uniform() * span - 4.0))
            .collect();
        let n = vertices as f64;
        let centre = (
            points.iter().map(|p| p.0).sum::<f64>() / n,
            points.iter().map(|p| p.1).sum::<f64>() / n,
        );
        points.sort_by(|a, b| {
            let angle = |p: &Point| (p.1 - centre.1).atan2(p.0 - centre.0);
            angle(a).total_cmp(&angle(b))
        });
        points
    }

    #[test]
    fn an_axis_aligned_square_is_exact() {
        let outline = vec![vec![(2.25, 3.5), (6.75, 3.5), (6.75, 8.0), (2.25, 8.0)]];
        let mask = coverage(&outline, 10, 10);
        assert!((mask.area() - 4.5 * 4.5).abs() < 1e-12);
        assert!((mask.at(2, 4) - 0.75).abs() < 1e-12);
        assert!((mask.at(4, 3) - 0.5).abs() < 1e-12);
        assert!((mask.at(4, 5) - 1.0).abs() < 1e-12);
        assert_eq!(mask.at(0, 0), 0.0);
    }

    #[test]
    fn either_winding_direction_fills_the_same() {
        let cw = vec![vec![(1.3, 1.1), (8.2, 2.7), (4.4, 9.6)]];
        let ccw = vec![cw[0].iter().rev().copied().collect()];
        assert_eq!(coverage(&cw, 10, 10), coverage(&ccw, 10, 10));
    }

    #[test]
    fn coverage_matches_a_supersampled_oracle() {
        let mut rng = Pcg32::new(22);
        for vertices in [3, 4, 5, 7] {
            for _ in 0..6 {
                let outline = vec![random_polygon(&mut rng, vertices, 24.0)];
                assert_close_to_oracle(&outline, 16, 16);
            }
        }
    }

    #[test]
    fn a_self_intersecting_outline_is_off_only_where_it_crosses() {
        // the one approximation (module docs): a pixel holding windings of both signs. A bow tie's two lobes wind
        // opposite ways, so the pixel at the crossing is wrong and every other one is right.
        let bow_tie = vec![vec![(1.0, 1.0), (11.0, 11.0), (11.0, 1.0), (1.0, 11.0)]];
        let ours = full(&coverage(&bow_tie, 12, 12), 12, 12);
        let oracle = supersampled(&bow_tie, 12, 12, 32);
        let off: Vec<usize> = (0..ours.len())
            .filter(|&i| (ours[i] - oracle[i]).abs() > 0.04)
            .collect();
        assert!(
            off.iter()
                .all(|&i| [5, 6].contains(&(i % 12)) && [5, 6].contains(&(i / 12))),
            "{off:?}"
        );
    }

    #[test]
    fn curves_and_clipping_match_the_oracle() {
        // partly off every edge of the image
        assert_close_to_oracle(&ellipse(8.0, 7.5, 11.3, 5.2), 16, 16);
        assert_close_to_oracle(&ellipse(3.1, 12.7, 6.0, 6.0), 16, 16);
        let mut path = PathBuilder::new();
        path.move_to((-3.0, 2.0));
        path.cubic_to((10.0, -6.0), (25.0, 9.0), (12.0, 18.0));
        path.quad_to((2.0, 20.0), (-3.0, 2.0));
        assert_close_to_oracle(&path.finish(), 16, 16);
    }

    #[test]
    fn same_direction_overlaps_count_once_and_holes_are_empty() {
        let square = |x: f64, y: f64, s: f64| vec![(x, y), (x + s, y), (x + s, y + s), (x, y + s)];
        let overlapping = vec![square(1.0, 1.0, 6.0), square(4.0, 4.0, 6.0)];
        assert_close_to_oracle(&overlapping, 12, 12);
        let mut hole = square(2.0, 2.0, 4.0);
        hole.reverse();
        let ring = vec![square(0.5, 0.5, 9.0), hole];
        let mask = coverage(&ring, 12, 12);
        assert_eq!(mask.at(3, 3), 0.0);
        assert_close_to_oracle(&ring, 12, 12);
    }

    #[test]
    fn flattening_stays_within_tolerance() {
        // a circle of radius 40 through the Bézier approximation: every flattened point within tolerance of the circle
        let outline = ellipse(50.0, 50.0, 40.0, 40.0);
        for &(x, y) in &outline[0] {
            let r = ((x - 50.0) * (x - 50.0) + (y - 50.0) * (y - 50.0)).sqrt();
            assert!((r - 40.0).abs() < 40.0 * 3e-4 + TOLERANCE, "radius {r}");
        }
    }

    #[test]
    fn over_compositing_and_rgba8() {
        let mut canvas = Canvas::new(2, 1);
        let all = Mask {
            x0: 0,
            y0: 0,
            width: 2,
            height: 1,
            data: vec![1.0, 0.5],
        };
        canvas.fill(&all, [1.0, 0.0, 0.0], 1.0);
        canvas.fill(&all, [0.0, 0.0, 1.0], 0.5);
        // pixel 0: blue at 0.5 over opaque red
        assert_eq!(&canvas.data[0..4], &[0.5, 0.0, 0.5, 1.0]);
        // pixel 1: red at 0.5, then blue at 0.25 over it
        assert_eq!(&canvas.data[4..8], &[0.375, 0.0, 0.25, 0.625]);
        assert_eq!(&canvas.to_rgba8()[0..4], &[128, 0, 128, 255]);
    }
}

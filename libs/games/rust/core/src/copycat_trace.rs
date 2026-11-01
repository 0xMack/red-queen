//! Copy Cat's structured pipeline (docs/design/0022 Decisions 4-5, the ladder's step 2): an image in, a clean, layered,
//! transparent SVG out, every choice made by description length rather than by a threshold where one can be had.
//!
//! 1. **Noise**: σ, a robust (median) estimate from neighbouring pixels' OKLab differences -- edges are the tail.
//! 2. **Background**: fitted to the image border in OKLab, constant or a linear gradient (whichever is shorter to
//!    describe), its outliers dropped twice -- or, when that fit misses the corners (a tile filling most of the
//!    border), started again from the corners. It is a cluster of its own whose centre varies across the image, and
//!    what it explains is left transparent.
//! 3. **Palette**: k-means in OKLab from `colors` k-means++ seeds (`Pcg32`), then merges, closest pair first, while a
//!    merge is shorter -- a palette entry (24 bits) and the boundary between them saved, against the residual it adds
//!    (`ΔSSE / 2σ² ln 2` bits) -- or the two are within `merge_delta` of each other (a part's own shading, which a
//!    flat fill can't say anyway) and aren't two flat colours (`two_flat_colours`), or a cluster is within
//!    `background_delta` of the background (faint clutter).
//! 4. **Labels**: a 3 x 3 majority filter; then components too small (`min_area`) or too thin (an anti-aliased or
//!    blurred edge's in-between colour) are dissolved into whatever surrounds them, but never the background where it
//!    reaches the image's edge. A background component enclosed by parts whose colour is not quite the background's (a
//!    white letter, an off-white background) is a colour of its own; so is a light part the background took for faint
//!    clutter, if it stands out from the background and touches the rest of the subject (`restore`).
//! 5. **Parts**: every connected component but the background is a part, filled with its label's interior mean (one
//!    palette entry per label); two of one colour join when one shape says them more briefly (a part an occluder cut in two).
//! 6. **Order**: for each pair of touching parts, the one whose shape completed under the other covers more of it (as a
//!    fraction) goes below (`copycat_fit`'s free pixels); a topological sort, largest first, breaks cycles and ties.
//! 7. **Shapes**: each part fitted with everything above it free, so a hidden part completes itself; cut in two
//!    between notches of its outline while two shapes say it more briefly (two overlapping parts of one colour); then
//!    each enclosed gap in a shape is cut out of it as a hole, while that is shorter (`Target::with_holes`); one `<g>`
//!    per shape.

use std::collections::{BTreeMap, VecDeque};

use crate::copycat::{Document, Group, Shape};
use crate::copycat_fit::{inside_polygon, largest_contour, notches, solve, Costs, Fit, Target};
use crate::pcg::Pcg32;

#[derive(Clone, Debug)]
pub struct TraceConfig {
    /// k-means seeds before merging.
    pub colors: usize,
    /// OKLab distance under which two clusters always merge.
    pub merge_delta: f64,
    /// OKLab distance from the background under which a cluster joins it (a generator's faint clutter).
    pub background_delta: f64,
    /// Bits per wrongly covered pixel when fitting a shape.
    pub pixel_bits: f64,
    /// Bits per pixel of boundary two clusters share (saved by merging them).
    pub edge_bits: f64,
    /// Components smaller than this many pixels are dissolved.
    pub min_area: usize,
    /// Bits a move of a polygon's or path's point must save, per 0.001 of noise σ above 8-bit quantization's: an
    /// outline refined against a clean image follows it closely, against a noisy one it doesn't chase the noise.
    pub outline_gain: f64,
    pub seed: u64,
}

impl Default for TraceConfig {
    fn default() -> Self {
        TraceConfig {
            colors: 16,
            merge_delta: 0.08,
            background_delta: 0.1,
            pixel_bits: 12.0,
            edge_bits: 2.0,
            min_area: 30,
            outline_gain: 12.0,
            seed: 0,
        }
    }
}

type Lab = [f64; 3];

fn distance2(a: &Lab, b: &Lab) -> f64 {
    (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])
}

/// sRGB (8-bit) to OKLab (Björn Ottosson, 2020).
pub fn oklab(rgb: [u8; 3]) -> Lab {
    let linear = rgb.map(|c| {
        let c = c as f64 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            libm::pow((c + 0.055) / 1.055, 2.4)
        }
    });
    let [r, g, b] = linear;
    let l = libm::cbrt(0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b);
    let m = libm::cbrt(0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b);
    let s = libm::cbrt(0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b);
    [
        0.210_454_255_3 * l + 0.793_617_785_0 * m - 0.004_072_046_8 * s,
        1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s,
        0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s,
    ]
}

/// The image, in OKLab and as 8-bit RGB (alpha ignored: the input is opaque).
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<[u8; 3]>,
    pub lab: Vec<Lab>,
}

impl Image {
    pub fn from_rgba8(rgba: &[u8], width: usize, height: usize) -> Result<Image, String> {
        if width == 0 || height == 0 || rgba.len() != width * height * 4 {
            return Err(format!("{} bytes is not a {width}x{height} RGBA image", rgba.len()));
        }
        let rgb: Vec<[u8; 3]> = rgba.as_chunks::<4>().0.iter().map(|p| [p[0], p[1], p[2]]).collect();
        let mut table: BTreeMap<[u8; 3], Lab> = BTreeMap::new();
        let lab = rgb
            .iter()
            .map(|&c| *table.entry(c).or_insert_with(|| oklab(c)))
            .collect();
        Ok(Image {
            width,
            height,
            rgb,
            lab,
        })
    }

    fn neighbours(&self, i: usize) -> impl Iterator<Item = usize> {
        let (w, h) = (self.width, self.height);
        let (x, y) = (i % w, i / w);
        [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ]
        .into_iter()
        .flatten()
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    *values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
}

/// The noise's standard deviation per OKLab channel: the median neighbouring difference is that of a chi variable with
/// 3 degrees of freedom (median 1.5382) scaled by σ√2. Floored at 8-bit quantization's.
pub fn noise_sigma(image: &Image) -> f64 {
    let mut diffs = Vec::with_capacity(2 * image.lab.len());
    for i in 0..image.lab.len() {
        let (x, y) = (i % image.width, i / image.width);
        if x + 1 < image.width {
            diffs.push(distance2(&image.lab[i], &image.lab[i + 1]).sqrt());
        }
        if y + 1 < image.height {
            diffs.push(distance2(&image.lab[i], &image.lab[i + image.width]).sqrt());
        }
    }
    (median(diffs) / (1.538_2 * std::f64::consts::SQRT_2)).max(SIGMA_FLOOR)
}

/// 8-bit quantization's noise σ in OKLab, about: no image's estimate goes lower.
const SIGMA_FLOOR: f64 = 0.002;

/// The background: per OKLab channel, `c + gx x + gy y` (x and y in 0..1); `gx`, `gy` are zero for a constant one.
#[derive(Clone, Copy, Debug)]
pub struct Background {
    pub coefficients: [[f64; 3]; 3],
    pub gradient: bool,
    /// How far (OKLab) a border pixel could stray and still be counted background: the robust fit's cut.
    pub cut: f64,
}

impl Background {
    pub fn at(&self, x: f64, y: f64) -> Lab {
        let c = &self.coefficients;
        [0, 1, 2].map(|k| c[k][0] + c[k][1] * x + c[k][2] * y)
    }

    pub fn at_pixel(&self, image: &Image, i: usize) -> Lab {
        let (x, y) = (i % image.width, i / image.width);
        self.at(
            x as f64 / (image.width.max(2) - 1) as f64,
            y as f64 / (image.height.max(2) - 1) as f64,
        )
    }

    /// Fitted to a 2-pixel border; outliers past 3 robust deviations dropped and refitted, twice. Pixels alone can't
    /// tell a tile filling most of the border from a background, so a prior decides: the background reaches the
    /// corners. When the border's fit leaves a corner unexplained, the fit is started again from the corners instead,
    /// and kept if it explains three of the four. A gradient the border doesn't bear out (`gradient_holds`) gives way
    /// to one colour, if two corners and much of the border share it.
    pub fn fit(image: &Image, sigma: f64) -> Background {
        let (w, h) = (image.width, image.height);
        let border: Vec<usize> = (0..w * h)
            .filter(|&i| {
                let (x, y) = (i % w, i / w);
                x < 2 || y < 2 || x + 2 >= w || y + 2 >= h
            })
            .collect();
        let corners: Vec<Vec<usize>> = [(0, 0), (w - CORNER, 0), (0, h - CORNER), (w - CORNER, h - CORNER)]
            .iter()
            .map(|&(x0, y0)| {
                (0..CORNER * CORNER)
                    .map(|k| (y0 + k / CORNER) * w + x0 + k % CORNER)
                    .collect()
            })
            .collect();
        let from_border = Background::robust(image, &border, &border, sigma);
        // started from the three corners that agree best: one may be a speck of clutter or a tile's blurred edge
        let mean = |c: &Vec<usize>| [0, 1, 2].map(|k| c.iter().map(|&i| image.lab[i][k]).sum::<f64>() / c.len() as f64);
        let means: Vec<Lab> = corners.iter().map(mean).collect();
        let middle = [0, 1, 2].map(|k| median(means.iter().map(|m| m[k]).collect()));
        let odd = (0..corners.len())
            .max_by(|&a, &b| distance2(&means[a], &middle).total_cmp(&distance2(&means[b], &middle)))
            .unwrap();
        let start: Vec<usize> = (0..corners.len())
            .filter(|&k| k != odd)
            .flat_map(|k| corners[k].clone())
            .collect();
        let from_corners = Background::robust(image, &border, &start, sigma);
        // judged by how much the background varies about a gradient through the corners -- tight even under a
        // gradient background -- and the corner fit may miss one corner (a speck of clutter, a tile's blurred edge)
        let cut = if from_corners.cut.is_finite() {
            from_corners.cut
        } else {
            6.0 * sigma
        };
        let fit = if from_border.explained(image, &corners, sigma, cut) == corners.len() {
            from_border.choose(image, sigma)
        } else if from_corners.explained(image, &corners, sigma, cut) + 1 >= corners.len() {
            from_corners.choose(image, sigma)
        } else {
            from_border.choose(image, sigma)
        };
        if !fit.gradient || fit.gradient_holds(image, &border) {
            return fit;
        }
        // Two colours, not a gradient: a subject on one edge, the background on the other. That needs two corners of
        // one colour that much of the border shares, a pixel being theirs only if it is nearer them than the other two
        // corners; a tile on a gradient background shows it only at its corners, whose two ends are nearly one colour.
        // Corners in the order top-left, top-right, bottom-left, bottom-right; the top first, since a subject more
        // often rests on the bottom edge than hangs from the top.
        [(0, 1, 2, 3), (0, 2, 1, 3), (1, 3, 0, 2), (2, 3, 0, 1)]
            .iter()
            .filter(|&&(a, b, ..)| distance2(&means[a], &means[b]).sqrt() <= fit.cut)
            .map(|&(a, b, c, d)| {
                let start: Vec<usize> = corners[a].iter().chain(&corners[b]).copied().collect();
                let mid = |p: &Lab, q: &Lab| [0, 1, 2].map(|k| (p[k] + q[k]) / 2.0);
                let span = distance2(&mid(&means[a], &means[b]), &mid(&means[c], &means[d])).sqrt();
                let flat = Background::flat_from(image, &border, &start, sigma);
                Background {
                    cut: flat.cut.min(span / 2.0),
                    ..flat
                }
            })
            .find(|flat| {
                let shared = border
                    .iter()
                    .filter(|&&i| distance2(&image.lab[i], &flat.at_pixel(image, i)).sqrt() <= flat.cut)
                    .count();
                shared as f64 >= BORDER_SHARE * border.len() as f64
            })
            .unwrap_or(fit)
    }

    /// Whether the border explains a gradient along its length: at least `GRADIENT_MIDDLE` of the border pixels in the
    /// middle half of it, along the axis it changes most on, are within `cut`. A gradient between two colours -- the
    /// background at the top, the sea at the bottom -- runs through the corners and nothing else.
    fn gradient_holds(&self, image: &Image, border: &[usize]) -> bool {
        let c = &self.coefficients;
        let slope = |j: usize| [0, 1, 2].map(|k| c[k][j] * c[k][j]).iter().sum::<f64>();
        let along_x = slope(1) >= slope(2);
        let (mut middle, mut explained) = (0usize, 0usize);
        for &i in border {
            let (x, y) = (i % image.width, i / image.width);
            let t = if along_x {
                x as f64 / (image.width.max(2) - 1) as f64
            } else {
                y as f64 / (image.height.max(2) - 1) as f64
            };
            if (0.25..=0.75).contains(&t) {
                middle += 1;
                if distance2(&image.lab[i], &self.at_pixel(image, i)).sqrt() <= self.cut {
                    explained += 1;
                }
            }
        }
        explained as f64 >= GRADIENT_MIDDLE * middle as f64
    }

    /// A flat background from `start`: the border pixels within 3 robust deviations of it, refitted, twice.
    fn flat_from(image: &Image, border: &[usize], start: &[usize], sigma: f64) -> Background {
        let mut fit = Background::least_squares(image, start, false);
        let mut kept = start.to_vec();
        let mut cut = 6.0 * sigma;
        for _ in 0..2 {
            let deviation = |i: usize| distance2(&image.lab[i], &fit.at_pixel(image, i)).sqrt();
            let spread = median(kept.iter().map(|&i| deviation(i)).collect());
            cut = 3.0 * spread.max(sigma) * 2.0;
            let within: Vec<usize> = border.iter().copied().filter(|&i| deviation(i) <= cut).collect();
            if within.len() < 8 {
                break;
            }
            kept = within;
            fit = Background::least_squares(image, &kept, false);
        }
        Background { cut, ..fit }
    }
}

/// How much of the border in the middle of a gradient must be explained by it.
const GRADIENT_MIDDLE: f64 = 0.25;
/// How much of the border a flat colour must explain to replace a gradient that doesn't hold.
const BORDER_SHARE: f64 = 0.25;

/// The side (px) of the square at each corner the background is expected to reach.
const CORNER: usize = 3;

/// A robust fit's state: the border pixels kept, its flat and gradient fits, and its cut (OKLab distance).
struct Robust {
    kept: Vec<usize>,
    flat: Background,
    gradient: Background,
    cut: f64,
}

impl Background {
    /// Starting from the fit to `start`, the border pixels within 3 robust deviations of it are kept and refitted,
    /// twice. From the whole border, the deviation is the border's from the flat fit. From the corners, it is the kept
    /// pixels' own from the gradient fit: a tile filling most of the border would otherwise set it, and a gradient
    /// background widen it.
    fn robust(image: &Image, border: &[usize], start: &[usize], sigma: f64) -> Robust {
        let mut state = Robust {
            kept: start.to_vec(),
            flat: Background::least_squares(image, start, false),
            gradient: Background::least_squares(image, start, true),
            cut: f64::INFINITY,
        };
        let whole = start.len() == border.len();
        for _ in 0..2 {
            let reference = if whole { &state.flat } else { &state.gradient };
            let deviation = |i: usize| distance2(&image.lab[i], &reference.at_pixel(image, i)).sqrt();
            let deviations: Vec<f64> = border.iter().map(|&i| deviation(i)).collect();
            let spread = if whole {
                median(deviations.clone())
            } else {
                median(state.kept.iter().map(|&i| deviation(i)).collect())
            };
            let cut = 3.0 * spread.max(sigma) * 2.0;
            let kept: Vec<usize> = border
                .iter()
                .zip(&deviations)
                .filter(|(_, d)| **d <= cut)
                .map(|(i, _)| *i)
                .collect();
            if kept.len() < 8 {
                break;
            }
            state = Robust {
                flat: Background::least_squares(image, &kept, false),
                gradient: Background::least_squares(image, &kept, true),
                kept,
                cut,
            };
        }
        state
    }
}

impl Robust {
    /// How many corners' means are within `cut` of the fit (flat or gradient, whichever is chosen).
    fn explained(&self, image: &Image, corners: &[Vec<usize>], sigma: f64, cut: f64) -> usize {
        let fit = self.choose(image, sigma);
        corners
            .iter()
            .filter(|corner| {
                let n = corner.len() as f64;
                let mean = [0, 1, 2].map(|k| corner.iter().map(|&i| image.lab[i][k]).sum::<f64>() / n);
                let expected = [0, 1, 2].map(|k| corner.iter().map(|&i| fit.at_pixel(image, i)[k]).sum::<f64>() / n);
                distance2(&mean, &expected).sqrt() <= cut
            })
            .count()
    }

    /// The gradient if its 6 more numbers (at about 12 bits each) are paid for by the residual it removes.
    fn choose(&self, image: &Image, sigma: f64) -> Background {
        let border = &self.kept;
        let fits = (self.flat, self.gradient);
        let sse = |b: &Background| {
            border
                .iter()
                .map(|&i| distance2(&image.lab[i], &b.at_pixel(image, i)))
                .sum::<f64>()
        };
        let saved = (sse(&fits.0) - sse(&fits.1)) / (2.0 * sigma * sigma * std::f64::consts::LN_2);
        let fit = if saved > 6.0 * 12.0 { fits.1 } else { fits.0 };
        Background {
            cut: if self.cut.is_finite() { self.cut } else { 6.0 * sigma },
            ..fit
        }
    }
}

impl Background {
    fn least_squares(image: &Image, pixels: &[usize], gradient: bool) -> Background {
        let mut coefficients = [[0.0; 3]; 3];
        let n = pixels.len().max(1) as f64;
        for (k, row) in coefficients.iter_mut().enumerate() {
            row[0] = pixels.iter().map(|&i| image.lab[i][k]).sum::<f64>() / n;
        }
        let flat = Background {
            coefficients,
            gradient: false,
            cut: f64::INFINITY,
        };
        if !gradient {
            return flat;
        }
        let (mut a, mut b) = ([[0.0; 3]; 3], [[0.0; 3]; 3]);
        for &i in pixels {
            let (x, y) = (i % image.width, i / image.width);
            let row = [
                1.0,
                x as f64 / (image.width.max(2) - 1) as f64,
                y as f64 / (image.height.max(2) - 1) as f64,
            ];
            for r in 0..3 {
                for c in 0..3 {
                    a[r][c] += row[r] * row[c];
                }
                for (k, bk) in b.iter_mut().enumerate() {
                    bk[r] += row[r] * image.lab[i][k];
                }
            }
        }
        for k in 0..3 {
            match solve(a, b[k]) {
                Some(x) => coefficients[k] = x,
                None => return flat,
            }
        }
        Background {
            coefficients,
            gradient: true,
            cut: f64::INFINITY,
        }
    }
}

// --- the palette --------------------------------------------------------------------------------------------------

/// Label 0 is the background; 1.. index `centres` (centres[0] is unused).
pub struct Segmentation {
    pub labels: Vec<usize>,
    pub centres: Vec<Lab>,
    /// Pixels a cluster had when it joined the background only for being within `background_delta` of it (a merge
    /// the description length alone wouldn't make).
    pub faint: Vec<bool>,
}

struct Clustering<'a> {
    image: &'a Image,
    background: Vec<Lab>,
    centres: Vec<Option<Lab>>,
}

impl Clustering<'_> {
    fn cost(&self, i: usize, label: usize) -> f64 {
        if label == 0 {
            distance2(&self.image.lab[i], &self.background[i])
        } else {
            self.centres[label].map_or(f64::INFINITY, |c| distance2(&self.image.lab[i], &c))
        }
    }

    fn assign(&self) -> Vec<usize> {
        (0..self.image.lab.len())
            .map(|i| {
                let mut best = (self.cost(i, 0), 0);
                for k in 1..self.centres.len() {
                    let c = self.cost(i, k);
                    if c < best.0 {
                        best = (c, k);
                    }
                }
                best.1
            })
            .collect()
    }

    fn update(&mut self, labels: &[usize]) {
        let mut sums = vec![([0.0; 3], 0usize); self.centres.len()];
        for (i, &l) in labels.iter().enumerate() {
            let s = &mut sums[l];
            for k in 0..3 {
                s.0[k] += self.image.lab[i][k];
            }
            s.1 += 1;
        }
        for (k, (sum, n)) in sums.iter().enumerate().skip(1) {
            self.centres[k] = (*n > 0).then(|| sum.map(|v| v / *n as f64));
        }
    }

    fn iterate(&mut self, rounds: usize) -> Vec<usize> {
        let mut labels = self.assign();
        for _ in 0..rounds {
            self.update(&labels);
            let next = self.assign();
            if next == labels {
                break;
            }
            labels = next;
        }
        labels
    }
}

/// The noise above which close clusters merge regardless (`two_flat_colours`): it is barely over `SIGMA_FLOOR`, since
/// keeping them apart on a noisy image only moves k-means around, and can tip faint clutter out of the background.
const FLAT_SIGMA: f64 = 0.0025;
/// The share of two clusters' pixels below which the middle third between their centres counts as empty.
const FLAT_VALLEY: f64 = 0.05;
/// The share of a cluster's pixels that must have all their neighbours in it, for it to be a colour of its own.
const FLAT_CORE: f64 = 0.2;

fn core_share(image: &Image, labels: &[usize], k: usize) -> f64 {
    let (mut total, mut core) = (0usize, 0usize);
    for (i, &l) in labels.iter().enumerate() {
        if l == k {
            total += 1;
            core += usize::from(image.neighbours(i).all(|j| labels[j] == k));
        }
    }
    core as f64 / total.max(1) as f64
}

/// Whether two close clusters are two flat colours rather than one colour's shading: few of their pixels lie in the
/// middle third between their centres, since a ramp fills that valley. Both need a core, or the valley says nothing: a
/// blurred edge's in-between colour is a thin rim, and its pixels are as far from the middle as a flat fill's.
fn two_flat_colours(image: &Image, labels: &[usize], a: usize, b: usize, centres: &[Option<Lab>]) -> bool {
    let (Some(ca), Some(cb)) = (centres[a], centres[b]) else {
        return false;
    };
    let axis = [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]];
    let length2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if length2 <= 0.0 {
        return false;
    }
    let (mut total, mut middle) = (0usize, 0usize);
    for (p, &l) in image.lab.iter().zip(labels) {
        if l == a || l == b {
            total += 1;
            let t = ((p[0] - ca[0]) * axis[0] + (p[1] - ca[1]) * axis[1] + (p[2] - ca[2]) * axis[2]) / length2;
            middle += usize::from((1.0 / 3.0..2.0 / 3.0).contains(&t));
        }
    }
    (middle as f64) < FLAT_VALLEY * total as f64
        && core_share(image, labels, a) >= FLAT_CORE
        && core_share(image, labels, b) >= FLAT_CORE
}

pub fn segment(image: &Image, background: &Background, sigma: f64, config: &TraceConfig) -> Segmentation {
    let n = image.lab.len();
    let bg: Vec<Lab> = (0..n).map(|i| background.at_pixel(image, i)).collect();
    // k-means++ seeds among the pixels the background doesn't explain (every 3rd, for speed)
    let mut rng = Pcg32::new(config.seed);
    let candidates: Vec<usize> = (0..n)
        .step_by(3)
        .filter(|&i| distance2(&image.lab[i], &bg[i]) > 9.0 * sigma * sigma)
        .collect();
    let mut centres: Vec<Option<Lab>> = vec![None];
    if !candidates.is_empty() {
        let mut nearest: Vec<f64> = candidates.iter().map(|&i| distance2(&image.lab[i], &bg[i])).collect();
        for _ in 0..config.colors {
            let total: f64 = nearest.iter().sum();
            if total <= 0.0 {
                break;
            }
            let mut pick = rng.next_u32() as f64 / 4_294_967_296.0 * total;
            let mut chosen = candidates.len() - 1;
            for (j, d) in nearest.iter().enumerate() {
                pick -= d;
                if pick <= 0.0 {
                    chosen = j;
                    break;
                }
            }
            let centre = image.lab[candidates[chosen]];
            centres.push(Some(centre));
            for (j, &i) in candidates.iter().enumerate() {
                nearest[j] = nearest[j].min(distance2(&image.lab[i], &centre));
            }
        }
    }
    let mut clustering = Clustering {
        image,
        background: bg,
        centres,
    };
    let mut labels = clustering.iterate(12);
    let mut faint = vec![false; n];
    // merges, closest pair first, while shorter to describe (or closer than merge_delta)
    let bits_per_sse = 1.0 / (2.0 * sigma * sigma * std::f64::consts::LN_2);
    loop {
        let k = clustering.centres.len();
        let mut n_of = vec![0usize; k];
        // each cluster's squared error against the background, so a merge into it is exact
        let mut bg_sse = vec![0.0; k];
        let mut own_sse = vec![0.0; k];
        for (i, &l) in labels.iter().enumerate() {
            n_of[l] += 1;
            bg_sse[l] += clustering.cost(i, 0);
            own_sse[l] += clustering.cost(i, l);
        }
        let mut shared: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for i in 0..n {
            for j in image.neighbours(i).filter(|&j| j > i) {
                let (a, b) = (labels[i].min(labels[j]), labels[i].max(labels[j]));
                if a != b {
                    *shared.entry((a, b)).or_default() += 1;
                }
            }
        }
        let mut best: Option<(f64, usize, usize, bool)> = None;
        for a in 0..k {
            for b in a + 1..k {
                let (Some(cb), true) = (clustering.centres[b], n_of[a] > 0 && n_of[b] > 0) else {
                    continue;
                };
                let (added, delta) = if a == 0 {
                    // b absorbed by the background
                    let mean_bg = [0, 1, 2].map(|c| {
                        labels
                            .iter()
                            .zip(&clustering.background)
                            .filter(|(l, _)| **l == b)
                            .map(|(_, p)| p[c])
                            .sum::<f64>()
                            / n_of[b] as f64
                    });
                    (bg_sse[b] - own_sse[b], distance2(&cb, &mean_bg).sqrt())
                } else {
                    let ca = clustering.centres[a].unwrap();
                    let (na, nb) = (n_of[a] as f64, n_of[b] as f64);
                    (na * nb / (na + nb) * distance2(&ca, &cb), distance2(&ca, &cb).sqrt())
                };
                let saved = 24.0 + config.edge_bits * *shared.get(&(a, b)).unwrap_or(&0) as f64;
                let close = if a == 0 {
                    config.background_delta
                } else {
                    config.merge_delta
                };
                let shorter = added * bits_per_sse < saved;
                let colours =
                    a != 0 && sigma < FLAT_SIGMA && two_flat_colours(image, &labels, a, b, &clustering.centres);
                let forced = delta < close && !colours;
                if (forced || shorter) && best.is_none_or(|(d, _, _, _)| delta < d) {
                    best = Some((delta, a, b, a == 0 && !shorter));
                }
            }
        }
        let Some((_, a, b, forced)) = best else { break };
        for (i, l) in labels.iter_mut().enumerate() {
            if *l == b {
                *l = a;
                faint[i] |= forced;
            }
        }
        clustering.centres[b] = None;
        clustering.update(&labels);
        labels = clustering.iterate(4);
    }
    // drop the emptied clusters, renumbering
    let mut renumber = vec![0usize; clustering.centres.len()];
    let mut centres = vec![[0.0; 3]];
    for (k, c) in clustering.centres.iter().enumerate().skip(1) {
        if let Some(c) = c {
            if labels.contains(&k) {
                renumber[k] = centres.len();
                centres.push(*c);
            }
        }
    }
    Segmentation {
        labels: labels.iter().map(|&l| renumber[l]).collect(),
        centres,
        faint,
    }
}

// --- labels and parts ---------------------------------------------------------------------------------------------

/// One pass of a 3 x 3 majority filter: a pixel takes the most common label around it (itself included), if more than
/// four of the nine have it.
pub fn majority(labels: &[usize], width: usize, height: usize) -> Vec<usize> {
    let mut out = labels.to_vec();
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for y in 0..height {
        for x in 0..width {
            counts.clear();
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        *counts.entry(labels[ny as usize * width + nx as usize]).or_default() += 1;
                    }
                }
            }
            if let Some((&label, &count)) = counts.iter().max_by_key(|(l, c)| (**c, std::cmp::Reverse(**l))) {
                if count > 4 {
                    out[y * width + x] = label;
                }
            }
        }
    }
    out
}

/// 4-connected components of equal labels: a component id per pixel, and each component's pixels.
pub fn components(image: &Image, labels: &[usize]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let n = labels.len();
    let mut id = vec![usize::MAX; n];
    let mut members: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if id[start] != usize::MAX {
            continue;
        }
        let c = members.len();
        let mut queue = VecDeque::from([start]);
        id[start] = c;
        let mut pixels = Vec::new();
        while let Some(i) = queue.pop_front() {
            pixels.push(i);
            for j in image.neighbours(i) {
                if id[j] == usize::MAX && labels[j] == labels[start] {
                    id[j] = c;
                    queue.push_back(j);
                }
            }
        }
        members.push(pixels);
    }
    (id, members)
}

/// Pixels that are still there after `steps` 4-neighbour erosions of the component.
fn eroded(image: &Image, id: &[usize], pixels: &[usize], steps: usize) -> usize {
    let c = id[pixels[0]];
    let mut depth: BTreeMap<usize, usize> = BTreeMap::new();
    let mut queue = VecDeque::new();
    for &i in pixels {
        let x = (i % image.width, i / image.width);
        let edge = x.0 == 0
            || x.1 == 0
            || x.0 + 1 == image.width
            || x.1 + 1 == image.height
            || image.neighbours(i).any(|j| id[j] != c);
        // the image border is not an edge of the part: a part running off the image is not thin there
        let off_image = image.neighbours(i).all(|j| id[j] == c);
        if edge && !off_image {
            depth.insert(i, 1);
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let d = depth[&i];
        for j in image.neighbours(i) {
            if id[j] == c && !depth.contains_key(&j) {
                depth.insert(j, d + 1);
                queue.push_back(j);
            }
        }
    }
    pixels
        .iter()
        .filter(|i| depth.get(i).is_none_or(|d| *d > steps))
        .count()
}

/// Dissolves components that are too small or too thin to be parts: each of their pixels takes the label of the
/// nearest pixel that stays (a breadth-first flood from what stays). The background where it reaches the image's edge
/// always stays: a tile's rounded corner leaves it a thin sliver there, which is still the background.
pub fn clean_labels(image: &Image, labels: &[usize], config: &TraceConfig) -> Vec<usize> {
    let mut labels = majority(labels, image.width, image.height);
    labels = majority(&labels, image.width, image.height);
    let (id, members) = components(image, &labels);
    let (w, h) = (image.width, image.height);
    let dissolve: Vec<bool> = members
        .iter()
        .map(|pixels| {
            let edge = labels[pixels[0]] == 0
                && pixels.iter().any(|&i| {
                    let (x, y) = (i % w, i / w);
                    x == 0 || y == 0 || x + 1 == w || y + 1 == h
                });
            !edge && (pixels.len() < config.min_area || eroded(image, &id, pixels, 1) * 4 < pixels.len())
        })
        .collect();
    if dissolve.iter().all(|d| *d) {
        return labels;
    }
    let mut out: Vec<Option<usize>> = (0..labels.len())
        .map(|i| (!dissolve[id[i]]).then_some(labels[i]))
        .collect();
    let mut queue: VecDeque<usize> = (0..labels.len()).filter(|&i| out[i].is_some()).collect();
    while let Some(i) = queue.pop_front() {
        for j in image.neighbours(i) {
            if out[j].is_none() {
                out[j] = out[i];
                queue.push_back(j);
            }
        }
    }
    out.into_iter().map(|l| l.unwrap_or(0)).collect()
}

/// The background reaches the image's edge: a background-labelled component enclosed by parts whose interior's mean
/// colour is further from the background's than one robust deviation of a border pixel (`Background::cut` / 6) -- the
/// white of a letter on a tile against an off-white background, within `background_delta` of it -- is a colour of its
/// own, a new label whose centre is its mean. One that is the background's colour stays a hole, and so does one with
/// too little interior to tell (a gap between parts, blurred by them). The interior's mean has no blurred edge, and
/// the noise averages away in it, so one deviation is a safe margin.
pub fn enclose(image: &Image, background: &[Lab], cut: f64, labels: &mut [usize], centres: &mut Vec<Lab>) {
    let (w, h) = (image.width, image.height);
    let (_, members) = components(image, labels);
    let enclosed: Vec<&Vec<usize>> = members
        .iter()
        .filter(|pixels| labels[pixels[0]] == 0)
        .filter(|pixels| {
            pixels.iter().all(|&i| {
                let (x, y) = (i % w, i / w);
                x > 0 && y > 0 && x + 1 < w && y + 1 < h
            })
        })
        .filter(|pixels| stands_out(image, background, cut, 1.0, pixels))
        .collect();
    let n: usize = enclosed.iter().map(|p| p.len()).sum();
    if n == 0 {
        return;
    }
    let label = centres.len();
    let mut mean = [0.0; 3];
    for &i in enclosed.iter().flat_map(|p| p.iter()) {
        for (k, m) in mean.iter_mut().enumerate() {
            *m += image.lab[i][k] / n as f64;
        }
        labels[i] = label;
    }
    centres.push(mean);
}

/// Whether a region's interior -- its pixels `INTERIOR` clear of its edge, clear of the blur it shares with what is
/// around it -- has a mean colour further from the background's than `deviations` robust deviations (`cut` / 6 each),
/// with at least `MIN_INTERIOR` pixels to tell by.
fn stands_out(image: &Image, background: &[Lab], cut: f64, deviations: f64, pixels: &[usize]) -> bool {
    deviations_out(image, background, cut, pixels).is_some_and(|d| d > deviations)
}

/// How many robust deviations (`cut` / 6 each) a region's interior's mean colour is from the background's, if it has
/// an interior to tell by.
fn deviations_out(image: &Image, background: &[Lab], cut: f64, pixels: &[usize]) -> Option<f64> {
    let (w, h) = (image.width, image.height);
    let mean = |pixels: &[usize], of: &dyn Fn(usize) -> Lab| {
        let n = pixels.len() as f64;
        [0, 1, 2].map(|k| pixels.iter().map(|&i| of(i)[k]).sum::<f64>() / n)
    };
    let mut inside = vec![false; w * h];
    for &i in pixels {
        inside[i] = true;
    }
    let r = INTERIOR as i64;
    let interior: Vec<usize> = pixels
        .iter()
        .copied()
        .filter(|&i| {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            x >= r
                && y >= r
                && x + r < w as i64
                && y + r < h as i64
                && (-r..=r).all(|dy| (-r..=r).all(|dx| inside[((y + dy) * w as i64 + x + dx) as usize]))
        })
        .collect();
    (interior.len() >= MIN_INTERIOR).then(|| {
        distance2(
            &mean(&interior, &|i| image.lab[i]),
            &mean(&interior, &|i| background[i]),
        )
        .sqrt()
            / (cut / 6.0)
    })
}

/// A light part on a light background -- a white plate, a grey speaker cone -- is within `background_delta` of it, so
/// `segment` gave it to the background with the faint clutter that rule is for (`Segmentation::faint`). Each region of
/// such pixels still labelled background after `clean_labels` and `enclose` is a part again, a colour of its own, if
/// - it is big and thick enough to be a part (`min_area`, and `clean_labels`' thinness test),
/// - its interior's colour is `RESTORE_DEVIATIONS` robust deviations from the background's (`stands_out`): on a clean
///   image any light part is, while faint clutter on a noisy one is barely off the background, and
/// - at least `RESTORE_SHARE` of its boundary is with parts: a part of the subject touches the rest of it, while clutter
///   mostly lies alone in the background. A region `RESTORE_FAR` deviations out needs no contact: a chicken's pale body
///   has little more than an eye and a comb on it, and clutter doesn't stand out that far.
pub fn restore(
    image: &Image,
    background: &[Lab],
    cut: f64,
    faint: &[bool],
    config: &TraceConfig,
    labels: &mut [usize],
    centres: &mut Vec<Lab>,
) {
    let candidate: Vec<usize> = (0..labels.len())
        .map(|i| usize::from(faint[i] && labels[i] == 0))
        .collect();
    let (id, members) = components(image, &candidate);
    for pixels in members.iter().filter(|p| candidate[p[0]] == 1) {
        if pixels.len() < config.min_area
            || eroded(image, &id, pixels, 1) * 4 < pixels.len()
            || !stands_out(image, background, cut, RESTORE_DEVIATIONS, pixels)
        {
            continue;
        }
        let c = id[pixels[0]];
        let (mut subject, mut empty) = (0usize, 0usize);
        for &i in pixels {
            for j in image.neighbours(i).filter(|&j| id[j] != c) {
                if labels[j] == 0 {
                    empty += 1;
                } else {
                    subject += 1;
                }
            }
        }
        if !stands_out(image, background, cut, RESTORE_FAR, pixels)
            && (subject as f64) < RESTORE_SHARE * (subject + empty).max(1) as f64
        {
            continue;
        }
        let label = centres.len();
        let mut mean = [0.0; 3];
        for &i in pixels {
            for (k, m) in mean.iter_mut().enumerate() {
                *m += image.lab[i][k] / pixels.len() as f64;
            }
            labels[i] = label;
        }
        centres.push(mean);
    }
}

/// How many robust deviations from the background a restored part's interior must be, how much of its boundary it must
/// share with parts, and how many deviations out it needn't share any.
const RESTORE_DEVIATIONS: f64 = 3.0;
const RESTORE_SHARE: f64 = 0.25;
const RESTORE_FAR: f64 = 5.0;

/// How far (px) inside an enclosed region its interior starts, and the fewest interior pixels it must have.
const INTERIOR: usize = 2;
const MIN_INTERIOR: usize = 16;

/// A part: a connected component of one (non-background) label.
pub struct Part {
    pub pixels: Vec<usize>,
    pub fill: [u8; 3],
    pub label: usize,
}

pub fn parts(image: &Image, labels: &[usize]) -> Vec<Part> {
    let fills = fills(image, labels);
    let (_, members) = components(image, labels);
    members
        .into_iter()
        .filter(|pixels| labels[pixels[0]] != 0)
        .map(|pixels| Part {
            fill: fills[labels[pixels[0]]],
            label: labels[pixels[0]],
            pixels,
        })
        .collect()
}

/// One fill per label (a palette entry): the mean of its components' interiors -- or of all its pixels, for a label
/// that is mostly edge. Label 0's is the background's mean colour.
pub fn fills(image: &Image, labels: &[usize]) -> Vec<[u8; 3]> {
    let (id, _) = components(image, labels);
    let k = labels.iter().max().map_or(1, |m| m + 1);
    let (mut interior, mut all) = (vec![([0.0; 3], 0usize); k], vec![([0.0; 3], 0usize); k]);
    for (i, &l) in labels.iter().enumerate() {
        let sums = if image.neighbours(i).all(|j| id[j] == id[i]) {
            &mut interior
        } else {
            &mut all
        };
        for c in 0..3 {
            sums[l].0[c] += image.rgb[i][c] as f64;
        }
        sums[l].1 += 1;
    }
    (0..k)
        .map(|l| {
            let (inside, edge) = (interior[l], all[l]);
            let (sum, n) = if inside.1 * 3 >= edge.1 {
                inside
            } else {
                ([0, 1, 2].map(|c| inside.0[c] + edge.0[c]), inside.1 + edge.1)
            };
            sum.map(|v| (v / n.max(1) as f64).round().clamp(0.0, 255.0) as u8)
        })
        .collect()
}

/// The segmented image: what a part's soft mask and a shape's fit are computed from.
pub struct Scene<'a> {
    pub image: &'a Image,
    pub labels: Vec<usize>,
    /// Each label's OKLab colour (label 0's is `background`, per pixel).
    pub centres: Vec<Lab>,
    pub background: Vec<Lab>,
}

impl Scene<'_> {
    fn colour(&self, label: usize, i: usize) -> Lab {
        if label == 0 {
            self.background[i]
        } else {
            self.centres[label]
        }
    }

    /// How much of each pixel is these parts' (one label's): 1 inside, 0 outside, and along their edge the fraction
    /// of the part's colour in the pixel's -- its colour projected onto the line from what is across the edge to the
    /// part's -- so an anti-aliased edge is placed to a fraction of a pixel.
    pub fn soft(&self, parts: &[&Part]) -> Vec<f64> {
        let (w, h) = (self.image.width, self.image.height);
        let n = w * h;
        let mut inside = vec![false; n];
        for p in parts {
            for &i in &p.pixels {
                inside[i] = true;
            }
        }
        let mut out: Vec<f64> = inside.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        let Some(first) = parts.first() else { return out };
        let own = first.label;
        let mut across: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..n {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            across.clear();
            let mut mixed = false;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if inside[j] != inside[i] {
                        mixed = true;
                    }
                    if !inside[j] {
                        *across.entry(self.labels[j]).or_default() += 1;
                    }
                }
            }
            if !mixed {
                continue;
            }
            let other = if inside[i] {
                across
                    .iter()
                    .max_by_key(|(l, c)| (**c, std::cmp::Reverse(**l)))
                    .map(|(l, _)| *l)
            } else {
                Some(self.labels[i])
            };
            let Some(other) = other.filter(|&l| l != own) else {
                continue;
            };
            let (a, b) = (self.colour(other, i), self.centres[own]);
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if len2 < 1e-6 {
                continue;
            }
            let c = self.image.lab[i];
            let t = ((c[0] - a[0]) * d[0] + (c[1] - a[1]) * d[1] + (c[2] - a[2]) * d[2]) / len2;
            out[i] = t.clamp(0.0, 1.0);
        }
        out
    }

    fn fit(&self, parts: &[&Part], above: &[&Part], costs: &Costs) -> Option<Fit> {
        let visible = self.soft(parts);
        let free = mask_of(self.labels.len(), above);
        Target {
            width: self.image.width,
            height: self.image.height,
            visible: &visible,
            free: &free,
        }
        .fit(costs)
    }

    /// A part's final shape, tightened (`Target::tightened`) and with its holes cut (`Target::with_holes`).
    fn holes(&self, part: &Part, above: &[&Part], fit: Fit, costs: &Costs) -> Fit {
        let visible = self.soft(&[part]);
        let free = mask_of(self.labels.len(), above);
        let target = Target {
            width: self.image.width,
            height: self.image.height,
            visible: &visible,
            free: &free,
        };
        // completing an outline can make it far worse than none (a thin, curved part on a larger one)
        let fit = target
            .fit_visible(costs)
            .filter(|as_seen| as_seen.bits() + VISIBLE_PIXELS * costs.pixel_bits < fit.bits())
            .unwrap_or(fit);
        target.with_holes(target.tightened(fit, costs), costs)
    }

    fn bits(&self, parts: &[&Part], above: &[&Part], costs: &Costs) -> f64 {
        self.fit(parts, above, costs).map_or(f64::INFINITY, |f| f.bits())
    }
}

/// Joins components of one label that one shape says more briefly than two -- a part an occluder cut in two -- with
/// every other part's pixels free (any of them could be above). But a piece that another part covers, once that part is
/// completed under its own occluders, is never joined: that part is beneath it (an eye on a face, a letter on a band),
/// so the piece is a detail on it, not a glimpse of something it hides. Returns each part's completed shape
/// (`completed`), for `order`.
pub fn join(scene: &Scene, parts: &mut Vec<Part>, costs: &Costs) -> Vec<Option<Vec<f64>>> {
    let mut completed = completed(scene, parts, costs);
    let covers = |m: &[f64], p: &Part| p.pixels.iter().map(|&i| m[i]).sum::<f64>() >= 0.5 * p.pixels.len() as f64;
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..parts.len() {
            for b in a + 1..parts.len() {
                if parts[a].label != parts[b].label {
                    continue;
                }
                let others: Vec<usize> = (0..parts.len()).filter(|&k| k != a && k != b).collect();
                let detail = others.iter().any(|&k| {
                    completed[k]
                        .as_ref()
                        .is_some_and(|m| covers(m, &parts[a]) || covers(m, &parts[b]))
                });
                if detail {
                    continue;
                }
                let above: Vec<&Part> = others.iter().map(|&k| &parts[k]).collect();
                let (pa, pb) = (&parts[a], &parts[b]);
                let saved = scene.bits(&[pa], &above, costs) + scene.bits(&[pb], &above, costs)
                    - scene.bits(&[pa, pb], &above, costs);
                if saved > 0.0 && best.is_none_or(|(s, _, _)| saved > s) {
                    best = Some((saved, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else { return completed };
        let joined = parts.remove(b);
        parts[a].pixels.extend(joined.pixels);
        // every other part's free pixels (all parts but itself) are the same as before; only the joined one changes
        completed.remove(b);
        completed[a] = completed_one(scene, parts, a, costs);
    }
}

/// Each part's shape fitted with every other part free (any of them could be above it), as coverage over the image.
fn completed(scene: &Scene, parts: &[Part], costs: &Costs) -> Vec<Option<Vec<f64>>> {
    (0..parts.len())
        .map(|k| completed_one(scene, parts, k, costs))
        .collect()
}

fn completed_one(scene: &Scene, parts: &[Part], k: usize, costs: &Costs) -> Option<Vec<f64>> {
    let others: Vec<&Part> = (0..parts.len()).filter(|&j| j != k).map(|j| &parts[j]).collect();
    let fit = scene.fit(&[&parts[k]], &others, costs)?;
    let (w, h) = (scene.image.width, scene.image.height);
    let shape = Shape {
        geometry: fit.geometry,
        fill: [0, 0, 0],
        opacity: 1.0,
        group: None,
    };
    let mask = shape.mask(w as u32, h as u32);
    let mut out = vec![0.0; w * h];
    for my in 0..mask.height {
        for mx in 0..mask.width {
            out[(mask.y0 + my) * w + mask.x0 + mx] = mask.data[my * mask.width + mx];
        }
    }
    Some(out)
}

/// A final shape takes the visible outline over the completed one only if it saves this many wrong pixels' worth of bits.
const VISIBLE_PIXELS: f64 = 40.0;
/// How deep (px) inside its convex hull a part's outline must go for a notch to be a place to cut it.
const NOTCH_DEPTH: f64 = 1.5;
/// How many of a part's deepest notches are tried as cut ends, in pairs.
const NOTCHES: usize = 4;
/// A cut must save this many outline moves' worth of bits (`Costs::outline_gain`): on a noisy image a notch can be
/// noise, and two shapes fitted to it are a less smooth answer than one.
const CUT_GAIN: f64 = 3.0;

/// One part's shapes: a part may be two of one colour that overlap (two touching parts of one colour are one
/// component), so it is cut along the chord between two of its outline's notches -- where two outlines cross -- and
/// each half fitted with the other's pixels free, so each completes under the other. The cut is kept, and each half
/// cut again, while the halves are shorter to describe than the whole. Each shape it ends with has its holes cut.
fn shapes_of(scene: &Scene, part: &Part, above: &[&Part], costs: &Costs) -> Vec<Fit> {
    let Some(whole) = scene.fit(&[part], above, costs) else {
        return Vec::new();
    };
    let (w, h) = (scene.image.width, scene.image.height);
    let mut mask = vec![0.0; w * h];
    for &i in &part.pixels {
        mask[i] = 1.0;
    }
    let Some(loop_) = largest_contour(&mask, w, h) else {
        return vec![scene.holes(part, above, whole, costs)];
    };
    let notches: Vec<usize> = notches(&loop_, NOTCH_DEPTH).into_iter().take(NOTCHES).collect();
    let mut best: Option<(f64, Part, Part)> = None;
    for a in 0..notches.len() {
        for b in a + 1..notches.len() {
            let (i, j) = (notches[a].min(notches[b]), notches[a].max(notches[b]));
            let side = &loop_[i..=j];
            let (one, two): (Vec<usize>, Vec<usize>) = part
                .pixels
                .iter()
                .partition(|&&k| inside_polygon(((k % w) as f64 + 0.5, (k / w) as f64 + 0.5), side));
            if one.len() < MIN_HALF || two.len() < MIN_HALF {
                continue;
            }
            let halves = [one, two].map(|pixels| Part {
                pixels,
                fill: part.fill,
                label: part.label,
            });
            let bits: f64 = (0..2)
                .map(|k| {
                    let mut free = above.to_vec();
                    free.push(&halves[1 - k]);
                    scene.bits(&[&halves[k]], &free, costs)
                })
                .sum();
            if bits + CUT_GAIN * costs.outline_gain < whole.bits() && best.as_ref().is_none_or(|(b, _, _)| bits < *b) {
                let [one, two] = halves;
                best = Some((bits, one, two));
            }
        }
    }
    let Some((_, one, two)) = best else {
        return vec![scene.holes(part, above, whole, costs)];
    };
    let mut out = Vec::new();
    for (half, other) in [(&one, &two), (&two, &one)] {
        let mut free = above.to_vec();
        free.push(other);
        out.extend(shapes_of(scene, half, &free, costs));
    }
    out
}

/// The fewest pixels a half of a cut part may have.
const MIN_HALF: usize = 12;

/// Shapes for what a part's shapes left uncovered. A part's pieces are joined with every other part free to lie above
/// them, but its final shape can only pass under what ended up above it: if the order disagrees, the far piece is simply
/// lost. So each connected piece left uncovered gets its own shape, if that's shorter than leaving its pixels wrong.
fn leftovers(scene: &Scene, part: &Part, above: &[&Part], fits: &[Fit], costs: &Costs) -> Vec<Fit> {
    let (w, h) = (scene.image.width, scene.image.height);
    let mut covered = vec![0.0f64; w * h];
    for fit in fits {
        let shape = Shape {
            geometry: fit.geometry.clone(),
            fill: [0, 0, 0],
            opacity: 1.0,
            group: None,
        };
        let mask = shape.mask(w as u32, h as u32);
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let i = (mask.y0 + my) * w + mask.x0 + mx;
                covered[i] = covered[i].max(mask.data[my * mask.width + mx]);
            }
        }
    }
    let mut left = vec![false; w * h];
    for &i in &part.pixels {
        left[i] = covered[i] < 0.5;
    }
    let mut out = Vec::new();
    for &start in &part.pixels {
        if !left[start] {
            continue;
        }
        left[start] = false;
        let mut pixels = vec![start];
        let mut next = 0;
        while next < pixels.len() {
            for j in scene.image.neighbours(pixels[next]) {
                if left[j] {
                    left[j] = false;
                    pixels.push(j);
                }
            }
            next += 1;
        }
        // a sliver along the shape's edge is its misfit, not a lost piece: a piece has a core
        let mut member = vec![false; w * h];
        for &i in &pixels {
            member[i] = true;
        }
        let core = pixels
            .iter()
            .filter(|&&i| scene.image.neighbours(i).all(|j| member[j]))
            .count();
        if core < MIN_HALF {
            continue;
        }
        let wrong = costs.pixel_bits * pixels.len() as f64;
        let piece = Part {
            pixels,
            fill: part.fill,
            label: part.label,
        };
        if let Some(fit) = scene.fit(&[&piece], above, costs).filter(|fit| fit.bits() < wrong) {
            out.push(fit);
        }
    }
    out
}

// --- order and shapes ---------------------------------------------------------------------------------------------

fn mask_of(n: usize, parts: &[&Part]) -> Vec<bool> {
    let mut m = vec![false; n];
    for p in parts {
        for &i in &p.pixels {
            m[i] = true;
        }
    }
    m
}

/// Paint order, bottom first, from each part's completed shape (`completed`: fitted with every other part free, since
/// any of them could be above it). Of two touching parts, the one whose shape covers more of the other, as a fraction
/// of it, goes below -- a part completed under its occluders claims their pixels, one on top claims none. Kahn's sort takes the
/// largest available part first, and the largest remaining one when a cycle blocks it.
pub fn order(image: &Image, parts: &[Part], completed: &[Option<Vec<f64>>]) -> Vec<usize> {
    let n = image.lab.len();
    let mut owner = vec![usize::MAX; n];
    for (k, p) in parts.iter().enumerate() {
        for &i in &p.pixels {
            owner[i] = k;
        }
    }
    let mut touching: BTreeMap<(usize, usize), ()> = BTreeMap::new();
    for i in 0..n {
        for j in image.neighbours(i) {
            let (a, b) = (owner[i], owner[j]);
            if a != usize::MAX && b != usize::MAX && a < b {
                touching.insert((a, b), ());
            }
        }
    }
    // claims[a][b]: how much of b's pixels a's completed shape covers
    let mut claims = vec![vec![0.0; parts.len()]; parts.len()];
    for (a, mask) in completed.iter().enumerate() {
        let Some(mask) = mask else { continue };
        for (i, &m) in mask.iter().enumerate() {
            if m != 0.0 && owner[i] != usize::MAX && owner[i] != a {
                claims[a][owner[i]] += m;
            }
        }
    }
    let mut below: Vec<Vec<usize>> = vec![Vec::new(); parts.len()]; // below[b] = parts that must be under b
    for &(a, b) in touching.keys() {
        // as fractions of the part claimed: a small part completed with a big one free can sprawl over more of its
        // pixels than the big one, completed under it, covers of the small one's -- yet it covers a sliver of it
        let (na, nb) = (parts[a].pixels.len() as f64, parts[b].pixels.len() as f64);
        let (ab, ba) = (claims[a][b] / nb, claims[b][a] / na);
        let a_first = if (ab - ba).abs() > 0.5 / na.min(nb) {
            ab > ba
        } else {
            parts[a].pixels.len() >= parts[b].pixels.len()
        };
        if a_first {
            below[b].push(a);
        } else {
            below[a].push(b);
        }
    }
    let mut placed = vec![false; parts.len()];
    let mut out = Vec::with_capacity(parts.len());
    while out.len() < parts.len() {
        let ready = |k: usize| !placed[k] && below[k].iter().all(|&j| placed[j]);
        let by_size = |k: &usize| (parts[*k].pixels.len(), std::cmp::Reverse(*k));
        let next = (0..parts.len())
            .filter(|&k| ready(k))
            .max_by_key(by_size)
            .or_else(|| (0..parts.len()).filter(|&k| !placed[k]).max_by_key(by_size))
            .unwrap();
        placed[next] = true;
        out.push(next);
    }
    out
}

/// What each stage made, for showing the pipeline at work (the game page draws them).
pub struct Stages {
    pub sigma: f64,
    /// Whether the background was fitted as a gradient (else a constant).
    pub gradient: bool,
    /// Per pixel, its k-means label after merging (0 = the background), before the labels were cleaned.
    pub clustered: Vec<usize>,
    /// Per pixel, its label after cleaning: what the parts are cut from.
    pub labels: Vec<usize>,
    /// Per label, its fill (`fills`).
    pub palette: Vec<[u8; 3]>,
    pub document: Document,
}

/// The whole pipeline: an opaque RGBA image to a transparent, layered SVG document.
pub fn trace(rgba: &[u8], width: usize, height: usize, config: &TraceConfig) -> Result<Document, String> {
    stages(rgba, width, height, config).map(|s| s.document)
}

/// The whole pipeline, keeping what each stage made.
pub fn stages(rgba: &[u8], width: usize, height: usize, config: &TraceConfig) -> Result<Stages, String> {
    let image = Image::from_rgba8(rgba, width, height)?;
    let sigma = noise_sigma(&image);
    let background = Background::fit(&image, sigma);
    let segmentation = segment(&image, &background, sigma, config);
    let mut labels = clean_labels(&image, &segmentation.labels, config);
    let mut centres = segmentation.centres.clone();
    let per_pixel: Vec<Lab> = (0..width * height).map(|i| background.at_pixel(&image, i)).collect();
    enclose(&image, &per_pixel, background.cut, &mut labels, &mut centres);
    restore(
        &image,
        &per_pixel,
        background.cut,
        &segmentation.faint,
        config,
        &mut labels,
        &mut centres,
    );
    let scene = Scene {
        image: &image,
        labels,
        centres,
        background: per_pixel,
    };
    let mut costs = Costs::new(width, height, config.pixel_bits);
    costs.outline_gain = config.outline_gain * (sigma - SIGMA_FLOOR) * 1000.0;
    let mut parts = parts(&image, &scene.labels);
    let completed = join(&scene, &mut parts, &costs);
    let order = order(&image, &parts, &completed);
    let mut shapes = Vec::new();
    let mut groups = Vec::new();
    for (place, &k) in order.iter().enumerate() {
        let above: Vec<&Part> = order[place + 1..].iter().map(|&j| &parts[j]).collect();
        let mut fits = shapes_of(&scene, &parts[k], &above, &costs);
        let more = leftovers(&scene, &parts[k], &above, &fits, &costs);
        fits.extend(more);
        for fit in fits {
            groups.push(Group { parent: None });
            shapes.push(Shape {
                geometry: fit.geometry,
                fill: parts[k].fill,
                opacity: 1.0,
                group: Some(groups.len() - 1),
            });
        }
    }
    Ok(Stages {
        sigma,
        gradient: background.gradient,
        palette: fills(&image, &scene.labels),
        clustered: segmentation.labels,
        labels: scene.labels,
        document: Document {
            width: width as u32,
            height: height as u32,
            groups,
            shapes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copycat::layer_recovery;
    use crate::raster::{coverage, PathBuilder};

    const SCENE: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
        <circle cx=\"26\" cy=\"28\" r=\"14\" fill=\"#d03030\"/>\
        <rect x=\"30\" y=\"30\" width=\"22\" height=\"20\" fill=\"#2050c0\"/>\
        </svg>";

    /// A 64 x 64 image of two colours side by side, the middle `rim` columns halfway between them, and its labels.
    fn side_by_side(rim: usize) -> (Image, Vec<usize>, Vec<Option<Lab>>) {
        let (one, two) = ([200u8, 60, 60], [200u8, 80, 60]);
        let half = |a: u8, b: u8| ((a as u16 + b as u16) / 2) as u8;
        let (mut rgba, mut labels) = (Vec::new(), Vec::new());
        for _ in 0..64 {
            for x in 0..64 {
                let (rgb, label) = if x < 32 {
                    (one, 1)
                } else if x < 32 + rim {
                    ([half(one[0], two[0]), half(one[1], two[1]), half(one[2], two[2])], 2)
                } else {
                    (two, 3)
                };
                rgba.extend([rgb[0], rgb[1], rgb[2], 255]);
                labels.push(label);
            }
        }
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let centre = |l: usize| labels.iter().position(|&m| m == l).map(|i| image.lab[i]);
        let centres = vec![None, centre(1), centre(2), centre(3)];
        (image, labels, centres)
    }

    #[test]
    fn two_solid_fills_are_two_flat_colours() {
        let (image, labels, centres) = side_by_side(0);
        assert!(two_flat_colours(&image, &labels, 1, 3, &centres));
    }

    #[test]
    fn a_blurred_edges_rim_is_not_a_flat_colour() {
        let (image, labels, centres) = side_by_side(1);
        assert!(!two_flat_colours(&image, &labels, 1, 2, &centres));
    }

    #[test]
    fn a_ramp_is_not_two_flat_colours() {
        let rgba: Vec<u8> = (0..64 * 64)
            .flat_map(|i| [200, 60 + ((i % 64) / 3) as u8, 60, 255])
            .collect();
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let labels: Vec<usize> = (0..64 * 64).map(|i| if i % 64 < 32 { 1 } else { 2 }).collect();
        let centres = vec![None, Some(image.lab[8]), Some(image.lab[56])];
        assert!(!two_flat_colours(&image, &labels, 1, 2, &centres));
    }

    /// How many colours `segment` finds, besides the background, in two close blocks on white, with this much noise.
    fn colours_of_two_close_blocks(noise: u32) -> usize {
        let mut rng = Pcg32::new(1);
        let mut rgba = Vec::new();
        for y in 0..64 {
            for x in 0..64 {
                let rgb = match (x, y) {
                    (8..=31, 8..=55) => [200, 60, 60],
                    (32..=55, 8..=55) => [200, 80, 60],
                    _ => [245, 245, 245],
                };
                for c in rgb {
                    let wobble = if noise == 0 {
                        0
                    } else {
                        (rng.next_u32() % (2 * noise + 1)) as i32 - noise as i32
                    };
                    rgba.push((c + wobble).clamp(0, 255) as u8);
                }
                rgba.push(255);
            }
        }
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let sigma = noise_sigma(&image);
        let background = Background::fit(&image, sigma);
        let centres = segment(&image, &background, sigma, &TraceConfig::default()).centres;
        centres.len() - 1
    }

    #[test]
    fn close_flat_colours_stay_apart_only_without_noise() {
        assert_eq!(colours_of_two_close_blocks(0), 2);
        assert_eq!(colours_of_two_close_blocks(6), 1);
    }

    /// A white image with these `(x0, y0, x1, y1)` boxes in red, its scene, and each box as a part.
    fn boxes(boxes: &[(usize, usize, usize, usize)]) -> (Image, Vec<Part>) {
        let (w, h) = (64, 64);
        let inside = |x: usize, y: usize, b: &(usize, usize, usize, usize)| x >= b.0 && x < b.2 && y >= b.1 && y < b.3;
        let mut rgba = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let red = boxes.iter().any(|b| inside(x, y, b));
                rgba.extend(if red { [200, 40, 40, 255] } else { [255, 255, 255, 255] });
            }
        }
        let image = Image::from_rgba8(&rgba, w, h).unwrap();
        let parts = boxes
            .iter()
            .map(|b| Part {
                pixels: (0..w * h).filter(|&i| inside(i % w, i / w, b)).collect(),
                fill: [200, 40, 40],
                label: 1,
            })
            .collect();
        (image, parts)
    }

    fn left_after_fitting_the_first(boxes_: &[(usize, usize, usize, usize)]) -> usize {
        let (image, parts) = boxes(boxes_);
        let n = image.lab.len();
        let white = image.lab[0];
        let mut labels = vec![0; n];
        for part in &parts {
            for &i in &part.pixels {
                labels[i] = 1;
            }
        }
        let centre = image.lab[parts[0].pixels[0]];
        let scene = Scene {
            image: &image,
            labels,
            centres: vec![white, centre],
            background: vec![white; n],
        };
        let costs = Costs::new(64, 64, 12.0);
        let fit = scene.fit(&[&parts[0]], &[], &costs).unwrap();
        let whole = Part {
            pixels: parts.iter().flat_map(|p| p.pixels.iter().copied()).collect(),
            fill: [200, 40, 40],
            label: 1,
        };
        leftovers(&scene, &whole, &[], &[fit], &costs).len()
    }

    #[test]
    fn a_piece_its_parts_shape_left_out_gets_a_shape() {
        assert_eq!(left_after_fitting_the_first(&[(4, 4, 24, 24), (40, 40, 56, 56)]), 1);
    }

    #[test]
    fn a_sliver_its_parts_shape_left_out_does_not() {
        assert_eq!(left_after_fitting_the_first(&[(4, 4, 24, 24), (30, 40, 60, 41)]), 0);
    }

    /// The scene over an opaque background colour, as RGBA bytes.
    #[test]
    fn two_overlapping_parts_of_one_colour_are_cut_apart() {
        // one component of one colour: a square with a disk standing out of its corner
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <rect x=\"10\" y=\"10\" width=\"30\" height=\"30\" fill=\"#d03030\"/>\
            <circle cx=\"42\" cy=\"42\" r=\"12\" fill=\"#d03030\"/></svg>";
        let (clean, rgba) = photograph(svg, [245, 240, 230]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        assert_eq!(traced.shapes.len(), 2, "{}", traced.canonical());
        let recovered = layer_recovery(&clean, &traced);
        assert!(
            recovered.iter().all(|r| *r > 0.9),
            "{recovered:?}\n{}",
            traced.canonical()
        );
    }

    #[test]
    fn a_part_an_occluder_cuts_in_two_is_joined() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <rect x=\"6\" y=\"24\" width=\"52\" height=\"16\" fill=\"#2050c0\"/>\
            <rect x=\"26\" y=\"8\" width=\"12\" height=\"48\" fill=\"#e0a020\"/></svg>";
        let (clean, rgba) = photograph(svg, [245, 240, 230]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        assert_eq!(traced.shapes.len(), 2, "{}", traced.canonical());
        let recovered = layer_recovery(&clean, &traced);
        assert!(
            recovered.iter().all(|r| *r > 0.9),
            "{recovered:?}\n{}",
            traced.canonical()
        );
    }

    #[test]
    fn details_of_one_colour_are_not_joined_under_what_they_sit_on() {
        // two eyes on a face; and two marks on a band across a frame of the marks' colour
        for svg in [
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
                <circle cx=\"32\" cy=\"32\" r=\"26\" fill=\"#ffcc4d\"/>\
                <ellipse cx=\"22\" cy=\"28\" rx=\"4\" ry=\"7\" fill=\"#664500\"/>\
                <ellipse cx=\"42\" cy=\"28\" rx=\"4\" ry=\"7\" fill=\"#664500\"/></svg>",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
                <rect x=\"4\" y=\"8\" width=\"56\" height=\"48\" fill=\"#2060a0\"/>\
                <rect x=\"4\" y=\"20\" width=\"56\" height=\"24\" fill=\"#f0a040\"/>\
                <rect x=\"14\" y=\"26\" width=\"12\" height=\"12\" fill=\"#2060a0\"/>\
                <rect x=\"38\" y=\"26\" width=\"12\" height=\"12\" fill=\"#2060a0\"/></svg>",
        ] {
            let (clean, rgba) = photograph(svg, [245, 240, 230]);
            let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
            let recovered = layer_recovery(&clean, &traced);
            assert!(
                recovered.iter().all(|r| *r > 0.9),
                "{recovered:?}\n{}",
                traced.canonical()
            );
        }
    }

    #[test]
    fn a_tile_filling_the_border_is_a_shape_and_its_letter_too() {
        // a rounded tile reaching every edge but the corners, a white mark on it, over an off-white background
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <path d=\"M16 0L48 0Q64 0 64 16L64 48Q64 64 48 64L16 64Q0 64 0 48L0 16Q0 0 16 0Z\" fill=\"#2060a0\"/>\
            <rect x=\"22\" y=\"16\" width=\"20\" height=\"32\" fill=\"#ffffff\"/></svg>";
        let (clean, rgba) = photograph(svg, [244, 240, 236]);
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let background = Background::fit(&image, noise_sigma(&image));
        assert!(distance2(&background.at(0.0, 0.0), &oklab([244, 240, 236])) < 1e-6);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        let recovered = layer_recovery(&clean, &traced);
        assert!(
            recovered.iter().all(|r| *r > 0.9),
            "{recovered:?}\n{}",
            traced.canonical()
        );
    }

    #[test]
    fn a_hole_of_the_background_colour_stays_a_hole() {
        // four bars around a gap the background shows through
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <rect x=\"12\" y=\"12\" width=\"40\" height=\"10\" fill=\"#d03030\"/>\
            <rect x=\"12\" y=\"42\" width=\"40\" height=\"10\" fill=\"#d03030\"/>\
            <rect x=\"12\" y=\"12\" width=\"10\" height=\"40\" fill=\"#2050c0\"/>\
            <rect x=\"42\" y=\"12\" width=\"10\" height=\"40\" fill=\"#2050c0\"/></svg>";
        let (_, rgba) = photograph(svg, [245, 240, 230]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        let rendered = traced.render().to_rgba8();
        assert_eq!(rendered[(32 * 64 + 32) * 4 + 3], 0, "{}", traced.canonical());
    }

    #[test]
    fn a_light_part_beside_the_subject_is_not_the_background() {
        // a light grey plate under a brown pudding, on an off-white background within background_delta of the plate
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <ellipse cx=\"32\" cy=\"42\" rx=\"28\" ry=\"12\" fill=\"#e1e8ed\"/>\
            <rect x=\"14\" y=\"16\" width=\"36\" height=\"32\" fill=\"#8a4b38\"/></svg>";
        let (clean, rgba) = photograph(svg, [246, 244, 240]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        let rendered = traced.render().to_rgba8();
        // the plate rim, left of the pudding, is drawn
        assert_eq!(rendered[(44 * 64 + 10) * 4 + 3], 255, "{}", traced.canonical());
        assert!(
            layer_recovery(&clean, &traced).iter().all(|r| *r > 0.9),
            "{}",
            traced.canonical()
        );
    }

    #[test]
    fn a_light_body_far_from_the_background_needs_little_contact() {
        // a light grey body with one small eye on it, and nothing else: its boundary is nearly all background
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <ellipse cx=\"32\" cy=\"34\" rx=\"25\" ry=\"22\" fill=\"#e1e8ed\"/>\
            <circle cx=\"32\" cy=\"12\" r=\"5\" fill=\"#8a4b38\"/></svg>";
        let (_, rgba) = photograph(svg, [246, 244, 240]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        let rendered = traced.render().to_rgba8();
        assert_eq!(rendered[(40 * 64 + 32) * 4 + 3], 255, "{}", traced.canonical());
    }

    #[test]
    fn a_subject_on_the_bottom_edge_is_not_a_gradient_of_the_background() {
        // the sea along the bottom, with a boat on it, and a pale sky above: the corners are two colours
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <rect x=\"0\" y=\"42\" width=\"64\" height=\"22\" fill=\"#55acee\"/>\
            <rect x=\"20\" y=\"30\" width=\"24\" height=\"14\" fill=\"#8a4b38\"/></svg>";
        let sky = [220, 236, 224];
        let (_, rgba) = photograph(svg, sky);
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let background = Background::fit(&image, noise_sigma(&image));
        assert!(!background.gradient);
        assert!(distance2(&background.at(0.5, 0.5), &oklab(sky)) < 1e-4);
    }

    #[test]
    fn a_tile_on_a_gradient_background_keeps_the_gradient() {
        // the background shows only at the tile's cut corners, and goes from pale pink to pale cream down the image
        let mut rgba = Vec::with_capacity(64 * 64 * 4);
        for y in 0..64usize {
            for x in 0..64usize {
                let (dx, dy) = (x.min(63 - x), y.min(63 - y));
                let t = y as f64 / 63.0;
                let pixel = if dx + dy < 10 {
                    [
                        (226.0 + 24.0 * t) as u8,
                        (200.0 + 30.0 * t) as u8,
                        (216.0 - 10.0 * t) as u8,
                    ]
                } else {
                    [64, 138, 196]
                };
                rgba.extend_from_slice(&pixel);
                rgba.push(255);
            }
        }
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let background = Background::fit(&image, noise_sigma(&image));
        assert!(background.gradient);
        assert!(distance2(&background.at(0.0, 0.0), &oklab([226, 200, 216])).sqrt() < 0.02);
    }

    #[test]
    fn a_ring_keeps_its_hole() {
        // a disc with a disc cut out of it, the inner one winding the other way
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <path d=\"M56 32C56 45.3 45.3 56 32 56C18.7 56 8 45.3 8 32C8 18.7 18.7 8 32 8C45.3 8 56 18.7 56 32Z\
            M44 32C44 25.4 38.6 20 32 20C25.4 20 20 25.4 20 32C20 38.6 25.4 44 32 44C38.6 44 44 38.6 44 32Z\" \
            fill=\"#f0a020\"/></svg>";
        let (_, rgba) = photograph(svg, [244, 240, 236]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        let rendered = traced.render().to_rgba8();
        let alpha = |x: usize, y: usize| rendered[(y * 64 + x) * 4 + 3];
        assert_eq!(
            (alpha(32, 32), alpha(32, 14), alpha(2, 2)),
            (0, 255, 0),
            "{}",
            traced.canonical()
        );
    }

    #[test]
    fn a_tile_keeps_its_rounded_corners() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <path d=\"M12 0L52 0Q64 0 64 12L64 52Q64 64 52 64L12 64Q0 64 0 52L0 12Q0 0 12 0Z\" fill=\"#2060a0\"/></svg>";
        let (_, rgba) = photograph(svg, [244, 240, 236]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        let rendered = traced.render().to_rgba8();
        let alpha = |x: usize, y: usize| rendered[(y * 64 + x) * 4 + 3];
        assert_eq!(
            (alpha(32, 32), alpha(1, 1), alpha(62, 62)),
            (255, 0, 0),
            "{}",
            traced.canonical()
        );
    }

    #[test]
    fn a_speck_on_one_corner_does_not_hide_the_background() {
        // the tile of `a_tile_filling_the_border_is_a_shape_and_its_letter_too`, and a dark speck on one corner
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\">\
            <path d=\"M16 0L48 0Q64 0 64 16L64 48Q64 64 48 64L16 64Q0 64 0 48L0 16Q0 0 16 0Z\" fill=\"#2060a0\"/>\
            <rect x=\"22\" y=\"16\" width=\"20\" height=\"32\" fill=\"#ffffff\"/>\
            <rect x=\"0\" y=\"0\" width=\"3\" height=\"3\" fill=\"#404040\"/></svg>";
        let (_, rgba) = photograph(svg, [244, 240, 236]);
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let background = Background::fit(&image, noise_sigma(&image));
        assert!(distance2(&background.at(1.0, 1.0), &oklab([244, 240, 236])) < 1e-4);
    }

    fn photograph(svg: &str, background: [u8; 3]) -> (Document, Vec<u8>) {
        let clean = Document::parse(svg).unwrap();
        let (w, h) = (clean.width as usize, clean.height as usize);
        let mut canvas = crate::raster::Canvas::new(w, h);
        let mut frame = PathBuilder::new();
        frame.move_to((0.0, 0.0));
        frame.line_to((w as f64, 0.0));
        frame.line_to((w as f64, h as f64));
        frame.line_to((0.0, h as f64));
        frame.close();
        canvas.fill(
            &coverage(&frame.finish(), w, h),
            background.map(|c| c as f64 / 255.0),
            1.0,
        );
        for shape in &clean.shapes {
            canvas.fill(
                &shape.mask(clean.width, clean.height),
                shape.fill.map(|c| c as f64 / 255.0),
                shape.opacity,
            );
        }
        (clean, canvas.to_rgba8())
    }

    #[test]
    fn oklab_of_white_and_black() {
        let white = oklab([255, 255, 255]);
        assert!(
            (white[0] - 1.0).abs() < 1e-4 && white[1].abs() < 1e-4 && white[2].abs() < 1e-4,
            "{white:?}"
        );
        assert!(oklab([0, 0, 0])[0].abs() < 1e-12);
    }

    #[test]
    fn a_clean_scene_comes_back_as_its_parts_in_order() {
        let (clean, rgba) = photograph(SCENE, [245, 240, 230]);
        let traced = trace(&rgba, 64, 64, &TraceConfig::default()).unwrap();
        assert_eq!(traced.shapes.len(), 2, "{}", traced.canonical());
        assert!(
            matches!(traced.shapes[0].geometry, crate::copycat::Geometry::Circle { .. }),
            "{}",
            traced.canonical()
        );
        assert_eq!(traced.shapes[1].fill, [0x20, 0x50, 0xc0]);
        let recovered = layer_recovery(&clean, &traced);
        assert!(
            recovered.iter().all(|r| *r > 0.95),
            "{recovered:?}\n{}",
            traced.canonical()
        );
        // transparent where the background was
        assert_eq!(traced.render().to_rgba8()[3], 0);
    }

    #[test]
    fn the_part_that_covers_more_of_the_other_goes_below() {
        // a 5 x 5 mark on a 20 x 20 tile: the tile, completed, covers all of the mark; the mark, completed with the tile
        // free, sprawls over a 5 x 20 column -- 75 of the tile's pixels against the tile's 25 of its, yet a fifth of it
        let (w, h) = (20, 20);
        let image = Image::from_rgba8(&vec![255; w * h * 4], w, h).unwrap();
        let mark: Vec<usize> = (0..w * h)
            .filter(|i| (8..13).contains(&(i % w)) && (8..13).contains(&(i / w)))
            .collect();
        let tile: Vec<usize> = (0..w * h).filter(|i| !mark.contains(i)).collect();
        let parts = [tile, mark].map(|pixels| Part {
            pixels,
            fill: [0, 0, 0],
            label: 1,
        });
        let column: Vec<f64> = (0..w * h)
            .map(|i| if (8..13).contains(&(i % w)) { 1.0 } else { 0.0 })
            .collect();
        let completed = [Some(vec![1.0; w * h]), Some(column)];
        assert_eq!(order(&image, &parts, &completed), vec![0, 1]);
    }

    #[test]
    fn the_background_and_noise_are_estimated() {
        let (_, rgba) = photograph(SCENE, [245, 240, 230]);
        let image = Image::from_rgba8(&rgba, 64, 64).unwrap();
        let sigma = noise_sigma(&image);
        assert!(sigma <= 0.002 + 1e-12, "{sigma}");
        let background = Background::fit(&image, sigma);
        assert!(!background.gradient);
        assert!(distance2(&background.at(0.5, 0.5), &oklab([245, 240, 230])) < 1e-8);
    }
}

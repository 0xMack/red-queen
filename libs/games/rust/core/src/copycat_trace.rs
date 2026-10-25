//! Copy Cat's structured pipeline (docs/design/0022 Decisions 4-5, the ladder's step 2): an image in, a clean, layered,
//! transparent SVG out, every choice made by description length rather than by a threshold where one can be had.
//!
//! 1. **Noise**: σ, a robust (median) estimate from neighbouring pixels' OKLab differences -- edges are the tail.
//! 2. **Background**: fitted to the image border in OKLab, constant or a linear gradient (whichever is shorter to
//!    describe), its outliers dropped twice. It is a cluster of its own whose centre varies across the image, and what
//!    it explains is left transparent.
//! 3. **Palette**: k-means in OKLab from `colors` k-means++ seeds (`Pcg32`), then merges, closest pair first, while a
//!    merge is shorter -- a palette entry (24 bits) and the boundary between them saved, against the residual it adds
//!    (`ΔSSE / 2σ² ln 2` bits) -- or the two are within `merge_delta` of each other (a part's own shading, which a
//!    flat fill can't say anyway), or a cluster is within `background_delta` of the background (faint clutter).
//! 4. **Labels**: a 3 x 3 majority filter; then components too small (`min_area`) or too thin (an anti-aliased or
//!    blurred edge's in-between colour) are dissolved into whatever surrounds them.
//! 5. **Parts**: every connected component but the background is a part, filled with its label's interior mean (one
//!    palette entry per label); two of one colour join when one shape says them more briefly (a part an occluder cut in two).
//! 6. **Order**: for each pair of touching parts, the one that is shorter to describe when completed under the other
//!    goes below (`copycat_fit`'s free pixels); a topological sort, largest first, breaks cycles and ties.
//! 7. **Shapes**: each part fitted with everything above it free, so a hidden part completes itself; cut in two
//!    between notches of its outline while two shapes say it more briefly (two overlapping parts of one colour); one
//!    `<g>` per shape.

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

    /// Fitted to a 2-pixel border; outliers past 3 robust deviations dropped and refitted, twice.
    pub fn fit(image: &Image, sigma: f64) -> Background {
        let (w, h) = (image.width, image.height);
        let mut border: Vec<usize> = (0..w * h)
            .filter(|&i| {
                let (x, y) = (i % w, i / w);
                x < 2 || y < 2 || x + 2 >= w || y + 2 >= h
            })
            .collect();
        let mut fits = (
            Background::least_squares(image, &border, false),
            Background::least_squares(image, &border, true),
        );
        for _ in 0..2 {
            let flat = fits.0;
            let deviations: Vec<f64> = border
                .iter()
                .map(|&i| distance2(&image.lab[i], &flat.at_pixel(image, i)).sqrt())
                .collect();
            let cut = 3.0 * median(deviations.clone()).max(sigma) * 2.0;
            let kept: Vec<usize> = border
                .iter()
                .zip(&deviations)
                .filter(|(_, d)| **d <= cut)
                .map(|(i, _)| *i)
                .collect();
            if kept.len() < 8 {
                break;
            }
            border = kept;
            fits = (
                Background::least_squares(image, &border, false),
                Background::least_squares(image, &border, true),
            );
        }
        // the gradient's 6 more numbers (at about 12 bits each) against the residual it removes
        let sse = |b: &Background| {
            border
                .iter()
                .map(|&i| distance2(&image.lab[i], &b.at_pixel(image, i)))
                .sum::<f64>()
        };
        let saved = (sse(&fits.0) - sse(&fits.1)) / (2.0 * sigma * sigma * std::f64::consts::LN_2);
        if saved > 6.0 * 12.0 {
            fits.1
        } else {
            fits.0
        }
    }

    fn least_squares(image: &Image, pixels: &[usize], gradient: bool) -> Background {
        let mut coefficients = [[0.0; 3]; 3];
        let n = pixels.len().max(1) as f64;
        for (k, row) in coefficients.iter_mut().enumerate() {
            row[0] = pixels.iter().map(|&i| image.lab[i][k]).sum::<f64>() / n;
        }
        let flat = Background {
            coefficients,
            gradient: false,
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
        }
    }
}

// --- the palette --------------------------------------------------------------------------------------------------

/// Label 0 is the background; 1.. index `centres` (centres[0] is unused).
pub struct Segmentation {
    pub labels: Vec<usize>,
    pub centres: Vec<Lab>,
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
        let mut best: Option<(f64, usize, usize)> = None;
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
                if (delta < close || added * bits_per_sse < saved) && best.is_none_or(|(d, _, _)| delta < d) {
                    best = Some((delta, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else { break };
        for l in labels.iter_mut() {
            if *l == b {
                *l = a;
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
/// nearest pixel that stays (a breadth-first flood from what stays).
pub fn clean_labels(image: &Image, labels: &[usize], config: &TraceConfig) -> Vec<usize> {
    let mut labels = majority(labels, image.width, image.height);
    labels = majority(&labels, image.width, image.height);
    let (id, members) = components(image, &labels);
    let dissolve: Vec<bool> = members
        .iter()
        .map(|pixels| pixels.len() < config.min_area || eroded(image, &id, pixels, 1) * 4 < pixels.len())
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
/// cut again, while the halves are shorter to describe than the whole.
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
        return vec![whole];
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
        return vec![whole];
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
/// any of them could be above it). Of two touching parts, the one whose shape reaches further under the other goes
/// below -- a part completed under its occluders claims their pixels, one on top claims none. Kahn's sort takes the
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
        let a_first = if (claims[a][b] - claims[b][a]).abs() > 0.5 {
            claims[a][b] > claims[b][a]
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
    let scene = Scene {
        image: &image,
        labels: clean_labels(&image, &segmentation.labels, config),
        centres: segmentation.centres.clone(),
        background: (0..width * height).map(|i| background.at_pixel(&image, i)).collect(),
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
        for fit in shapes_of(&scene, &parts[k], &above, &costs) {
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

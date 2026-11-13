//! Chroma restored before tracing (docs/design/0022, v30). A JPEG usually keeps chroma at half resolution (4:2:0), and
//! a decoder fills it back in with libjpeg's triangle filter, so every colour edge is smeared over a couple of pixels
//! while the luma edge stays sharp. The tracer's structural choices -- a ring's hole, a hexagon's corners, a thin
//! stripe's colour -- tip over on that smear: on light corruption, the same image saved at 4:4:4 traced about 1.7 dB
//! better.
//!
//! Under flat colours chroma is, window by window, a linear function of luma (exactly so where two colours meet). So
//! the decoded chroma is regressed on luma degraded the way the chroma was (a 2x2 mean, then the triangle filter), and
//! the fit applied to the sharp luma: a guided filter whose guide is blurred to match. It runs only on chroma that
//! shows the upsampler's signature (`subsampled`), in proportion to how clearly: on an image that never went through
//! 4:2:0, the regression would smear isoluminant edges instead.

/// The window's radius (px): 5 x 5.
const RADIUS: usize = 2;
/// The regression's ridge (on luma's variance, 0..1 units): a window flat in luma keeps its own mean chroma.
const RIDGE: f64 = 1e-4;
/// `subsampled` below `FULL` restores fully, above `NONE` not at all, linearly between: 4:4:4 images measure about 1
/// (0.54 at the least, among the training sets' clean images), 4:2:0 at quality 85-90 about 0.25.
const FULL: f64 = 0.3;
const NONE: f64 = 0.5;
/// And chroma must be blurrier than luma (`blurrier` under 0.99, as two sharp steps tie at 1): a clean image whose edges happen to fall between pixel pairs
/// passes `subsampled` too (a 2 px stripe on even rows reads 0.36), but its chroma is as sharp as its luma. Every
/// training image the two tests together pass is JPEG: of the clean ones, none.
const BLURRIER: f64 = 0.99;

/// `rgba` (straight 8-bit RGBA, row-major) with its chroma restored, or unchanged if it doesn't look subsampled (or
/// either side is odd: 4:2:0 pairs pixels).
pub fn restore(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    if width % 2 == 1 || height % 2 == 1 || width < 8 || height < 8 || rgba.len() != width * height * 4 {
        return rgba.to_vec();
    }
    let n = width * height;
    let (mut y, mut cb, mut cr) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let [r, g, b] = [0, 1, 2].map(|k| rgba[4 * i + k] as f64 / 255.0);
        y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b;
        cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b;
    }
    let strength = ((NONE - subsampled(&cb, &cr, width, height)) / (NONE - FULL)).clamp(0.0, 1.0);
    if strength == 0.0 || blurrier(&y, &cb, &cr, width, height) >= BLURRIER {
        return rgba.to_vec();
    }
    let blurred = degrade(&y, width, height);
    let mean = |v: &[f64]| boxed(v, width, height);
    let my = mean(&blurred);
    let vy: Vec<f64> = mean(&blurred.iter().map(|v| v * v).collect::<Vec<_>>())
        .iter()
        .zip(&my)
        .map(|(s, m)| s - m * m)
        .collect();
    let restored = [&cb, &cr].map(|p| {
        let mp = mean(p);
        let cross = mean(&blurred.iter().zip(p.iter()).map(|(a, b)| a * b).collect::<Vec<_>>());
        let a: Vec<f64> = (0..n).map(|i| (cross[i] - my[i] * mp[i]) / (vy[i] + RIDGE)).collect();
        let b: Vec<f64> = (0..n).map(|i| mp[i] - a[i] * my[i]).collect();
        let (a, b) = (mean(&a), mean(&b));
        (0..n)
            .map(|i| strength * (a[i] * y[i] + b[i]) + (1.0 - strength) * p[i])
            .collect::<Vec<f64>>()
    });
    let mut out = rgba.to_vec();
    for i in 0..n {
        let (l, u, v) = (y[i], restored[0][i], restored[1][i]);
        let rgb = [l + 1.402 * v, l - 0.344136 * u - 0.714136 * v, l + 1.772 * u];
        for k in 0..3 {
            out[4 * i + k] = (rgb[k].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

/// How far chroma is from the output of libjpeg's 2x triangle upsampling, against the same test half a sample out of
/// phase: about 1 for chroma at full resolution, well under for 4:2:0. The filter makes outputs `2i` and `2i + 1` of
/// samples `d[i - 1]`, `d[i]`, `d[i + 1]` in fixed proportions, so `d[i]` read from either neighbouring pair agrees.
pub fn subsampled(cb: &[f64], cr: &[f64], width: usize, height: usize) -> f64 {
    let residual = |c: &[f64], across: bool, offset: usize| {
        let (n, m) = if across { (width, height) } else { (height, width) };
        let at = |i: usize, j: usize| if across { c[j * width + i] } else { c[i * width + j] };
        let mut sum = 0.0;
        for j in 0..m {
            let mut i = 1 + offset;
            while i + 3 < n {
                let r = 1.5 * (at(i, j) - at(i + 1, j)) - 0.5 * (at(i - 1, j) - at(i + 2, j));
                sum += r * r;
                i += 2;
            }
        }
        sum
    };
    let total = |offset: usize| -> f64 {
        [cb, cr]
            .iter()
            .flat_map(|c| [true, false].map(|across| residual(c, across, offset)))
            .sum()
    };
    total(1) / total(0).max(1e-18)
}

/// How much sharper chroma is than luma: per channel, the energy of 1 px differences against 2 px ones (a sharp
/// step is half, a smooth ramp less), chroma's mean over luma's. About 1 when chroma is at full resolution; under it
/// when chroma was smoothed and luma wasn't.
pub fn blurrier(y: &[f64], cb: &[f64], cr: &[f64], width: usize, height: usize) -> f64 {
    let sharpness = |v: &[f64]| {
        let (mut near, mut far) = (0.0, 0.0);
        for j in 0..height {
            for i in 0..width {
                let at = |x: usize, y: usize| v[y * width + x];
                if i + 1 < width {
                    near += (at(i + 1, j) - at(i, j)).powi(2);
                }
                if i + 2 < width {
                    far += (at(i + 2, j) - at(i, j)).powi(2);
                }
                if j + 1 < height {
                    near += (at(i, j + 1) - at(i, j)).powi(2);
                }
                if j + 2 < height {
                    far += (at(i, j + 2) - at(i, j)).powi(2);
                }
            }
        }
        near / far.max(1e-18)
    };
    (sharpness(cb) + sharpness(cr)) / 2.0 / sharpness(y).max(1e-18)
}

/// Luma as 4:2:0 would leave chroma: 2x2 means, then the triangle filter (3/4 the nearest sample, 1/4 the next).
fn degrade(v: &[f64], width: usize, height: usize) -> Vec<f64> {
    let (w2, h2) = (width / 2, height / 2);
    let half: Vec<f64> = (0..w2 * h2)
        .map(|k| {
            let (x, y) = (2 * (k % w2), 2 * (k / w2));
            (v[y * width + x] + v[y * width + x + 1] + v[(y + 1) * width + x] + v[(y + 1) * width + x + 1]) / 4.0
        })
        .collect();
    let up = |i: usize, n: usize| -> (usize, usize) {
        let near = i / 2;
        let next = if i.is_multiple_of(2) {
            near.saturating_sub(1)
        } else {
            (near + 1).min(n - 1)
        };
        (near, next)
    };
    // rows first, as the separable filter's order doesn't matter
    let rows: Vec<f64> = (0..height * w2)
        .map(|k| {
            let (x, y) = (k % w2, k / w2);
            let (a, b) = up(y, h2);
            0.75 * half[a * w2 + x] + 0.25 * half[b * w2 + x]
        })
        .collect();
    (0..width * height)
        .map(|k| {
            let (x, y) = (k % width, k / width);
            let (a, b) = up(x, w2);
            0.75 * rows[y * w2 + a] + 0.25 * rows[y * w2 + b]
        })
        .collect()
}

/// The mean over each pixel's (2 RADIUS + 1)-square window, edges repeated.
fn boxed(v: &[f64], width: usize, height: usize) -> Vec<f64> {
    let k = (2 * RADIUS + 1) as f64;
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    let mut across = vec![0.0; v.len()];
    for y in 0..height {
        for x in 0..width {
            across[y * width + x] = (-(RADIUS as isize)..=RADIUS as isize)
                .map(|d| v[y * width + clamp(x as isize + d, width)])
                .sum::<f64>();
        }
    }
    let mut out = vec![0.0; v.len()];
    for y in 0..height {
        for x in 0..width {
            out[y * width + x] = (-(RADIUS as isize)..=RADIUS as isize)
                .map(|d| across[clamp(y as isize + d, height) * width + x])
                .sum::<f64>()
                / (k * k);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 32 x 32 image: a dark red disc on pale green, as straight RGBA8.
    fn disc() -> Vec<u8> {
        (0..32 * 32)
            .flat_map(|i| {
                let (x, y) = ((i % 32) as f64 - 15.3, (i / 32) as f64 - 16.6);
                if x * x + y * y < 81.0 {
                    [170u8, 30, 40, 255]
                } else {
                    [150, 220, 150, 255]
                }
            })
            .collect()
    }

    fn chroma(rgba: &[u8]) -> (Vec<f64>, Vec<f64>) {
        rgba.chunks(4)
            .map(|p| {
                let [r, g, b] = [0, 1, 2].map(|k| p[k] as f64);
                (
                    -0.168736 * r - 0.331264 * g + 0.5 * b,
                    0.5 * r - 0.418688 * g - 0.081312 * b,
                )
            })
            .unzip()
    }

    /// `rgba`'s chroma through 4:2:0 and back (2x2 means, the triangle filter), luma kept.
    fn subsample(rgba: &[u8]) -> Vec<u8> {
        let (cb, cr) = chroma(rgba);
        let (cb, cr) = (degrade(&cb, 32, 32), degrade(&cr, 32, 32));
        let mut out = rgba.to_vec();
        for (i, p) in rgba.chunks(4).enumerate() {
            let [r, g, b] = [0, 1, 2].map(|k| p[k] as f64);
            let l = 0.299 * r + 0.587 * g + 0.114 * b;
            let rgb = [
                l + 1.402 * cr[i],
                l - 0.344136 * cb[i] - 0.714136 * cr[i],
                l + 1.772 * cb[i],
            ];
            for k in 0..3 {
                out[4 * i + k] = rgb[k].clamp(0.0, 255.0).round() as u8;
            }
        }
        out
    }

    #[test]
    fn full_resolution_chroma_is_left_alone() {
        let image = disc();
        let (cb, cr) = chroma(&image);
        assert!(subsampled(&cb, &cr, 32, 32) > NONE);
        assert!(restore(&image, 32, 32) == image);
    }

    #[test]
    fn edges_between_pixel_pairs_are_not_taken_for_subsampling() {
        // a 2 px stripe on even rows: every edge falls between a pair, as 4:2:0's would, but chroma is as sharp as luma
        let image: Vec<u8> = (0..32 * 32)
            .flat_map(|i| {
                if (12..14).contains(&(i / 32)) {
                    [220u8, 150, 120, 255]
                } else {
                    [60, 110, 30, 255]
                }
            })
            .collect();
        let (cb, cr) = chroma(&image);
        assert!(subsampled(&cb, &cr, 32, 32) < NONE);
        assert!(restore(&image, 32, 32) == image);
    }

    #[test]
    fn subsampled_chroma_is_sharpened_back_toward_the_original() {
        let (image, smeared) = (disc(), subsample(&disc()));
        let (cb, cr) = chroma(&smeared);
        assert!(subsampled(&cb, &cr, 32, 32) < FULL);
        let error = |a: &[u8]| -> f64 { a.iter().zip(&image).map(|(&p, &q)| (p as f64 - q as f64).powi(2)).sum() };
        let restored = restore(&smeared, 32, 32);
        assert!(
            error(&restored) < error(&smeared) / 4.0,
            "{} vs {}",
            error(&restored),
            error(&smeared)
        );
    }

    #[test]
    fn odd_sizes_are_left_alone() {
        let image = vec![128u8; 9 * 8 * 4];
        assert_eq!(restore(&image, 9, 8), image);
    }
}

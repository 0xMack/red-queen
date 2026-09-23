//! Randomness for learning (docs/design/0010 Decision 2): the games' specified PCG32, plus the draws a learner needs
//! on top of it -- uniform floats, normals, and independent *streams* per concern (exploration, weight init, episode
//! seeds, replay sampling), so adding a draw to one concern never shifts another's sequence.
//!
//! Everything here is integer arithmetic plus `libm`, so a seed gives the same numbers natively and in WASM.

use redqueen_games::pcg::Pcg32;

/// What a stream is for. Its discriminant is mixed into the seed, so each concern gets its own sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    /// Choosing actions: epsilon-greedy draws, sampling from a policy.
    Explore = 1,
    /// Network weight initialization.
    Init = 2,
    /// Which game (seed) each training episode is.
    Episodes = 3,
    /// Sampling minibatches from a replay buffer.
    Replay = 4,
    /// Synthetic data in tests and benchmarks.
    Data = 5,
}

/// SplitMix64's finalizer: spreads (seed, stream) over the whole 64-bit seed space.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Clone, Debug)]
pub struct Rng {
    pcg: Pcg32,
}

impl Rng {
    /// The `stream` sequence of run `seed`.
    pub fn new(seed: u64, stream: Stream) -> Rng {
        Rng {
            pcg: Pcg32::new(mix(seed ^ mix(stream as u64))),
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.pcg.next_u32()
    }

    /// Uniform in `0..n` (unbiased).
    pub fn below(&mut self, n: u32) -> u32 {
        self.pcg.bounded(n)
    }

    /// Uniform in `[lo, hi)`, from 53 random bits.
    pub fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(hi > lo, "empty range {lo}..{hi}");
        let span = hi - lo;
        let wide = ((self.next_u32() as u64) << 32) | self.next_u32() as u64;
        lo + wide % span // the modulo bias is < span / 2^64: irrelevant for seed pools of millions
    }

    /// Uniform in `[0, 1)`: 53 bits, so every value is exactly representable.
    pub fn uniform(&mut self) -> f64 {
        let hi = (self.next_u32() >> 5) as u64; // 27 bits
        let lo = (self.next_u32() >> 6) as u64; // 26 bits
        ((hi << 26) | lo) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by Box-Muller (one draw per call; the second value is discarded for simplicity).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform(); // (0, 1]: log is finite
        let u2 = self.uniform();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_independent_and_reproducible() {
        let draws = |seed, stream| {
            let mut rng = Rng::new(seed, stream);
            (0..5).map(|_| rng.next_u32()).collect::<Vec<_>>()
        };
        assert_eq!(draws(7, Stream::Explore), draws(7, Stream::Explore));
        assert_ne!(draws(7, Stream::Explore), draws(7, Stream::Init));
        assert_ne!(draws(7, Stream::Explore), draws(8, Stream::Explore));
    }

    #[test]
    fn uniform_and_normal_look_right() {
        let mut rng = Rng::new(1, Stream::Data);
        let n = 20_000;
        let (mut sum, mut sq) = (0.0, 0.0);
        for _ in 0..n {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
            let z = rng.normal();
            sum += z;
            sq += z * z;
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((sq / n as f64 - 1.0).abs() < 0.05);
        let x = rng.range_u64(100, 105);
        assert!((100..105).contains(&x));
    }
}

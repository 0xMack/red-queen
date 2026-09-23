//! The one PRNG every game uses (docs/design/0009 Decision 2), specified here so any future port can
//! reproduce a game from its seed bit for bit. Games used to seed Python's `random` (Mersenne
//! Twister), which only CPython reproduces; this is PCG32 (O'Neill, "PCG: A Family of Simple Fast
//! Space-Efficient Statistically Good Algorithms for Random Number Generation", 2014), XSH-RR
//! output, 64-bit state:
//!
//! - seeding: `state = 0; inc = (STREAM << 1) | 1; next(); state += seed; next()`, with the fixed
//!   stream `STREAM = 54` (the reference implementation's demo stream);
//! - `next_u32`: `old = state; state = old * 6364136223846793005 + inc` (wrapping), output
//!   `rotr32(((old >> 18) ^ old) >> 27, old >> 59)`;
//! - `bounded(n)`: rejection sampling, `threshold = (2^32 - n) % n`, draw until `r >= threshold`,
//!   return `r % n` -- unbiased, and exactly `pcg32_boundedrand_r` from the reference code.
//!
//! `libs/games/tests/reference_snake.py` carries an independent pure-Python copy that the tests
//! check this against.

pub const STREAM: u64 = 54;
const MULTIPLIER: u64 = 6364136223846793005;

#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: (STREAM << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `0..n`. Panics on `n == 0`.
    pub fn bounded(&mut self, n: u32) -> u32 {
        assert!(n > 0, "bounded(0)");
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_implementation() {
        // pcg32-demo's first outputs for pcg32_srandom_r(&rng, 42u, 54u).
        let mut rng = Pcg32::new(42);
        let got: Vec<u32> = (0..6).map(|_| rng.next_u32()).collect();
        assert_eq!(
            got,
            vec![0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]
        );
    }

    #[test]
    fn bounded_stays_in_range() {
        let mut rng = Pcg32::new(7);
        for n in [1u32, 2, 3, 7, 100, u32::MAX] {
            for _ in 0..1000 {
                assert!(rng.bounded(n) < n);
            }
        }
    }
}

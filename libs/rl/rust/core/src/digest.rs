//! Determinism checks and a benchmark kernel (docs/design/0010 Decision 2, Phase 0).
//!
//! A *digest* is a fixed training computation reduced to one hash of every bit it produced. The native build
//! (through PyO3) and the WASM build (in Node, and on `/dev/rl` in a browser) compute the same digests and compare
//! them with the one checked-in fixture (`libs/rl/tests/determinism.json`): equal digests mean a seed trains the same
//! agent in a job and in a reader's browser.

use crate::nn::{Activation, Adam, Mlp, Shape};
use crate::rng::{Rng, Stream};

/// FNV-1a over the exact bits of a sequence of floats.
#[derive(Clone, Copy, Debug)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    pub fn f64(&mut self, value: f64) {
        for byte in value.to_bits().to_le_bytes() {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }

    pub fn all(&mut self, values: &[f64]) {
        values.iter().for_each(|&v| self.f64(v));
    }

    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

/// A small regression problem trained with Adam: He init, ReLU and tanh layers, a transcendental target, MSE.
/// Touches every piece of arithmetic a learner uses.
pub fn training_digest(seed: u64, updates: u32) -> String {
    let shape = Shape::new(
        vec![8, 32, 32, 3],
        vec![Activation::Relu, Activation::Tanh, Activation::Linear],
    )
    .unwrap();
    let mut mlp = Mlp::init(shape, &mut Rng::new(seed, Stream::Init));
    let mut data = Rng::new(seed, Stream::Data);
    let rows = 64;
    let inputs: Vec<f64> = (0..rows * 8).map(|_| data.normal()).collect();
    let targets: Vec<f64> = (0..rows)
        .flat_map(|r| {
            let x = &inputs[r * 8..(r + 1) * 8];
            [libm::sin(x[0] + x[1]), libm::exp(-x[2] * x[2]), x[3] * x[4] - x[5]]
        })
        .collect();
    let mut adam = Adam::new(mlp.params.len(), 1e-3);
    let mut hash = Fnv::default();
    for _ in 0..updates {
        let cache = mlp.forward_batch(&inputs, rows);
        let grad: Vec<f64> = cache
            .outputs()
            .iter()
            .zip(&targets)
            .map(|(y, t)| 2.0 * (y - t) / (rows * 3) as f64)
            .collect();
        let grads = mlp.backward(&cache, &grad);
        adam.step(&mut mlp.params, &grads);
        hash.f64(cache.outputs()[0]);
    }
    hash.all(&mlp.params);
    hash.hex()
}

/// One benchmark kernel: `iterations` training updates (forward + backward + Adam) on a `batch` of random inputs
/// for `shape`. Timed by the caller (`std::time` doesn't exist in WASM); returns a checksum so none of it is
/// optimized away.
pub fn bench_updates(shape: &Shape, batch: usize, iterations: u32, seed: u64) -> f64 {
    let mut mlp = Mlp::init(shape.clone(), &mut Rng::new(seed, Stream::Init));
    let mut data = Rng::new(seed, Stream::Data);
    let inputs: Vec<f64> = (0..batch * shape.inputs()).map(|_| data.uniform()).collect();
    let upstream: Vec<f64> = (0..batch * shape.outputs()).map(|_| data.normal() * 1e-3).collect();
    let mut adam = Adam::new(mlp.params.len(), 1e-4);
    for _ in 0..iterations {
        let cache = mlp.forward_batch(&inputs, batch);
        let grads = mlp.backward(&cache, &upstream);
        adam.step(&mut mlp.params, &grads);
    }
    mlp.params.iter().sum()
}

/// `iterations` single-observation forward passes -- the cost of choosing one action.
pub fn bench_forwards(shape: &Shape, iterations: u32, seed: u64) -> f64 {
    let mlp = Mlp::init(shape.clone(), &mut Rng::new(seed, Stream::Init));
    let mut data = Rng::new(seed, Stream::Data);
    let observation: Vec<f64> = (0..shape.inputs()).map(|_| data.uniform()).collect();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        checksum += mlp.forward(&observation)[0];
    }
    checksum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests_are_stable_and_seed_dependent() {
        assert_eq!(training_digest(1, 20), training_digest(1, 20));
        assert_ne!(training_digest(1, 20), training_digest(2, 20));
        assert_ne!(training_digest(1, 20), training_digest(1, 21));
    }
}

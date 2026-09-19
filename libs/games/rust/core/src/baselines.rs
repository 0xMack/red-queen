//! Fixed reference policies (docs/design/0007: a ranking says nothing without baselines), ported from
//! `games/baselines.py`. Here so the evaluation job (via PyO3) and the browser (via WASM) run the same
//! code.

use crate::pcg::Pcg32;

/// `snake/features.v1+relative3.v1`, uniformly random turn every step, reproducible per seed.
pub struct SnakeRandom {
    rng: Pcg32,
}

impl SnakeRandom {
    pub fn new(seed: u64) -> Self {
        SnakeRandom { rng: Pcg32::new(seed) }
    }

    pub fn decide(&mut self, _observation: &[f64]) -> i32 {
        self.rng.bounded(3) as i32 - 1
    }
}

/// `snake/features.v1+relative3.v1`: turn toward the food unless that's immediately fatal; otherwise
/// any safe move; otherwise straight. Reads danger[0..3] (straight/left/right), heading one-hot[3..7]
/// (RIGHT/DOWN/LEFT/UP), food left/right/up/down[7..11].
pub fn snake_greedy(observation: &[f64]) -> i32 {
    let danger = |action: i32| match action {
        0 => observation[0],
        -1 => observation[1],
        _ => observation[2],
    };
    let heading = observation[3..7].iter().position(|&v| v == 1.0).unwrap_or(0) as i32;
    let (food_left, food_right, food_up, food_down) = (observation[7], observation[8], observation[9], observation[10]);
    let wants = |direction: i32| match direction.rem_euclid(4) {
        0 => food_right,
        1 => food_down,
        2 => food_left,
        _ => food_up,
    };
    for action in [0, -1, 1] {
        if danger(action) == 0.0 && wants(heading + action) != 0.0 {
            return action;
        }
    }
    for action in [0, -1, 1] {
        if danger(action) == 0.0 {
            return action;
        }
    }
    0
}

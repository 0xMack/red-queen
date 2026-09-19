//! ReachTarget1D, ported from `libs/games/src/games/reach1d.py` (see its docstring): drive a point
//! with (position, velocity) to a target by choosing an acceleration each step. Only +, -, *, abs,
//! min and max on f64 -- all exactly specified by IEEE 754 -- so this matches the Python original bit
//! for bit on every platform, WASM included. (A future physics game using sin/exp would need the
//! `libm` crate on both targets for that, docs/design/0009.)

#[derive(Clone, Debug)]
pub struct Reach1D {
    pub target: f64,
    pub start_position: f64,
    pub start_velocity: f64,
    pub dt: f64,
    pub max_acceleration: f64,
    pub damping: f64,
    pub position: f64,
    pub velocity: f64,
}

impl Reach1D {
    pub fn new(target: f64, start_position: f64, start_velocity: f64, dt: f64, max_acceleration: f64, damping: f64) -> Reach1D {
        Reach1D { target, start_position, start_velocity, dt, max_acceleration, damping, position: start_position, velocity: start_velocity }
    }

    pub fn reset(&mut self) -> (f64, f64) {
        self.position = self.start_position;
        self.velocity = self.start_velocity;
        self.observation()
    }

    /// `action` is clamped to [-1, 1] and scaled by `max_acceleration`. Returns (observation, reward);
    /// the task never ends on its own (episodes are cut by the caller's step budget).
    pub fn step(&mut self, action: f64) -> ((f64, f64), f64) {
        // max(-1, min(1, action)) exactly as Python evaluates it (NaN included: Python's min/max keep
        // the first argument when a comparison with NaN is false).
        let clamped = if action < 1.0 { action } else { 1.0 };
        let clamped = if -1.0 < clamped { clamped } else { -1.0 };
        let acceleration = clamped * self.max_acceleration;
        self.velocity = (self.velocity + acceleration * self.dt) * self.damping;
        self.position += self.velocity * self.dt;
        let reward = -(self.position - self.target).abs();
        (self.observation(), reward)
    }

    /// Position relative to the target, and velocity.
    pub fn observation(&self) -> (f64, f64) {
        (self.position - self.target, self.velocity)
    }
}

//! Red Queen's reinforcement-learning core (docs/design/0010): environments as a trait, small networks with explicit
//! backprop, optimizers, agents and the training loop -- run by training jobs through PyO3 (`rust/python`) and by
//! Learn's live demos as WebAssembly (`rust/wasm`). Never names a game: `redqueen-rl-envs` adapts the games crate.

pub mod agent;
pub mod digest;
pub mod dqn;
pub mod env;
pub mod nn;
pub mod rng;
pub mod tabular;

//! Red Queen's game core (docs/design/0009 Decision 2): rules, observers, action adapters and
//! baselines, written once and run both by training/evaluation (Python, through `rust/python`) and
//! by visitors' browsers (WebAssembly, through `rust/wasm`). Pure Rust, no I/O; the one dependency is `libm`
//! (pure-Rust maths, so a transcendental gives the same bits natively and in WASM).

pub mod bandit;
pub mod baselines;
pub mod checkers;
pub mod checkers_strategies;
pub mod copycat;
pub mod copycat_climb;
pub mod nets;
pub mod pcg;
pub mod raster;
pub mod reach1d;
pub mod snake;

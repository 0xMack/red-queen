//! Red Queen's game core (docs/design/0009 Decision 2): rules, observers, action adapters and
//! baselines, written once and run both by training/evaluation (Python, through `rust/python`) and
//! by visitors' browsers (WebAssembly, through `rust/wasm`). Pure Rust, no dependencies, no I/O.

pub mod baselines;
pub mod pcg;
pub mod snake;

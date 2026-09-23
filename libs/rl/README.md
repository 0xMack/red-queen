# rl

Reinforcement learning from scratch ([docs/design/0010](../../docs/design/0010-reinforcement-learning.md)): one
agent that improves from its own experience, next to `evolve`'s populations. The ladder is tabular Q-learning →
DQN → policy gradients → Checkers self-play; each rung lands with a tracked experiment and a Learn chapter.

**Phase 0 (this):** the foundation, no learning yet — the environments behind a trait, networks with backprop,
Adam, the training loop, a random agent, and the checks that everything is deterministic across targets.

## Layout

A mixed Rust/Python package, like `libs/games`:

- `rust/core` (`redqueen-rl`) — the learning core. **Never names a game.**
  - `env.rs`: the `Env` trait (`reset(seed)`, `step(action)`, `score()`, observation size, action space), and
    `play()`, the evaluation loop `jobs/evaluate.py` also runs.
  - `nn.rs`: a batched MLP (`linear`/`relu`/`tanh` layers) with explicit backprop, and Adam. Parameters are one
    flat vector in `evolve.WeightVector`'s layout.
  - `agent.rs`: the `Agent` trait, `Params` (hyperparameters by name; unknown names are errors), `RandomAgent`,
    and `Trainer`, which advances training by a budget of environment steps (one *iteration* of an RL run).
  - `rng.rs`: the games' PCG32 plus uniform/normal draws and one independent stream per concern (exploration,
    init, episode seeds, replay, data).
  - `digest.rs`: determinism digests and the benchmark kernels.
- `rust/envs` (`redqueen-rl-envs`) — adapters from the games crate: Snake (any native observer, `relative3.v1`
  actions, the game's score) and Reach1D (continuous acceleration; game `seed` draws the target), plus the games'
  baselines as policies. Shared by both binding crates.
- `rust/python` → `rl._native` (PyO3). `rust/wasm` → `apps/frontend/app/wasm/rl` (built by `build-wasm.py`).
- `src/rl` — the Python face.
- `tests/` — parity against `libs/autodiff` (`reference_nn.py`), the adapters against `jobs/evaluate.py`'s scores,
  and the determinism fixture (`determinism.json`) that the native build, Node and the browser all check against.

## Determinism

Transcendentals go through the `libm` crate and every arithmetic order is fixed, so a seed produces bit-identical
training natively and in WASM (Decision 2). `determinism.json` holds digests computed natively; `tests/` checks the
native build against it, `apps/frontend/scripts/check-rl-determinism.mjs` checks the WASM build in Node (CI), and
`/dev/rl` checks it in a browser. Regenerate the fixture only for an intended change:
`uv run python libs/rl/tests/make_determinism_fixture.py`.

## Usage

```python
import rl

trainer = rl.Trainer("random", "snake/features.v1+relative3.v1", seed=0)
stats = trainer.train(10_000)         # one iteration: episodes finished, entropy, extras, totals
trainer.evaluate([20_000, 20_001], 1000)  # the greedy policy on given games
```

A run: `uv run python jobs/rl_run.py --algo random --iterations 5` (records to telemetry like every job).

After changing `rust/`, rebuild the browser copy: `uv run python libs/rl/build-wasm.py`
(`tests/test_wasm_build.py` fails until you do).

# 0010 — Reinforcement learning, from a Q-table to PPO and self-play

Status: **Phases 0-4 implemented** (see "Implementation notes"); Phase 5 is a stretch. Each phase lands as its own PR,
with its measured results written back into this doc and into the Learn section.
Relates to: [0003](0003-algorithm-landscape-and-roadmap.md) (gradient-based RL as a *sibling* of the evolution
machinery), [0004](0004-small-transformer-from-scratch.md) (`libs/autodiff`, the gradient oracle here),
[0007](0007-representations-leaderboards-and-tradeoffs.md) (interfaces, protocols, measured training cost — all
reused), [0008](0008-neat-and-tracked-comparisons.md) (tracked experiments: arms × seeds, permutation tests),
[0009](0009-client-side-inference-at-scale.md) (the Rust core → PyO3 + WASM convention, model packages).

## Context

Every trained policy so far was *evolved*: a population, a fitness, selection. The project's next paradigm is
reinforcement learning — one agent that improves from its own experience, step by step, with a learning rule
instead of selection. Doc 0003 already placed it: gradient-based RL shares the environment interface and the
telemetry with `evolve`, and nothing else; it gets its own loop, not a `Population` subclass.

The goal is educational as much as competitive, so the plan is a **ladder**: start with the simplest algorithm
that can possibly work, then add one idea per rung, measure what that idea buys, and write it up in Learn with a
live, in-browser demo. Decided with the user (2026-09-23):

| Question | Decision |
|---|---|
| Ladder | **Tabular Q-learning → DQN → policy gradients** (REINFORCE → baseline → A2C → PPO) |
| Where learning runs | **The Rust core**, bound to Python (jobs, telemetry, leaderboards) and compiled to WASM (live training in Learn). A pure-Python version on `libs/autodiff` is the gradient oracle. |
| Games | **Reach1D** as a sanity check on every rung, **Snake** as the main benchmark, **Checkers self-play** late |
| Learn | **A chapter per family**: Q-learning, DQN, policy gradients, self-play |
| Browser training | **Demos only.** Live training in Learn shows *how* learning works, for models small enough to learn visibly; each demo shape is committed only after Phase 0's in-browser benchmark says it fits. Leaderboard models train in jobs. |
| Browser inference | **Model packages** (doc 0009), with graceful, device-based restrictions and fallbacks to lighter models; some models may not run in a browser at all, and we'd like to find those limits rather than avoid them. |
| Follow-ups | the `libm` dependency is fine; new champion formats load through `modelpack` |

### What runs where

| Layer | Runs | Why |
|---|---|---|
| **Rust core** (`libs/rl/rust/core`, plus the existing games core) | the inner loop: environments, network forward/backward, Adam, replay, every algorithm's update rule, exploration and sampling, seeded randomness | millions of steps per run (the Python forward pass was 87% of a Snake generation); one implementation for jobs and demos; one seed, one run, everywhere |
| **Python** (`libs/rl/src/rl`, `jobs/rl_*.py`, `modelpack`) | orchestration: configs, jobs and experiments, telemetry and checkpoints, evaluation and leaderboards, ONNX export; the `libs/autodiff` oracles (tests only) | everything around a run is already Python; it calls Rust once per training *iteration*, not per step |
| **WASM** (the Rust core, compiled) | Learn's live-training demos; game rules | a demo trains in front of the reader |
| **ONNX Runtime Web** (model packages) | *inference* of every trained champion | one path for every model family, with doc 0009's device matching, variants, parity and explanations -- not the Rust nets |
| **TypeScript** | UI, the workers driving WASM/ONNX, device probing and fallback choice | |

**Every live demo has a recorded fallback:** when the device benchmark says live training is too slow (or the
device is a phone), the page replays a *real recorded training run* -- its curve and policy snapshots from
telemetry -- in the same visualization. Inference keeps 0009's behaviour: a model the device can't run says why
and offers a lighter one.

What there is to beat, on the `snake.score.v2` leaderboard (200 held-out games, 1,000-step cap, 10×10 board):

| Entrant | Held-out score | Training env steps | Active training time |
|---|---|---|---|
| NEAT, 28 hidden nodes (features.v1) | **37.95** | 409 M | 41 min |
| NEAT, 11 hidden nodes (features.v1) | 20.25 | 11.8 M | 76 s |
| Greedy heuristic (baseline) | 17.89 | – | – |
| Neuroevolution 11→16→3, lexicase | 17.62 | 48.8 M | 16 min |

Every run already records `env_steps` (doc 0007), so the headline comparison for RL isn't only "higher score" —
it's **score per environment step**. Evolution threw 409 M steps at its best champion; sample efficiency is
where RL is supposed to win, and now it can be measured on the same axis.

## Decision 1: a new mixed Rust/Python package, `libs/rl`, with a game-agnostic core

```
libs/rl/
  rust/core/     redqueen-rl: Env trait, MLP + backprop + Adam, replay buffer, the algorithms. Pure Rust.
  rust/python/   PyO3 -> rl._native: trainers, plus Env adapters for the games crate's Snake / Reach1D / Checkers
  rust/wasm/     wasm-bindgen -> apps/frontend/app/wasm/rl: the same trainers + adapters, for live training
  src/rl/        Python face: configs (pydantic), trainer wrappers, policy export, the wire format
  tests/         reference_*.py oracles on libs/autodiff; parity and learning tests
  build-wasm.py  like libs/games's, with its own source-hash test
```

- **Why its own package, not more of `libs/games`.** An RL algorithm isn't a game, just as `evolve` isn't. One
  package per *domain* (not per algorithm — the package-granularity rule): every RL algorithm lives in `libs/rl`.
- **The core never names a game.** `redqueen-rl` defines `trait Env { fn reset(&mut self, seed: u64) -> Obs;
  fn step(&mut self, action: Action) -> (Obs, f64, bool); fn obs_size(); fn action_space(); }`. The binding crates
  implement it for `redqueen-games`' types (path dependency) — integration glue in the bindings, the same
  direction as `jobs/` wiring `evolve` to `telemetry`. A new game needs an adapter, not a core change.
- **Why Rust and not `libs/autodiff`.** Two reasons: live training in the browser (Learn's demos train in front of
  the reader, which Python can't do without Pyodide, retired in 0009), and speed (a numpy autodiff graph per
  minibatch is slow for millions of steps). The networks involved are small (≤ a few thousand weights), so a
  hand-written MLP with explicit backprop is a few hundred lines, not a framework.
- **`libs/autodiff` stays the reference.** `tests/reference_*.py` re-implement each update (TD target, DQN loss,
  policy-gradient loss) on `autodiff.Tensor`; the Rust gradients must match to ~1e-12 on random batches. That's the
  project's existing pattern (`reference_snake.py`, `reference_checkers_strategies.py`) applied to learning.

## Decision 2: determinism — one seed, one training run, in Python and in the browser

Evolution's parity story was "the Rust core is the only implementation". Training adds a wrinkle: a learning run
is chaotic, so a one-ulp difference in `exp`/`tanh` between native and WASM builds (platform libm vs.
compiler-builtins) would make the "same" run diverge after a few thousand updates.

- **The RL core does its own transcendental math** through the pure-Rust `libm` crate (its one dependency,
  accepted 2026-09-23): identical bits on every target, so a seed trains
  the same agent in a job and in a reader's browser. A test trains N updates in both the native and the WASM
  build (via `wasm-bindgen-test` or Node) and compares every weight exactly.
- **Randomness is the games' PCG32** (ε-greedy draws, replay sampling, weight init, action sampling), one stream
  per concern, seeded from the run seed. Python's `random` never touches a training run.
- The Python oracle is compared with a tolerance, not exactly: it uses the system's `math.exp`, and it only
  checks gradients, never whole runs.
- Inference is unaffected: a trained policy is exported as a model package (Decision 5) and plays through ONNX
  Runtime like every other champion.

## Decision 3: what an RL run looks like to telemetry — no schema change

`GenerationStats` was built generic on purpose (doc 0002, `extras` in 0008). An RL run maps onto it as:

| Field | For an RL run |
|---|---|
| `generation` | one **training iteration**: a fixed budget of environment steps (e.g. 10,000), not a population |
| `best/mean/worst_fitness` | max / mean / min **episode return** among episodes finished during the iteration |
| `diversity` | policy **entropy** (mean, nats) — how undecided the agent still is; ε-greedy methods report the entropy of their ε-mixed policy |
| `champion_ref` | a snapshot of the current policy (the wire format of Decision 5) |
| `held_out_score` | unchanged: the greedy policy's mean score on `MONITOR_SEEDS`, every N iterations |
| `extras` | per algorithm: `epsilon`, `td_loss`, `q_mean`, `grad_norm`, `value_loss`, `kl`, `clip_fraction`, `env_steps`, `updates`, … |

- Jobs use `run_context.recorded_run()` and `TrainingCostMeter` unchanged, so pause/resume, the cost block and
  the `failed` status come for free. `config.representation` is the algorithm (`q_learning`, `dqn`,
  `reinforce`, `a2c`, `ppo`, `td_lambda`); `config.interface` is the usual interface id.
- RL policies act through the **same interfaces**: a Q-network's outputs are 3 action values, and `relative3.v1`'s
  argmax already turns them into a move. A policy-gradient net's outputs are logits: sampled during training,
  argmax at evaluation (the leaderboard protocol stays deterministic). No new interface is needed until an
  observer is.
- Training seeds: RL naturally uses **fresh games every episode** (`--seeds resample`), drawn from the training
  pool, never the held-out or monitor seeds.

## Decision 4: every rung is a tracked experiment

Each rung ships with a `jobs/rl_experiment.py` comparison — arms × ≥5 seeds, tagged runs, exact permutation test
(doc 0008's rules) — and the comparison always includes the previous rung, so each new idea is measured against
exactly what it changed. Reported for every arm:

1. held-out score (`snake.score.v2`, the leaderboard protocol),
2. **held-out score vs. environment steps** — the sample-efficiency curve, plotted against the evolved champions'
   curves (their runs log `env_steps` in the cost block; per-generation step counts come from population × seeds
   × episode length),
3. wall time and CPU time, and the learning curve's variance across seeds (RL is famously seed-sensitive — the
   Learn chapters should *show* that, not hide it).

The best untagged run of each algorithm is promoted to the leaderboard through the existing path (`evaluate.py`,
`publish_models.py`); tagged experiment runs stay off it, as today.

## Decision 5: the wire format and model packages

Two new champion kinds, loaded and exported by `modelpack` (decided 2026-09-23: it already depends on every
network kind, so `evolve` never has to know about `rl`):

- `{"type": "qtable", "observer": "features.v1", "bits": 11, "values": [[...3...] × 2048]}` — tabular. The ONNX
  graph is `index = obs · [1, 2, 4, …]` → `Gather` → the 3 action values; argmax is the interface's job, as now.
  Only possible for a binary observer; the exporter refuses anything else.
- `{"type": "mlp", "layer_sizes": [...], "activations": ["relu", "relu", "linear"], "weights": [...]}` — DQN
  and policy-gradient nets (ReLU hidden layers, linear outputs — unlike `WeightVector`'s tanh-everywhere). The
  core's `nets.rs` gains the matching forward pass (per-layer activation), so Checkers can search with one
  (Phase 4) exactly like it searches with an evolved evaluator today.

## The ladder

Each phase lists what it builds, the Learn content, the experiments it runs, and when it's done. The numbers
predicted below are hypotheses to test, not promises.

### Phase 0 — foundation (no learning yet)

- `libs/rl` skeleton: Cargo workspace (path dependency on `libs/games/rust/core`), `uv` workspace member, PyO3 and
  wasm crates, `build-wasm.py` + source-hash test, CI (the python job builds it; the rust job adds the workspace).
- `Env` trait + adapters for Snake (any native observer, `relative3.v1` actions) and Reach1D; a **random agent**
  and the existing greedy baseline running through the trait, reproducing `evaluate.py`'s baseline scores exactly
  (the adapter is right when the numbers are).
- `libm`-based math, PCG32 streams, the native-vs-WASM determinism test harness.
- `jobs/rl_run.py` skeleton on `recorded_run()`; `runMeta.ts` labels for the new representations.
- Done when: a random agent's run appears on `/runs` with the Decision 3 fields, and the determinism test passes.

### Phase 1 — tabular Q-learning (chapter: "Reinforcement Learning: Q-learning")

- **Algorithm:** one-step Q-learning, `Q(s,a) ← Q(s,a) + α (r + γ max Q(s′,·) − Q(s,a))`, ε-greedy with a decay
  schedule, optional optimistic initial values. State = the 11 binary features of `features.v1` → 2,048 rows × 3
  actions. Reach1D with a discretized state (position/velocity bins) and actions (−1, 0, +1).
- **Iterations inside the rung** (each an experiment arm): ε schedule (constant vs. decaying), γ (0.9 / 0.95 /
  0.99), α, optimistic init, then **SARSA vs. Q-learning** (on- vs. off-policy — the classic cliff-walking lesson,
  seen on Snake), then **n-step returns**.
- **Hypothesis:** tabular Q on `features.v1` beats greedy (17.9) in a few hundred thousand steps — three orders of
  magnitude fewer than neuroevolution — but plateaus below NEAT (38): the 11 features alias states the 28-node
  NEAT net distinguishes by *combining* them, and a table can't generalize between rows at all. How much of the
  2,048-state table is ever visited is itself a result worth showing.
- **Learn:** what an MDP is (state, action, reward, return, discount); the Bellman equation; exploration vs.
  exploitation. Live demo: a Snake agent learning from scratch in the reader's browser in ~30 s, the Q-table
  as a heatmap filling in as states are visited, ε/α/γ sliders, the learning curve next to evolution's
  (steps on the x-axis).
- **Done when:** the Q-learning champion is on the leaderboard with its env-step cost, the chapter is live, and the
  arms' results are written back here.

### Phase 2 — DQN (chapter: "Deep Q-Networks")

- **Algorithm:** a Q-*network* (MLP, ReLU, Adam, Huber loss) replacing the table, then the pieces that make it
  stable, **added one at a time as arms** so each one's effect is measured: (1) naive online Q-learning with a
  network (expected to be unstable — worth showing), (2) + experience replay, (3) + target network, (4) Double
  DQN, (5) Dueling heads, (6) n-step targets, (7) prioritized replay.
- **Observers:** `features.v1` first (direct comparison with the table: what does generalization buy?), then
  `egocentric.v1` (27 inputs, which evolution couldn't use better than features.v1 — doc 0007's null result) and
  `grid-flat.v1` (100 inputs, evolution's representation ceiling). **The interesting question of this phase:** does
  a gradient learner make the richer observations pay off where evolution couldn't?
- **Oracle:** `reference_dqn.py` on `libs/autodiff` — the same minibatch through both, gradients equal to ~1e-12.
- **Learn:** function approximation, why naive deep Q-learning diverges (the deadly triad), replay and target
  networks as the fixes, overestimation and Double DQN. Live demo: DQN training in the browser on a small net
  (11→64→64→3), a toggle that switches replay/target off to watch it fall apart.
- **Done when:** each stability arm has a 5-seed result, the best DQN is on the leaderboard, and the observer
  question has an answer (positive or null — 0007's egocentric result was a null and was just as useful).

### Phase 3 — policy gradients (chapter: "Policy Gradients")

- **Algorithm ladder:** REINFORCE (Monte-Carlo returns) → + a learned baseline (variance reduction) → A2C
  (bootstrapped critic, GAE(λ), parallel environments) → PPO (clipped objective, minibatch epochs, entropy
  bonus, advantage normalization). Each rung an arm against the previous one.
- **Reach1D becomes interesting here:** a Gaussian policy over the *continuous* acceleration — something neither
  the table nor DQN can express without discretizing. Snake uses a softmax over the 3 actions.
- **Bridge to evolution:** the policy network has the same shape as neuroevolution's `WeightVector` (11→16→3), so
  "gradient vs. evolution on the *same* architecture" is a direct, controlled comparison; and **evolution
  strategies** (OpenAI-ES — a population estimate of the same gradient) is a natural extra arm, using the existing
  `evolve` machinery, that shows the two paradigms meeting.
- **Oracle:** `reference_pg.py` — log-prob gradients, the clipped surrogate, GAE, against autodiff.
- **Learn:** the policy-gradient theorem, the variance problem and baselines, actor-critic, why PPO clips. Live
  demo: PPO on Reach1D (continuous control, learns in seconds) and on Snake, with the policy's action
  probabilities drawn on the board as it plays.
- **Done when:** PPO has a leaderboard entry and the gradient-vs-evolution-on-the-same-net comparison is written up.

### Phase 4 — Checkers by self-play (chapter: "Learning by Self-Play")

- **Algorithm:** **TD(λ) on afterstate values**, TD-Gammon style: a value network over `board32.v1` scores
  positions from the mover's side, learns from its own games against itself, and plays by searching over it —
  which means it plugs straight into the existing `evaluator` strategy (`nets.rs` + `checkers_strategies.rs`,
  any search depth), the versus leaderboard (`checkers.versus.v1`) and the Checkers page, with no new game code.
  Then an arm with **opponent pools** (past selves, the hall-of-fame idea from `checkers_training.py`) against
  pure self-play.
- **Stretch:** PPO self-play with legal-move masking (a policy over moves rather than a value over positions).
- **What to beat:** `material-4` (0.90 points per game) and the best evolved evaluator (0.76).
- **Learn:** self-play, afterstates, why TD(λ) worked for backgammon and what changes for a deterministic game.
- **Done when:** a self-play evaluator is on the versus leaderboard, compared at equal search depth with the
  evolved ones and material search.

### Phase 5 — where RL and evolution meet (stretch, not committed)

Doc 0003's bridges, in order of fit: **population-based training** (a population of DQN/PPO learners, with
evolution's exploit/explore over their hyperparameters and weights — `evolve` selecting `rl` checkpoints),
**ES vs. PPO at scale**, and a **learned world model** as a surrogate fitness. Each is its own design decision
when it comes; listed so the earlier phases don't close doors (e.g. trainers must be checkpointable and resumable
for PBT).

## Frontend integration

- `apps/frontend/app/wasm/rl/` (checked in, built by `libs/rl/build-wasm.py`), loaded only by the pages that train
  live. A `rlTraining.worker.ts` runs a trainer in a worker: `step(n)` advances n environment steps and posts back
  stats and a policy snapshot, so the page stays responsive and the demo can be paused, sped up or reset.
- A `useRlTrainer` composable (the session-composable pattern of `useSnakeSession`) and reusable pieces:
  `LearningCurve` (on the existing `LineChart`, x-axis in env steps, optional evolution overlay), `QTableView`
  (Phase 1), `HyperparameterPanel`, and a policy overlay for the existing `GridBoard`/`SnakeWatchStage`.
- Trained RL champions need nothing new to *watch*: they're model packages (Decision 5), so the game pages,
  `/watch/{runId}` and the leaderboards show them like any other entrant.
- `learnChapters.ts`: four chapters (Q-learning, DQN, policy gradients, self-play), each `coming-soon` until its
  phase lands; prerequisites point at `neuroevolution` and `autodiff`.

## Risks and open questions

- **Browser training speed.** The live demos need visible learning in under a minute. Tabular Q on Snake is cheap
  (millions of steps/s natively); DQN with 11→64→64→3 in WASM is the tight one — measure in Phase 0 with a
  forward/backward microbenchmark before promising a demo shape.
- **Reward shaping.** Snake's reward (+1 food, −1 death, +0.01/−0.02 shaping, starvation) was tuned for
  evolution's episode totals. RL sees it per step, discounted: the shaping may dominate the food signal at low γ.
  Phase 1 tests shaped vs. sparse reward as an arm rather than assuming either.
- **Episode length vs. the protocol.** Training episodes and the leaderboard both end at starvation or the step
  cap; γ < 1 means the agent values food it can reach soon over survival later. Worth a Learn callout.
- **Seed sensitivity.** RL results swing more across seeds than evolution's; ≥5 seeds per arm is a floor, and the
  chapters show the spread.

## Implementation order

Phase 0 → 1 → 2 → 3 → 4, one PR per phase (larger phases may split into algorithm + Learn PRs). Each PR writes
its measured results into an "Implementation notes" section here, updates AGENTS.md's map, and adds any
non-obvious lessons to docs/CODING_GUIDELINES.md.

## Implementation notes

### Phase 0 (2026-09-23)

Built as planned, with one structural addition: **a fourth crate, `rust/envs`** (`redqueen-rl-envs`). The Snake and
Reach1D adapters are needed by *both* binding crates; putting them in either would duplicate them, and putting them
in the core would make it name games. So the core depends on the games crate only for `pcg.rs`, and `rust/envs`
holds the glue (plus the games' baselines as policies and the rollout digest).

- **Adapters are checked against the leaderboard, exactly.** The games' random and greedy baselines played through
  the Snake adapter reproduce `jobs/evaluate.py`'s protocol — score *and* episode length — on the first 60 held-out
  games (`tests/test_envs_and_trainer.py`).
- **Gradients agree with the oracle.** The batched MLP's outputs and gradients match `libs/autodiff` to 1e-12 on
  four shapes (ReLU, tanh, linear layers); Adam matches the paper to 1e-13; finite differences in the Rust tests too.
- **Determinism holds across targets.** Six digests (three training runs of up to 500 Adam updates through ReLU, tanh,
  `sin` and `exp`; three random-agent rollout sets on Snake and Reach1D) are identical from the native build, the
  WASM build in Node (CI) and the WASM build in Chromium (`/dev/rl`).
- **An RL run is an ordinary run.** `jobs/rl_run.py` records iterations through `recorded_run()` with no telemetry
  change; the run page relabels generation/fitness/diversity as iteration/return/policy entropy from
  `config.paradigm`. RL runs are kept off the leaderboard and have no Watch link until Phase 1's `modelpack` loaders.

**Speed** (`jobs/rl_benchmark.py` natively, `/dev/rl` in Chromium; same desktop, Windows x86-64):

| Case | Native | Browser (WASM) | Browser / native |
|---|---|---|---|
| env steps/s, Snake, random agent | 7.05 M | 10.4 M | 147% |
| act/s, one forward of 11→64→64→3 | 530 K | 430 K | 81% |
| updates/s, batch 32, 11→64→64→3 | 8.3 K | 3.5 K | 42% |
| updates/s, batch 32, 27→128→128→3 | 1.9 K | 1.1 K | 59% |
| updates/s, batch 32, 100→256→256→3 | 429 | 241 | 57% |

- **The browser steps the environment faster than native.** Not a measurement error: every step allocates a new
  observation `Vec`, and Windows' native allocator is slower than WASM's; the random agent does little else. Writing
  observations into a reused buffer is a cheap Phase 1 optimization.
- **What it means for the demos** (a desktop; a phone is likely 3–5× slower — each demo gates on the device's own
  measurement and falls back to a recorded run):
  - *Tabular Q-learning:* no constraint at all — millions of steps per second, so the demo is paced by rendering.
  - *DQN on features.v1 (11→64→64→3):* ~3,500 updates/s is ~14,000 env steps/s at one update per 4 steps: 100 K steps
    in ~7 s. Live is comfortable on a desktop.
  - *DQN on egocentric.v1 (27 inputs, 128-wide):* ~1,100 updates/s, 100 K steps in ~25 s — live only on fast devices.
  - *DQN on grid-flat.v1 (100 inputs, 256-wide):* ~240 updates/s — recorded-run replay, not live.

### Phase 1a: tabular Q-learning (2026-09-23)

Built: `tabular.rs` (`Discretizer`; `QTableAgent` = Q-learning or SARSA, n-step returns, decaying epsilon-greedy,
optimistic initial values), Snake's `features.v1` as 11 bits (2,048 rows) and Reach1D as 25 x 13 bins, a `sparse`
reward option (+1 food, -1 death, nothing else), the `qtable` champion in `modelpack` (`load_champion`, and
`export_qtable`: `(observation != 0) . [1, 2, 4, ...]` then `Gather`), `jobs/rl_experiment.py`.

- **Exact against the textbook.** The Rust update rule replays random transitions to *equal* Q-tables as a
  plain-Python Sutton & Barto implementation (`reference_tabular.py`), for Q-learning and SARSA at n = 1, 2, 5.
  SARSA picks its next action in `observe`, not `act`, which is what makes each update a pure function of a transition.
- **Determinism covers learning now:** a `learning` digest (Q-learning and 3-step SARSA trained on Snake) is
  identical natively and in WASM.
- **Truncation is not termination.** A training episode cut by the 1,000-step cap still bootstraps from the next
  state; only a game over doesn't.

**`rl-tabular-v1`** (12 arms x seeds 0-4, `features.v1`, 1M env steps unless noted; final champion on the 200
held-out games; paired permutation test against `q-learning`):

| Arm | Held-out (mean ± sd) | vs `q-learning` | Notes |
|---|---|---|---|
| `q-learning` (alpha 0.1, gamma 0.95, eps 1 → 0.05 over 100k) | 17.63 ± 0.98 | -- | reference |
| `q-optimistic` (initial Q 2, eps 0.02) | **19.38 ± 1.49** | +1.75 (p 0.19) | best, not significant |
| `q-gamma-0.9` | 18.90 ± 1.06 | +1.26 (p 0.06) | |
| `q-eps-fast` (decay over 20k) | 18.69 ± 0.73 | +1.05 (p 0.25) | |
| `sarsa-n3` | 18.59 ± 0.94 | +0.95 (p 0.06) | |
| `q-sparse` (no shaping) | 18.24 ± 0.41 | +0.60 (p 0.25) | shaping isn't needed |
| `q-n3` | 17.96 ± 0.81 | +0.33 (p 0.31) | |
| `q-long` (5M steps) | 17.90 ± 0.64 | +0.26 (p 0.69) | 1M is converged |
| `q-alpha-0.3`, `q-eps-slow`, `sarsa`, `q-gamma-0.99` | 17.35-17.60 | within ±0.3 | |

- **Every arm plateaus at about the greedy baseline (17.9); no variant is significantly better with 5 seeds.**
  Every run visits exactly **256 of the 2,048 rows** -- the reachable ones (one heading bit is always set, food
  bits come in pairs) -- by the end, and 5M steps is no better than 1M: these are converged answers, not unfinished ones.
- **The plan's hypothesis was wrong, and why is the lesson.** It predicted a plateau because "a table can't generalize".
  But any deterministic policy on `features.v1` *is* a 256-row table, and NEAT's 38-point champion reads the same
  11 features -- so a 38-point table exists. What stops Q-learning is **aliasing**: different board situations
  share one row (the features don't see the body beyond one cell), so the row's value is an average over
  situations with different futures, and the Markov assumption that value-based learning rests on fails.
  Evolution searches policies directly, judged by whole-game returns, and doesn't need values to be consistent.
  Value methods need observations that are (close to) Markov -- the case for Phase 2's richer observers.
- **Sample efficiency is where it shines:** 1M steps and **0.4 s** of training for ~18-19 points; neuroevolution
  needed 48.8M steps (16 min) for 17.6 and NEAT 11.8M for 20.3.
- **On the leaderboard:** `q-optimistic`'s settings, rng seed 0 (fixed in advance, not the best seed -- that would be
  selecting on the test set), published as a package (fp64 and fp32 both play every game identically): **#4 at
  19.51 ± 0.94**, behind only NEAT, trained in 0.42 s, 10 µs per decision.
- Not done in 1a: the per-step allocation (0.4 s per million steps makes it moot for tabular; revisit for DQN).
  The Learn chapter and its live demo are Phase 1b.

### Phase 1b: the Learn chapter (2026-09-23)

- **`/learn/q-learning`**, in a new "Reinforcement learning" part: return and discount, the Bellman update, the
  n-step/terminal-vs-truncated handling and SARSA's one-line difference (quoting `tabular.rs`, so Rust joined the
  highlighter), ε-greedy vs. optimism, the `rl-tabular-v1` results (`QLearningResults`, hard-coded like the other
  chapters' cited results), and aliasing illustrated with two boards that share one `features.v1` row.
- **The live demo (`QLearningLab`)** is the real `Trainer` in WASM in a worker (`workers/rlLab.worker.ts`): the WASM
  `Trainer` now takes an algorithm, `name=value` params and a reward mode; `DemoGame` steps a Snake for the board
  while the table plays greedily (`Trainer::act_greedy`), and the core's `Agent::table()` exposes values + visit
  counts (`TableView`) for `QTableGrid` -- the 256 reachable rows as four 8×8 grids. Training is paced in the worker
  (5k / 40k / 400k steps/s); every 25k steps the greedy table plays 30 held-out games for the curve. At the fast
  speed a million steps takes ~2.5 s in the browser -- the pacing, not the WASM, is the limit.
- **Fallback:** without WebAssembly, the leaderboard entrant's run (`jobs/export_rl_recording.py`: curve + final
  table, 27 KB) stands in. A table carries no visit counts, so the recording carries `initial_q` and a row counts
  as learned when it differs from it -- *not* when its three values differ: in a trapped row (danger on all three
  sides) every move dies and all three converge to the same number.
- **Found while building it: coverage follows competence.** ε = 1 random play reaches ~100 of the 256 rows in 5k
  steps but only ~200 by 60k: a random snake stays short, and body-on-two-sides situations need a long one. The
  last rows fill in only once the table plays well (~100k+). Exploration's reach is bounded by the agent's own
  skill, which matters more when Phase 2's larger observations make coverage the question.

### Phase 2a: DQN (2026-09-24)

Built: `dqn.rs` -- a Q-network (ReLU MLP, Huber loss, Adam) with every stabilizer a parameter: experience replay
(`replay_capacity`, 0 = train on each transition once, in order), a target network (`target_update`, 0 = none),
Double DQN, dueling heads, n-step returns (accumulated like the table's) and proportional prioritized replay (a sum tree
whose parents are recomputed, never adjusted, so every target sums the same bits). `td_gradients` is the update as a
pure function, checked against `reference_dqn.py` (`libs/autodiff`) for plain/dueling x single/double to ~1e-12. A
dueling network's `V + A - mean(A)` is linear in the trunk's output, so the snapshot folds the heads into one output
layer: every DQN champion is a plain `{"type": "mlp"}` network (`modelpack.MlpPolicy`, `export_mlp` -- `Gemm`/`Relu`),
and the Watch page draws it with its own activations. A `dqn` digest (every stabilizer on, `egocentric.v1`) joins the
determinism fixture; the WASM build reproduces it bit for bit.

**Results (`rl-dqn-v1`: 1M env steps per run, final network on the 200 held-out games; each arm tested against the
arm it adds one thing to):**

| Arm | Seeds | Held-out (mean ± sd) | Notes |
|---|---|---|---|
| `dqn-naive` (no replay, no target net), egocentric.v1 | 20 | 25.79 ± 6.01 | 1 of 20 diverged (Q -> -98, never learned); the rest 26.2-28.0 |
| + replay (`dqn-replay`) | 20 | 26.66 ± 6.30 | 1 of 20 diverged (Q -> 1.3e10); the rest 25.9-28.7 |
| + target network (`dqn`) | 20 | **28.45 ± 0.90** | 0 of 20 diverged (0 of all 55 runs with a target network); +1.79 vs. replay (p 0.063) |
| + Double DQN | 5 | 28.47 ± 0.80 | -0.47 (p 0.56) |
| + dueling heads | 5 | 28.49 ± 0.65 | +0.02 (p 1.0) |
| + 3-step returns | 5 | 29.36 ± 0.33 | +0.87, better in all 5 pairs (p 0.062, the floor for 5) |
| + prioritized replay (`dqn-per`) | 5 | 29.54 ± 0.75 | +0.18 (p 0.69) |
| `dqn-per` for 5M steps (`dqn-long`) | 5 | **30.45 ± 0.77** | +0.91, better in all 5 pairs (p 0.062) |
| `dqn` on features.v1 | 5 | 18.83 ± 1.20 | -10.1 vs. egocentric: the table's ceiling (Phase 1) |
| `dqn` on grid-flat.v1 | 5 | 0.38 ± 0.05 | never learns to eat |

- **The observer question has an answer, and it's the headline.** On `egocentric.v1` a DQN reaches 28.5 in 1M
  steps and 30.5 in 5M; on `features.v1` it stops at 18.8, where the table stopped. Evolution couldn't make
  `egocentric.v1` pay (doc 0007: NEAT 35.8 vs. 38.0 on features.v1, not better); a gradient learner can -- the
  observation that aliases situations caps value learning, and one that tells them apart lifts it by 10 points.
- **Sample efficiency is the other half.** 5M steps for 30.5 against NEAT's 409M for 38 on features.v1 (and 11.8M
  for 20.3): about 80x fewer environment steps for 80% of the best score. Not a like-for-like horse race --
  different observers and budgets -- but the axis doc 0010 set out to measure.
- **The target network prevents divergence -- which is rare here, but total.** Without one, 1 run in 20 blew up
  (one to Q = 1.3e10 for rewards of about +-1 a step, one to a negative spiral that never learned); with one, 0 of the
  55 runs that had it did (45 on egocentric.v1, 10 on the other observers). Five seeds per arm (the first report) showed one divergence per arm and couldn't tell a
  rate from an anecdote; twenty make it a ~5% failure mode. Excluding divergences, replay and the target network are
  each worth under a point.
- **The later rungs are small.** Double DQN barely lowers the mean `Q(s, a)` (2.59 vs. 2.65) and doesn't change the
  score: with rewards of +-1, gamma 0.95 and three actions there is little max-bias to remove. Dueling and prioritized
  replay: nothing measurable at 5 seeds. 3-step returns: +0.9 and the tightest spread -- consistent, small.
- **`grid-flat.v1` defeats every learner so far** (evolution 0.06-0.17, DQN 0.38). It is one number per cell (empty 0,
  body 1, head 2, food 3) in the absolute frame, while the actions are relative turns: categories share one numeric
  scale, and the heading is implicit in where the neck is. The learner isn't the bottleneck; a one-hot or egocentric
  grid would be the test of that, not more steps.
- **On the leaderboard:** `dqn-long`'s settings (every stabilizer on, 5M steps), rng seed 0 fixed in advance,
  egocentric.v1: **#2 at 29.28 ± 1.00**, behind only the 38-point NEAT -- trained alone in 6.6 min (395 s active),
  27→64→64→3 (6,147 parameters), 12 µs per decision. fp64 and fp32 packages both agree on all 50,262 decisions.
- Timing: experiment runs ran 5-20 at once on one machine, so their `active_s` is inflated; the leaderboard entrant
  was trained alone.

### Phase 2b: the DQN chapter (2026-09-24)

- **`/learn/dqn`**: from a table to a network, the update as a Huber loss (quoting `td_gradients`), the deadly
  triad and the two data fixes (replay, target network), the stability ladder's results, and the observer comparison.
  Results are drawn by `ArmResults` -- a generic strip plot + comparison table that `QLearningResults` now uses too.
- **The live lab (`DqnLab`)** trains a DQN in the browser on the lab worker, now algorithm-generic
  (`workers/rlLab.worker.ts` + `useRlLab`: algorithm, observer and a `name=value` params string; it reports a DQN's
  mean Q and loss per evaluation, and every agent's action values through the core's new `Agent::action_values`). The
  reader picks the observer, the stabilizers and a seed; each run is drawn over the previous one. Measured: ~9k env
  steps/s in WASM on a desktop, so 200k steps take ~25 s -- enough for egocentric.v1 to reach the mid-20s while
  features.v1 stays at ~18, live.
- **The planned "switch the stabilizers off and watch it fall apart" toggle doesn't work as planned**: divergence is
  a 1-in-20 event (Phase 2a), so a reader would almost always see training go fine. The chapter shows the experiment's
  two real divergences instead (`DqnDivergence`, from `jobs/export_rl_curves.py`, Q on a symmetric log scale),
  and the lab offers ten seeds: with both stabilizers off, seed 0 diverges in the browser too (Q -> 2.4e7, score 0)
  and seeds 1-9 learn (24.5-27.3). Seed 0 also diverged in the experiment, though the browser draws different games --
  what the two share is the initial weights and the exploration draws.

### The observation follow-up (2026-09-24)

The Phase 2 results said the observation, not the learner, was the lever; doc 0007's death analysis of the best NEAT
snake said what was missing (whether the space ahead is enclosed). Two new observers (doc 0007's implementation note)
test that, each paired seed-for-seed with the arm it differs from only in what the snake sees (`rl-dqn-v1`, 1M steps
unless noted, 200 held-out games per run):

| Arm | Held-out (mean ± sd) | vs. |
|---|---|---|
| `dqn-ego2`: `dqn` on egocentric.v2 | **41.48 ± 0.79** (40.1-41.9) | +12.5 vs. `dqn` on egocentric.v1; better in all 5 pairs (p 0.062) |
| `dqn-ego2-long`: the full ladder, 5M steps | 40.05 ± 1.09 | +9.6 vs. `dqn-long` (p 0.062) -- but no better than `dqn-ego2` at 1M |
| `dqn-onehot`: `dqn` on grid-onehot.v1 | 8.72 ± 0.47 | +8.3 vs. `dqn` on grid-flat.v1 (0.38; p 0.062) |

- **Reachable space is worth 12.5 points to a DQN**, taking it past every entrant on the leaderboard (NEAT's 37.95 on
  features.v1) after 1M steps and ~90 s of training. The rays were the right idea and the wrong horizon: they see along
  lines, and the fatal mistakes were about areas.
- **With it, the extras stop paying**: every stabilizer plus 5x the steps scored 40.1, not above plain DQN at 1M.
- **`grid-flat.v1`'s failure was largely its encoding** (+8.3 from one-hot channels and an explicit heading), but a
  full board in the board's frame is still far behind the head-frame summaries (8.7 vs. 41.5 at the same budget).
- **Evolution gains even more from it.** NEAT on egocentric.v2 (`snake-ego-v1`'s `neat-max-ego2`, the budget of its
  egocentric.v1 run) scored **60.02** (56.2-62.6) against 35.79 on egocentric.v1: +24.2, better on every seed (p 0.062),
  with ~717M env steps and ~54 min per run. So the observation, not the paradigm, was the ceiling for both -- doc
  0007's death analysis named the missing information, and both learners used it once they could see it. (PPO on
  egocentric.v2, Phase 3 below, reaches 63.0 with 2M steps: the same level from about 1/360 of the experience.)
- **On the leaderboard** (seed 0 fixed in advance, each the best arm's settings, trained alone -- 2 at a time on a
  16-core machine, active time within 1% of CPU time):

  | # | Entrant (egocentric.v2) | Held-out | Env steps | Training |
  |---|---|---|---|---|
  | 1 | **PPO** 33->64->64->3 (`pg-ppo-ego2-long`, 10M steps) | **70.17 ± 2.01** | 10M | 15 min |
  | 2 | NEAT 33 -> 27 hidden -> 3 (`neat-max-ego2`) | 59.93 ± 1.06 | 732M | 53 min |
  | 3 | DQN 33->64->64->3 (`dqn-ego2`) | 41.95 ± 1.53 | 1M | 88 s |
  | 4 | NEAT 11 -> 28 hidden -> 3, features.v1 (the previous #1) | 37.95 ± 1.21 | 409M | 41 min |

  The NEAT champion's fp32 package plays differently from its fp64 one (57.03 as its own `@fp32` entrant; its graph
  sums in a different order), the reason packages carry both. Also found on the way: `evaluate.py` and
  `publish_models.py` admitted any run not still running, so two runs stopped mid-training (marked `failed`) appeared
  on the leaderboard with their partial champions. Entrants are now completed runs only, and both jobs prune records
  and catalog entries for runs that stop qualifying.

### Phase 3: policy gradients (2026-09-24)

Built: `pg.rs` -- one agent for `reinforce` (whole episodes; `baseline=1` adds a learned V), `a2c` (a critic, GAE(0.95),
an update every 128 steps; one sequential environment, not A3C's parallel workers) and `ppo` (2048-step rollouts, 4
epochs of 64-sample minibatches, ratio clipped to 1 ± 0.2, advantages normalized), sharing one GAE (REINFORCE is λ = 1
with V = 0 or the baseline), an entropy bonus (0.01) and global gradient-norm clipping (0.5). Tanh MLPs; a softmax head
for Snake, a Gaussian with a learned log std for Reach1D. `pg_gradients` and `gae` are checked against
`reference_pg.py` (softmax and Gaussian, with and without clipping) to ~1e-12; a `pg` digest joins the determinism
fixture. The Learn chapter (`/learn/policy-gradients`) trains PPO live on Snake (`PolicyGradientLab`, ~6.5k steps/s in
WASM) and on Reach1D (`ReachLab`, through a WASM `DemoEnv`).

**Results (`rl-pg-v1`: 2M env steps, final policy's most likely move on the 200 held-out games, 5 seeds; each arm
against the one before it):**

| Arm | Held-out (mean ± sd) | vs. |
|---|---|---|
| `pg-reinforce` (egocentric.v1) | 28.62 ± 3.17 | -- (DQN: 28.45 at 1M) |
| + baseline | 34.85 ± 1.26 | +6.2, all 5 pairs (p 0.062) |
| `pg-a2c` | 33.25 ± 2.50 | -1.6 (p 0.31) |
| `pg-ppo` | **43.33 ± 3.64** | +10.1, all 5 pairs (p 0.062); every run above NEAT's 37.95 |
| `pg-ppo-ego2` (egocentric.v2) | **63.03 ± 2.43** (59.2-65.5) | +19.7 vs. `pg-ppo` (p 0.062) |
| `pg-ppo-ego2-long` (10M steps) | **66.16 ± 4.19** (60.9-70.2) | +3.1 vs. 2M (p 0.31): more, but not reliably, and noisier |
| `pg-ppo-features16` (11->16->3, features.v1) | 21.32 ± 2.34 | vs. neuroevolution on the same net below |

- **PPO is the strongest learner in the project by a wide margin**: 43 on egocentric.v1 (DQN 28.5, NEAT 35.8 on the
  same observer with a far larger budget) and 63 on egocentric.v2 (DQN 41.5).
- **The baseline and PPO's clipping are the rungs that matter**; A2C's bootstrapping was a wash here.
- **Gradient vs. evolution on the same network** (doc 0010's Phase 3 question): PPO on neuroevolution's exact
  architecture and observation (11 -> 16 -> 3 tanh, features.v1; PPO's output skips the final tanh, which doesn't change
  the argmax) scored 21.3 against neuroevolution's 18.2 (tournament) and 16.5 (lexicase) from `neat-vs-neuro-v1`,
  better on every seed (+3.2 and +5.0, p 0.062 each), with 2M env steps against 15-16.5M and 15 s against 6.1-6.4 min.
  Both sit near the greedy heuristic: on these features the observation, not the learner, is the ceiling.
- **Continuous control works unchanged**: a Gaussian policy (PPO) on Reach1D reaches any target within ~10k steps; the
  spread widens while the mean is wrong, then narrows (0.74 -> 0.91 -> 0.26 over 200k steps, seed 0) -- but only with
  the entropy bonus off; at 0.01 it stays at 0.6-0.75.
- Not built: OpenAI-ES as an extra arm (the `evolve` machinery exists; left for Phase 5, where RL and evolution meet).

### Phase 4: Checkers by self-play (2026-09-26)

Built: `rust/envs/src/selfplay.rs` -- a two-player loop beside the single-agent `Trainer` (it needs the games crate).
One 32 -> 16 -> 1 tanh value network scores positions from the side to move (`checkers::encode`), plays both sides
(a move is worth `-V(position it leaves)`), and learns by **TD(λ), forward view, offline per game**: the last position
before the end targets the outcome for its mover, every earlier one `-((1-λ) V(s_t+1) + λ G_t+1)` (sign flipping each
ply; zero-sum, undiscounted), one Adam step per game. Exploration, since there are no dice: 4 random opening plies and ε
0.1 -> 0.02. An **opponent pool** option plays a fraction of games against frozen past selves. The champion is
`evolve.WeightVector` JSON, so it is the *same* evaluator the evolved runs produce: `jobs/checkers_selfplay_run.py`
records it as an ordinary Checkers run (`representation: td_lambda`) and it joins the versus leaderboard, packaging
and the Checkers page unchanged. A `selfplay` digest joins the determinism fixture (native = WASM); the WASM
`SelfPlayTrainer` powers `/learn/self-play`, where a reader trains one (~700 games/s in a browser) and plays it.

**Results (`rl-selfplay-v1`, jobs/checkers_selfplay_experiment.py: 200k self-play games per run, 5 seeds; each final
network searching 3 plies against material search at 2/3/4 plies and the best evolved evaluator (3-ply), 20 games
each, points per game):**

| Arm | Points (mean ± sd) | material-2 | material-3 | material-4 | evolved 3-ply |
|---|---|---|---|---|---|
| `sp` (λ 0.7) | 0.601 ± 0.099 | 0.75 | **0.595** | 0.42 | **0.64** |
| `sp-pool` (half the games vs the last 10 selves) | **0.605 ± 0.022** | 0.785 | 0.57 | 0.36 | **0.705** |
| `sp-lambda0` (one-step TD) | 0.571 ± 0.035 | 0.775 | 0.56 | 0.395 | 0.555 |
| `sp-lambda1` (Monte-Carlo) | 0.349 ± 0.033 | 0.49 | 0.35 | 0.26 | 0.295 |

- **Self-play beats the best evolved evaluator and material search at equal depth**, in ~3 minutes of training and no
  opponents; only material search a ply *deeper* still wins (0.36-0.42). The evolved evaluators were selected against
  fixed opponents for far longer.
- **λ = 1 fails** (-0.25, every seed): a 70-ply game's result alone is too noisy a target for any one position;
  bootstrapping from the next position's estimate is what makes TD work here. λ 0 and 0.7 are within noise.
- **The opponent pool buys reliability, not strength**: same mean, spread between seeds 4.5x smaller (the worst pure
  self-play seed, 0.45, has no counterpart) -- the role the hall of fame played for evolution.
- **On the leaderboard** (`sp-pool`'s settings, seed 0 fixed in advance, 200k games, trained alone in ~3 min; versus
  round robin, 3-ply): **#2 at 0.812** points per game (201W 85D 14L), behind only material-4 (0.853) and ahead of
  material-3 (0.777) and every evolved evaluator (best 0.748). `evaluate_versus.py` now prunes records of entrants
  that stopped qualifying, like `evaluate.py` -- a versus score is relative to the field, so a stale one is wrong twice
  (it removed a 30k-game pilot run of this job, since tagged `selfplay-pilot`, and a 3-generation smoke run).
- Not done: PPO self-play with legal-move masking (the plan's stretch), and search-improved targets (TD-Leaf,
  AlphaZero-style) -- the obvious next rung.

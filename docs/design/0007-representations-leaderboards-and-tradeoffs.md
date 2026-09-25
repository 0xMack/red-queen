# 0007 — Game representations, leaderboards, and measuring tradeoffs

## Context

Three asks, one underlying gap:

1. **Leaderboards per game**, so different algorithms can be compared on the same problem —
   ranked in a way that fits the game (best score for single-agent games, a tournament rating for
   versus games), with room for humans to compete too.
2. **Swappable representations**: the same game seen as pixels, as a full grid, or as
   pre-chewed features like Snake's current 11 — clearly described, sorted into levels, swappable
   for compatible models, and *explainable* (animate the board turning into the input vector).
3. **Tradeoffs, not just raw score**: training cost, inference cost, model size, hardware, speed,
   robustness.

The gap: the only number this project records about a trained policy is its **training fitness** —
shaped reward (per-step nudges toward food included), averaged over the same 5 seeds it was trained
on. That number isn't comparable across representations, reward-function versions, or algorithms,
and it isn't the game's actual score.

### What running it showed

Measured before writing this doc (`jobs/run-data`, this machine: AMD Zen 4, CPython 3.11):

| Policy (Snake, relative actions) | Mean score, 200 held-out seeds | Training | Inference |
|---|---|---|---|
| Evolved champion `464e2e3b` gen 249 (11 features, 243 weights) | **11.6** ±0.7 (95% CI) — but 16–19 on its 5 training seeds | 125,000 training episodes (100 × 250 × 5), ~3.3 min | 11.7 µs/decision, 5 KB |
| 3-line greedy heuristic, same 11 features | **18.2** | none | 0.38 µs/decision |
| Uniform random | 0.2 | none | — |

- The "best fitness 17.28" the site shows isn't a game score, and the champion **overfits its 5
  training seeds** — only a held-out evaluation shows that.
- A hand-written baseline beats the evolved policy on *every* axis. That's a real, useful result
  that a raw-score, no-baselines leaderboard would never surface.
- The older Snake champion (`81d10ef2`, trained on a 100-float flattened grid) now **crashes**
  (`IndexError`) because `Snake._observation()` was rewritten in place. Nothing records which
  representation a model was trained against.
- The run registry's wall-clock times are unreliable as cost: two identical linear-GP runs show
  0s and 117s (the second includes time spent paused/stalled). Nothing records episodes simulated,
  CPU time, memory, hardware, or seeds.

## Decision 1: split *game*, *observer*, and *action adapter*; name the combination

Today each game hard-codes its observation. Instead:

- **The game** owns rules and full state. Observers read the game object directly (they're in the
  same package), so hidden-from-render state like Snake's heading is available to them.
- **An observer** turns state into a flat `list[float]` (`encode`), and — off the hot path —
  explains each feature (`explain`: value, the board cells it reads, a one-line rule). `encode` runs
  inside training loops and must stay as cheap as today; `explain` exists only for UIs.
- **An action adapter** turns a model's raw outputs into a game action (`decode`), and says how many
  outputs a model needs.

An **interface** is `(game, observer, action adapter)`, with a stable, versioned id, e.g.
`snake/features.v1+relative3.v1`. Every run config, champion, and leaderboard entry records its
interface id. That makes "compatible" precise: a model runs only under the interface it was trained
for (a `WeightVector`'s input/output sizes are baked in). Swapping representations means training
a new model under a different interface and comparing them, not re-using weights across interfaces.

Rewriting an observer's semantics is a new version (`features.v2`), never an in-place edit — that is
exactly what broke `81d10ef2`. The old flattened grid comes back as `snake/grid-flat.v1`, so that
champion runs again.

Where observers are selected: `Snake(observer=...)`, defaulting to today's features, so
`SimulationFitnessEvaluator` and every existing caller are unchanged. `games.interfaces` holds the
registry (`get(interface_id)`, `for_game(game)`), usable from jobs, `apis/backend`, and the Pyodide
worker (the frontend runs the same Python, so there is no JS copy of any observer to drift).

### Representation levels

| Level | Idea | Snake | Checkers |
|---|---|---|---|
| **L0 visual** | what a human sees, rasterized | board pixels (downsampled) | board pixels |
| **L1 full state** | the whole board, symbolically | flattened grid (`grid-flat.v1`, 100 floats); one-hot channels | 32 playable squares, signed, mover's perspective (today's) |
| **L2 local / egocentric** | only what's near the agent, in its frame | window around the head, rotated to heading | — (perspective flip is already partly this) |
| **L3 engineered** | human-chosen features | today's 11 features (`features.v1`) | material, kings, mobility, center control |

Actions have levels too, orthogonal to observations: Snake's absolute 4-way vs. relative 3-way
(`relative3.v1`, which makes reversing into yourself impossible) is itself a representation choice.
Leaderboards show both axes.

L0 is feasible to generate but won't train well yet — there's no convolution in `libs/autodiff`
(only `matmul`), and a flat network already hit a representational ceiling on L1. It's worth having
as the benchmark that shows *why* engineered representations matter, not as a contender, until a
conv layer exists.

### Explaining a representation (the "board → vector" animation)

`explain(game)` returns, per feature: `{index, name, group, value, cells: [(x, y), ...], rule}`.
The frontend runs it in Pyodide against the live game and animates:

- **L1**: overlay each cell's encoded value on the board, then fly cells into a row (the vector),
  then into the network's input nodes (`NetworkDiagram` already exists).
- **L3**: highlight what each feature reads (the danger rays from the head, the food half-planes),
  then drop its 0/1 into its vector slot.
- **L0**: downsample the board into a pixel grid, then flatten.

Because provenance comes from the Python observer, not a re-implementation, the explanation can't
drift from what training actually sees. (The JS forward pass in `apps/frontend/app/utils/
snakePolicy.ts` is exactly that kind of duplicate; it's acceptable only because it visualizes and
never decides.)

### How far this generalizes

| Game type | Observer split | Animation | Leaderboard |
|---|---|---|---|
| Grid/board, perfect information (Snake, Checkers, Connect-4, Othello, Go) | ✅ | ✅ provenance is cells | ✅ |
| Continuous / physics (`reach1d`, CartPole-like) | ✅ | ⚠️ provenance is state variables, not cells: a panel-to-vector animation | ✅ |
| Hidden information (cards, fog of war) | ✅, but observers must be per-player | ⚠️ only show that player's view | ✅ |
| Large, state-dependent action spaces (Checkers) | ⚠️ the action adapter is the hard part (score each legal move vs. a fixed output layer) | ⚠️ | ✅ |

## Decision 2: leaderboards come from a separate *evaluation*, not from training

An **entrant** is something that can play: a run's champion (artifact + interface), a fixed
baseline (random, greedy), or — later — a human. An **evaluation** runs one entrant under a fixed,
versioned **protocol** and stores the results. Training fitness is never used for ranking.

- **Single-agent (Snake), `snake.score.v1`**: 200 held-out seeds (disjoint from training seeds),
  step cap, metric = game score (food eaten). Report mean with a 95% interval, median, min, max,
  zero-score rate, plus the mean on the entrant's *training* seeds so the generalization gap is
  visible. Rank by mean; overlapping intervals are shown as statistical ties, not hidden.
- **Versus (Checkers), `checkers.versus.v1` -- built** (`jobs/evaluate_versus.py`): round-robin, both
  seats, 20 games per pair, reusing `evolve.play_match()`; every finished champion plus the fixed
  baselines. Score = points per game against every other entrant (win 1, draw ½) with a 95% interval, and
  per-opponent W/D/L kept for a head-to-head matrix. It is *relative to the field* (adding an entrant
  shifts every score), which is the drawback of this first version. Still planned: ratings via Glicko-2
  (tracks uncertainty, which suits few-game entrants such as humans), anchored by fixed baselines so the
  scale doesn't drift.
- **Humans, later**: for Snake, a short fixed "challenge" seed set, compared on those same seeds;
  for Checkers, a human is just another `Strategy` behind a UI. Humans always play L0 visual with
  absolute controls, and are labelled that way.

Baselines are always entered: without random and greedy, a ranking says nothing about whether a
score is good.

A protocol version is part of every result's key. Changing seeds, step caps, rules, or reward
invalidates results by *version*, rather than silently mixing incomparable numbers.

## Decision 3: an entry is a vector of measurements, not one score

| Group | Metrics | How measured |
|---|---|---|
| **Quality** | mean / CI / median / min / max score, zero-score rate, train-vs-held-out gap | evaluation job |
| **Inference cost** | µs per decision split into encode vs. decide (median of repeats, after warm-up), parameter count, artifact bytes | evaluation job |
| **Training cost** | fitness evaluations, episodes, env steps simulated (exact, hardware-independent); CPU time, active wall time excluding pauses, peak memory | training job (`jobs/costs.py`), stored in the run summary |
| **Reproducibility** | spread across repeated training runs with different RNG seeds | optional, leaderboard-bound configs only (it multiplies training compute) |
| **Context** | interface + level, model family, hardware fingerprint (CPU, Python impl/version, platform), pure-Python vs. C++ | recorded with every measurement |

Rules that keep the numbers honest:

- **Counters first, clocks second.** Episodes/steps/evaluations are exact and comparable across
  machines. Every timing or memory figure carries a hardware fingerprint, and timings are only
  compared within a hardware class.
- **Exclude paused time** — the pause/resume control API (doc 0005 step 7) would otherwise inflate
  wall time, as the 117s linear-GP run shows.
- **Legacy runs are labelled estimated.** Runs recorded before this doc lack counters; their
  training cost is derived from config (population × generations × benchmark size) and metrics
  timestamps, and marked as such.
- **No single weighted score.** The weights would be arbitrary and would hide exactly the tradeoffs
  worth seeing. The UI shows a sortable table and a Pareto view: score vs. a chosen cost axis,
  with non-dominated entries highlighted (the same idea as `ParetoSelection`).

## Where things live

- `libs/games`: `observation.py` (the `FeatureSource` explanation type, shared across games),
  observers + action adapters inside each game module, `interfaces.py` (the registry). Still no
  dependency on `evolve`.
- `libs/telemetry`: an `EvaluationStore` protocol + `SqliteEvaluationStore` (its own
  `evaluations.db` beside `runs.db`), same Protocol-first shape as `RunRegistry`. Run summaries gain
  a `cost` block — `summary` is already a free-form dict, so no schema change.
- `jobs/`: `costs.py` (training-cost meter: an `on_generation` callback plus a pause-excluding
  wrapper around the control callback), `evaluate.py` (runs entrants under a protocol, benchmarks
  inference, writes `EvaluationStore`), `backfill_interfaces.py` (one-off: records the interface
  of pre-0007 runs by inspecting their champion artifact's layer sizes). Evaluation logic starts in
  `jobs/` like other wiring; it graduates to a `libs/` package if/when versus ratings need to share
  it with `apis/backend`.
- `apis/backend`: `GET /games/{game}/leaderboard`, `GET /games/{game}/interfaces`.
- `apps/frontend`: a per-game leaderboard page (table + Pareto view + protocol/hardware notes);
  watch mode loads a champion's own interface instead of assuming Snake's current one.

## Incremental plan

1. **Interfaces for Snake** — observer/adapter split, `features.v1` + restored `grid-flat.v1`,
   interface id recorded in new runs and backfilled for old ones; watch mode honors it (fixes the
   crashing `81d10ef2` champion).
2. **Snake evaluation + leaderboard with cost metrics** — `EvaluationStore`, `jobs/evaluate.py`
   with random/greedy baselines, training-cost meter in `jobs/snake_neuro_run.py`, backend endpoint,
   leaderboard page.
3. **Representation explorer + animation** — `explain()` for each observer, board → vector → network
   animation, first for L1 and L3; then L2 egocentric and L0 pixel observers.
4. **Human challenge seeds for Snake.**
5. **Checkers** — observers per level, round-robin + Glicko-2, then a human-play UI.
6. **Reproducibility** — repeated-seed training for leaderboard-bound configs.

## Implementation note: `egocentric.v1`, Snake's first L2 observer

The L2 row of the levels table had no Snake entry. `snake/egocentric.v1+relative3.v1` (27 inputs) is it: what the
head can *see*, in its own frame, so there is no absolute heading in it.

- **7 line-of-sight rays** (left, front-left, front, front-right, right, back-left, back-right; straight back is the
  neck), each giving the wall, the first body segment, and food as `1 / distance` (0 = not seen; food is hidden behind
  body). A body segment counts only if it will still be there when the head arrives (the tail vacates one segment per
  move; a diagonal step is two moves), so at distance 1 a ray is exactly `features.v1`'s danger.
- **Food and tail** as (ahead, right) offsets, scaled by the board's longer side.
- **Apples eaten** (score / board cells: the count of rewards collected, not the snake's length) and a **hunger**
  clock (steps since food / the starvation limit -- state `features.v1` cannot see).

Chosen from a measurement, not a guess: the leaderboard-#1 `features.v1` champion (37.95) dies by hitting its own body
in 137 of 139 deaths, 71% of them with room to spare (it had not yet boxed itself in), and 30% of its games are ended
by the 1000-step cap, not by death. Cost is O(rays x board side) per step, in the Rust core, so the browser runs the
same code. Checked against an independent pure-Python oracle (`tests/reference_snake.py`) on every step of ~100 games.

**Result (`snake-ego-v1`, 5 seeds, the `neat-max` config, paired seed-for-seed with `snake-long-v1`'s `neat-max` on
`features.v1`; 200 held-out games each):** `egocentric.v1` scored **35.79 +- 3.21** against `features.v1`'s
**37.98 +- 1.17** (per seed 32.6 / 39.3 / 35.8 / 38.7 / 32.6 vs. 39.4 / 38.9 / 37.7 / 37.4 / 36.5; paired sign-flip
p=0.25, unpaired exact p=0.21) -- **not better**, and less consistent. It learns much faster early (27.7 vs. 19.5 at
generation 50) and plateaus lower, with larger networks (170 vs. 105 parameters).

Why, from the death analysis of the same held-out games: the rays make the snake more *efficient* (15-18 steps per food
vs. 22) but it dies by hitting its own body more (163-200 of 200 games vs. 137 of 139 for the `features.v1`
flagship, which also survives the 1000-step cap in 30% of games) and is already boxed into a region smaller than its
body before the fatal move in 49-58% of deaths (vs. 29%). A ray sees along a line; it cannot tell that the space it is
heading into is enclosed. That points at *reachable space* as the missing information -- not more rays. Caveats: five
seeds, and the difference is within noise; the interface is registered and published-package-ready but has no
leaderboard entry (experiment-tagged runs stay off it).

## Implementation note: `egocentric.v2` and `grid-onehot.v1` (2026-09-24)

Two observers added to answer the two questions the results above left open (both in the Rust core, both checked on
every step of the parity games against independent Python in `tests/reference_snake.py`):

- **`egocentric.v2`** (33 values) = `egocentric.v1` + what the death analysis said a ray cannot see: for each move
  (left, straight, right), the share of the board's free cells still reachable from the cell it enters -- a flood fill
  over the body as it will be after the move -- and whether the tail is reachable from there. A fatal move reads 0, 0.
  Cost is O(cells) per move, three moves per step.
- **`grid-onehot.v1`** (304 values on 10x10) = `grid-flat.v1`'s information as three 0/1 channels per cell (body,
  head, food) plus the heading one-hot: is `grid-flat.v1`'s failure (every learner ~0) its encoding?

Results are in docs/design/0010 ("The observation follow-up"): on `egocentric.v2` NEAT scores **60.0** (vs. 35.8 on
`egocentric.v1`, same budget, every seed better), a DQN 41.5 (vs. 28.5) and PPO 63.0 (vs. 43.3). The death analysis
above was right about what was missing: the representation, not the learner, had been the ceiling for evolution too.
A DQN on `grid-onehot.v1` scores 8.7 (vs. 0.4 on `grid-flat.v1`): the flat grid's encoding was most of its failure.

## Explicitly out of scope for now

- Energy/power measurement (not measurable reliably from inside a Python process on this machine;
  CPU time is the proxy).
- Cross-machine timing comparisons.
- A composite "overall" score.
- Symbolic-regression leaderboards for the linear-GP benchmark runs — same machinery would apply
  (protocol = held-out inputs, metric = error), but no game is involved and nothing asks for it yet.

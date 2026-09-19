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
- **Versus (Checkers), later**: round-robin, both seats, several games per pair, reusing
  `evolve.play_match()`. Ratings via Glicko-2 (tracks uncertainty, which suits few-game entrants
  such as humans), anchored by fixed baselines so the scale doesn't drift.
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

## Explicitly out of scope for now

- Energy/power measurement (not measurable reliably from inside a Python process on this machine;
  CPU time is the proxy).
- Cross-machine timing comparisons.
- A composite "overall" score.
- Symbolic-regression leaderboards for the linear-GP benchmark runs — same machinery would apply
  (protocol = held-out inputs, metric = error), but no game is involved and nothing asks for it yet.

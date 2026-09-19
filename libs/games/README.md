# games

Toy games/simulations, one module per game — a single package rather than one `libs/` package per
game (per [docs/design/0003](../../docs/design/0003-algorithm-landscape-and-roadmap.md)'s "datasets
and games" plan), so shared utilities have an obvious home as more games are added instead of being
duplicated per package or living awkwardly in `evolve`.

Every single-agent game implements `evolve.simulation.Environment`'s `reset()`/`step()` shape; the
one two-player game so far (`checkers.py`) implements `evolve.match.MultiAgentEnvironment` instead
(docs/design/0006) — a board game's legal actions depend on state and there's more than one reward
stream, so the single-agent shape doesn't fit. Either way this package never imports `evolve` — the
dependency points the other way (an evaluator depends on a game's shape; a game doesn't know
`evolve` exists), same as `evolve`/`telemetry`.

Every game also implements a **separate, on-demand `render_state() -> dict`** (`rendering.py`'s
`Renderable`) returning the full board/scene as plain data — not part of `Environment`, and never
called from the fast training path. This is the split that makes "connect it to a frontend/API/
human-play later" cheap: `step()`/`reset()` stay fast and numeric (what a genome trains against);
`render_state()` is what a console renderer, a future Vue3 frontend, or a stored trajectory
artifact would consume, none of which need to touch or slow down the training path.

## The Rust game core (docs/design/0009)

Every game's rules, observers, action adapters and baselines are implemented once, in Rust
(`rust/core`), and reach Python through PyO3 (`rust/python` → the `games._native` extension; the
modules in `src/games/` wrap it in the Python API everything already used) and the browser through
WebAssembly (`rust/wasm` → `apps/frontend/app/wasm/games/`, regenerated with
`uv run python libs/games/build-wasm.py`). Randomness is a specified PCG32 (`rust/core/src/pcg.rs`),
so a seed is the same game in training, evaluation and a visitor's browser.

- Installing needs cargo (maturin builds the extension during `uv sync`; `[tool.uv] cache-keys`
  rebuilds it when `rust/**` changes). On Windows, stop anything that has `_native.pyd` loaded first.
- `tests/reference_*.py` are the original pure-Python implementations, kept as oracles:
  `tests/test_native_parity.py` requires identical trajectories (observations, rewards, render order,
  Checkers' move *order*) across many seeds and boards. `tests/test_wasm_build.py` fails when the Rust
  changed but the checked-in WASM build wasn't regenerated.
- Moving to PCG32 changed which game each Snake seed produces, so the leaderboard protocol became
  `snake.score.v2` (docs/design/0007: a protocol change is a new version, never an edit).

## Contents

- `reach1d.py` — `ReachTarget1D`: a toy 1D continuous-control task. See
  `notebooks/0004-neuroevolution-reach1d.ipynb`.
- `observation.py` / `interfaces.py` — docs/design/0007's split of *what a model sees* from the
  game itself. An `Observer` encodes a game into `list[float]` (plus feature names); an
  `ActionAdapter` decodes model outputs into an action; an `Interface` is the versioned combination
  (`snake/features.v1+relative3.v1`), and `interfaces.get()/for_game()/find()` is the one registry
  jobs and `apis/backend` use (the browser runs the same observers compiled to WASM). Observers carry a representation level
  (0 visual, 1 full state, 2 local/egocentric, 3 engineered). Never change what an existing observer
  encodes — add a new version, or every champion trained on the old one silently breaks.
- `snake.py` — `Snake`: a grid game. Its observer is pluggable (`Snake(observer=...)`); the default,
  `SnakeFeatures` (`features.v1`), is 11 hand-engineered features (danger
  straight/left/right, heading one-hot, food direction) — replaced an earlier board-size-dependent
  flattened-grid observation once `notebooks/0005-neuroevolution-snake.ipynb`'s own conclusion
  ("a representation ceiling, not a compute shortage") pointed at the representation, not more
  compute, as the next thing to fix; see the module docstring for the full reasoning and
  `jobs/snake_neuro_run.py` for the retrained result (best_fitness 0.65 → 17.28, same generation
  budget class, now actually eating food instead of dying near-immediately). The replaced
  100-float grid lives on as `SnakeGridFlat` (`grid-flat.v1`), so champions trained on it still run
  and the two can be compared on the leaderboard. Action is a relative turn
  (left/straight/right), so reversing into your own body is structurally impossible. Reward is
  intentionally asymmetric (penalize moving away from food more than moving closer is rewarded) — a
  symmetric version let an evolved policy oscillate between two cells forever for ~0 net reward, a
  real reward-hacking failure mode found by running it, not guessed at in advance.
- `rendering.py` — `Renderable` protocol + `render_grid_ascii()`, a shared ASCII renderer for any
  grid-based game (not Snake-specific) — the kind of reuse consolidating into one package was for.
- `checkers.py` — `Checkers`: American/English draughts, the first two-player game (docs/design/0006)
  — real rules, not a simplification: mandatory captures, mandatory multi-jump continuation
  (represented as one `Move = (from, *landing_squares)` per turn, not one `step()` per jump),
  kinging. `render_state()` reuses the exact same `{width, height, cells}` shape family as
  `snake.py`, just a wider label vocabulary (`"black_man"`/`"red_king"`/...) — no protocol changes
  needed for a whole new *kind* of piece. `simulate(move)` returns the observation a move would
  produce without mutating the environment, supporting a 1-ply position-evaluator strategy (score
  every legal move's resulting position, play the best) with no per-move feature engineering.
  Validated (see `libs/evolve/tests/test_match.py` and this module's own tests): two static
  strategies can play a full match via `evolve.play_match()`, and a small `WeightVector` population
  evolved against `MatchFitnessEvaluator` measurably improves against a randomized opponent
  (mean fitness ~0.09 → ~0.21 over 40 generations) — noisy with few opponent samples per genome,
  clean once given more (multiple opponent-strategy instances in the pool to average out the
  opponent's own randomness) — the same "verify by running, not just by reasoning" lesson this
  project keeps re-learning, this time about fitness-signal noise rather than reward shaping.

## Usage

```python
from games.reach1d import ReachTarget1D, benchmark_environments

env = ReachTarget1D(target=5.0, start_position=0.0)
observation = env.reset()
observation, reward, done = env.step(action=1.0)
```

```python
from games.rendering import render_grid_ascii
from games.snake import Snake

env = Snake(width=10, height=10, seed=0)
env.reset()
state = env.render_state()
print(render_grid_ascii(state["width"], state["height"], state["cells"], {"head": "@", "body": "o", "food": "*"}))
```

```python
import random

from evolve import play_match
from games.checkers import Checkers

rng = random.Random(0)


def random_strategy(observation, legal_moves):
    return rng.choice(legal_moves)


result = play_match(Checkers(), {0: random_strategy, 1: random_strategy}, max_moves=200)
print(result.winner, result.moves_played)
```

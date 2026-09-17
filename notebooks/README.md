# notebooks

Jupyter notebooks for exploration: algorithm experiments, analysis, and write-ups. Per
docs/design/0003, this is the first home for comparing techniques — raw runs live in `telemetry`;
a notebook is the narrative on top explaining what a comparison shows.

## Contents

- `0001-tournament-vs-lexicase.ipynb` — the first comparison notebook (docs/design/0003 phase 2):
  `TournamentSelection` vs `LexicaseSelection` on the fixed benchmark from
  `jobs/baseline_gp_run.py`, plus an isolated demo of why lexicase can prefer a "specialist" over a
  higher-mean "generalist".
- `0002-pareto-selection.ipynb` — accuracy vs. program size (docs/design/0003 phase 3):
  `ParetoSelection` vs `TournamentSelection` on the same fixed benchmark, using
  `LinearProgram.effective_instruction_count()` as the complexity objective. An honest tradeoff,
  not a strict improvement — smaller programs, at a real accuracy cost on this benchmark.
- `0003-linear-vs-tree.ipynb` — the complementary axis to 0001/0002 (docs/design/0003 phase 4):
  same selection strategy, different genome representation (`LinearProgram` vs. `TreeProgram`).
  The main point is architectural — `evolve()` and every selection strategy worked unchanged
  against a structurally different genome — with a real (if benchmark-specific) accuracy result
  alongside it.

- `0004-neuroevolution-reach1d.ipynb` — the biggest jump yet (docs/design/0003 phase 5): a
  non-program-shaped genome (`WeightVector`, a neural network's weights) and simulation-based
  fitness (`SimulationFitnessEvaluator` against `games.reach1d`) instead of a static dataset.
  Includes a real bug found by running it — the environment's first version made two benchmark
  scenarios observationally indistinguishable — fixed, not edited out of the story.
- `0005-neuroevolution-snake.ipynb` — `games.snake`, a harder learning problem than `reach1d`
  (100-float flattened-grid observation vs. 2 floats) and an honestly modest result: real
  improvement (best fitness -1.08 → +0.65) but inconsistent across scenarios, not a clean win.
  Includes a second real bug found by running it, not designed around in advance — an initial
  symmetric shaping reward let the evolved policy oscillate in place forever for ~0 net reward.

Re-run any notebook in place with `PYTHONUTF8=1 uv run jupyter execute --inplace
notebooks/<name>.ipynb` (see AGENTS.md for why `PYTHONUTF8=1` is needed on Windows).

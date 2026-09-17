# evolve

The pure-Python evolutionary computation prototype: linear GP genome, fitness/selection/variation
interfaces, and a telemetry-agnostic `evolve()` loop. This is phase 1 of
[docs/design/0003](../../docs/design/0003-algorithm-landscape-and-roadmap.md)'s roadmap — validate
the `Population`/`FitnessEvaluator`/`SelectionStrategy`/`VariationStrategy` abstractions here,
where they're cheap to iterate on and easy to introspect, before anything is committed to C++.

## Contents

- `genome.py` — `LinearProgram`: a fixed-length register-machine individual (from-scratch Python
  reimplementation of the same idea as `libs/RedQueenCbind`, not a port of it), plus
  `effective_instruction_count()` — a standard linear-GP "structural intron" analysis giving a
  naturally-varying complexity measure with no representation change needed.
- `fitness.py` — `FitnessEvaluator` protocol + `SymbolicRegressionFitness` (dataset-based, genome-
  generic via an injected `run: (genome, x) -> float`, defaulting to `LinearProgram`'s
  `.output()`). Returns **per-test-case** fitness (higher is better), not a single aggregate —
  `LexicaseSelection` needs the breakdown; anything that wants one scalar (elitism, telemetry)
  reduces it itself.
- `selection.py` — `SelectionStrategy` protocol + `TournamentSelection`, `LexicaseSelection`, and
  `ParetoSelection`. The first two are genome-generic — operate only on fitness values, so they
  work unchanged for any future representation. Lexicase can prefer a "specialist" over a
  higher-mean "generalist" — a real tradeoff (see the docstring and
  `notebooks/0001-tournament-vs-lexicase.ipynb`), not a bug. `ParetoSelection` (accuracy vs. an
  injected `complexity` function — for `LinearProgram`, `effective_instruction_count`) optimizes
  for a genuinely different thing, a tradeoff curve rather than a single "better" — see
  `notebooks/0002-pareto-selection.ipynb`.
- `variation.py` — `VariationStrategy` protocol + `LinearCrossoverMutation`. Representation-specific
  by nature (see docs/design/0003) — this is the interface tree GP, ES, or LLM-driven mutation
  plugs into as peers.
- `tree.py` — `TreeProgram`: the second genome representation (docs/design/0003 phase 4), a
  Koza-style expression tree, plus `TreeCrossoverMutation` (subtree crossover/mutation, depth-
  limited to control bloat) and `node_count()` (this representation's complexity measure, the
  `ParetoSelection` counterpart to `effective_instruction_count()`). Added to prove
  `evolve()`/`TournamentSelection`/`LexicaseSelection`/`ParetoSelection` are genuinely
  genome-generic, not just written to look that way — see `libs/evolve/tests/test_tree.py` and
  `notebooks/0003-linear-vs-tree.ipynb`.
- `neuro.py` — `WeightVector`: the third genome representation (docs/design/0003 phase 5), and the
  first that isn't program-shaped at all — a small feedforward network's flattened weights,
  evolved directly (Evolution Strategies). `GaussianMutation` is mutation-only, deliberately no
  crossover — averaging two networks' weights doesn't generally combine their behavior the way
  swapping GP instructions/subtrees does. `l2_norm()` is this representation's `ParetoSelection`
  complexity measure.
- `simulation.py` — `Environment` protocol (`reset()`/`step()`, docs/design/0002) and
  `SimulationFitnessEvaluator`: dataset-based fitness's simulation counterpart, one fitness value
  per environment/episode, same per-test-case contract as `SymbolicRegressionFitness` so
  `LexicaseSelection` works on simulation fitness with no changes. A concrete environment (e.g.
  `games.reach1d` in `libs/games`) implements `Environment` but never imports this module — same
  dependency direction as `evolve`/`telemetry`.
- `population.py` — `evolve()`, the orchestration loop, and `GenerationSummary` (this package's own
  telemetry-agnostic per-generation type — see docs/design/0001 "Decoupling from telemetry").

`evolve()` has **no import of and no dependency on `libs/telemetry`**. Wiring a run to telemetry is
an adapter that lives outside this package — see `jobs/baseline_gp_run.py`.

See `notebooks/0004-neuroevolution-reach1d.ipynb` for `WeightVector` +
`SimulationFitnessEvaluator` actually solving a toy environment end to end.

## Usage

```python
import random

from evolve import LinearCrossoverMutation, SymbolicRegressionFitness, TournamentSelection, evolve, random_program

rng = random.Random(0)
population = [random_program(num_instructions=12, num_registers=4, num_inputs=1, rng=rng) for _ in range(50)]

final = evolve(
    population,
    # same fixed benchmark as jobs/baseline_gp_run.py -- see docs/design/0003 "fixed benchmark
    # problems as an anchor"
    fitness=SymbolicRegressionFitness(
        target=lambda x: x**4 - 3 * x**2 + 2, inputs=[i / 5 for i in range(-5, 6)]
    ),
    selection=TournamentSelection(k=3),
    variation=LinearCrossoverMutation(mutation_rate=0.1),
    generations=60,
    on_generation=[lambda s: print(s.generation, s.best_fitness)],
    rng=rng,
)
```

# evolve

The pure-Python evolutionary computation prototype: linear GP genome, fitness/selection/variation
interfaces, and a telemetry-agnostic `evolve()` loop. This is phase 1 of
[docs/design/0003](../../docs/design/0003-algorithm-landscape-and-roadmap.md)'s roadmap — validate
the `Population`/`FitnessEvaluator`/`SelectionStrategy`/`VariationStrategy` abstractions here,
where they're cheap to iterate on and easy to introspect, before anything is committed to C++.

## Contents

- `genome.py` — `LinearProgram`: a fixed-length register-machine individual (from-scratch Python
  reimplementation of the same idea as `libs/RedQueenCbind`, not a port of it).
- `fitness.py` — `FitnessEvaluator` protocol + `SymbolicRegressionFitness` (dataset-based, -MSE
  against a target function). Fitness is always "higher is better" throughout this package.
- `selection.py` — `SelectionStrategy` protocol + `TournamentSelection`. Genome-generic — operates
  only on fitness values, so it works unchanged for any future representation.
- `variation.py` — `VariationStrategy` protocol + `LinearCrossoverMutation`. Representation-specific
  by nature (see docs/design/0003) — this is the interface tree GP, ES, or LLM-driven mutation will
  plug into as peers later.
- `population.py` — `evolve()`, the orchestration loop, and `GenerationSummary` (this package's own
  telemetry-agnostic per-generation type — see docs/design/0001 "Decoupling from telemetry").

`evolve()` has **no import of and no dependency on `libs/telemetry`**. Wiring a run to telemetry is
an adapter that lives outside this package — see `jobs/baseline_gp_run.py`.

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

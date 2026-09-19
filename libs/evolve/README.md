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
  complexity measure. `act()` returns just the first output (a bounded scalar, e.g. `reach1d`);
  `forward()` returns every output, for multi-output policies that pick a discrete action via
  argmax (e.g. `games.snake`'s left/straight/right). `to_json()`/`from_json()` are the real
  (round-trippable) wire format -- unlike `LinearProgram`'s `repr()`-based prototype serialization
  in `jobs/baseline_gp_run.py`, these have two real readers: `jobs/snake_neuro_run.py` writing
  champions to `ArtifactStore`, and `apps/frontend`'s Pyodide bridge loading one back to actually
  run it. JSON, not pickle/numpy, so the exact same Python code works loading it back inside
  Pyodide -- every `evolve` submodule is pure stdlib on purpose, verified before that bridge was
  built (see `docs/CODING_GUIDELINES.md`).
- `neat.py` — `NeatGenome`: the fourth representation (docs/design/0008), and the first whose
  *structure* evolves — a feedforward graph of connection genes, each tagged with a global innovation
  number from an `InnovationTracker`, so genomes of different shapes can be aligned for crossover
  (`align`, `crossover`) and compared (`compatibility_distance`). Mutations add connections (never a
  cycle), split connections into a new node (near-neutral by construction), and tweak weights;
  `evolve_neat()` adds speciation + fitness sharing on top, reporting through the same
  `GenerationSummary` `evolve()` uses, with NEAT's own numbers (species, champion size, adaptive
  threshold) in its `extras` field. Its own loop rather than a `SelectionStrategy`/`VariationStrategy`
  pair, because speciation is a population-level operation the per-child `vary(parents)` shape can't
  express. `NeatConfig.target_species` makes the compatibility threshold adaptive — needed for any
  genome that starts at ≥20 genes, where the paper's fixed 3.0 never separates anything. `forward()`
  compiles the graph once per genome (`cached_property`), pruning nodes that can't reach an output.
- `networks.py` — `network_from_json()`: one loader for every trained-network wire format
  (`WeightVector`, or a NEAT genome, which carries `"type": "neat"`), plus `parameter_count()`/
  `describe()` — what `jobs/evaluate.py` and the Pyodide bridge use so they don't care which kind a
  champion is.
- `simulation.py` — `Environment` protocol (`reset()`/`step()`, docs/design/0002) and
  `SimulationFitnessEvaluator`: dataset-based fitness's simulation counterpart, one fitness value
  per environment/episode, same per-test-case contract as `SymbolicRegressionFitness` so
  `LexicaseSelection` works on simulation fitness with no changes. A concrete environment (e.g.
  `games.reach1d` in `libs/games`) implements `Environment` but never imports this module — same
  dependency direction as `evolve`/`telemetry`.
- `population.py` — `evolve()`, the orchestration loop, and `GenerationSummary` (this package's own
  telemetry-agnostic per-generation type — see docs/design/0001 "Decoupling from telemetry").
- `match.py` — `MultiAgentEnvironment` (docs/design/0006), the two-player (player-count-generic)
  sibling of `simulation.py`'s single-agent `Environment` — `reset()`/`legal_moves()`/
  `current_player()`/`step(move)`/`winner()` instead of `reset()`/`step(action)`, since a board
  game's legal actions depend on state and there's more than one reward stream. `play_match()` runs
  one match between any two `Strategy` callables (`(observation, legal_moves) -> move` — no new
  class hierarchy; a static heuristic, an evolved genome via `functools.partial(act, genome)`, or a
  classifier are all just functions with this shape) — the reusable "pit any strategy against any
  strategy" mechanic requested as first-class infrastructure, not built into `games.checkers`
  specifically. `MatchFitnessEvaluator` is `SimulationFitnessEvaluator` with "N fixed environments"
  replaced by "N fixed reference opponents" (one fitness value per opponent per seat, so
  `LexicaseSelection` works unchanged). A concrete game (e.g. `games.checkers`) implements
  `MultiAgentEnvironment` but never imports this module — same dependency direction as
  `Environment`/`games`.

`evolve()` has **no import of and no dependency on `libs/telemetry`**. Wiring a run to telemetry is
an adapter that lives outside this package — see `jobs/baseline_gp_run.py`.

See `notebooks/0004-neuroevolution-reach1d.ipynb` for `WeightVector` +
`SimulationFitnessEvaluator` actually solving a toy environment end to end.

## Usage

```python
import random

from evolve import (
    LinearCrossoverMutation,
    SymbolicRegressionFitness,
    TournamentSelection,
    evolve,
    random_program,
)

rng = random.Random(0)
population = [
    random_program(num_instructions=12, num_registers=4, num_inputs=1, rng=rng)
    for _ in range(50)
]

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

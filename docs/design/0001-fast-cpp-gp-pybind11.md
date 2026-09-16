# 0001 — Fast C++ Genetic Programming Engine, exposed to Python via pybind11

Status: **Draft** — first pass, open for discussion.
Owner: mack
Relates to: `libs/RedQueenCbind`

## Context

`libs/RedQueenCbind` already has a working linear/register-machine GP implementation
(`Individual`, `Population`, `DatasetLoader`, `Utils`) bound to Python via pybind11, exercised
against the Iris dataset in `examples/gp.py`. This document formalizes that as the first real
design pass for the monorepo's genetic-programming use case, rather than starting over — the
existing code is the starting point, not a prototype to discard.

The broader monorepo goal is to build custom RL/GA implementations, run them against
purpose-built games/simulations, and visualize what's happening internally to understand and
debug the algorithms. This engine is the first building block: it needs to be fast enough to
actually evolve things, and inspectable enough to feed the visualizations later.

## Goals

- A GP engine core (population, selection, variation, evaluation loop) implemented in C++ for
  speed, with a Python API via pybind11.
- **Pluggable genome representation** — start with the existing linear/register-machine
  representation, leave room to add tree-based GP later without rewriting the population/GA loop.
- **Pluggable fitness evaluation** — support both:
  - dataset-based tasks (symbolic regression / supervised learning, like the current Iris example)
  - simulation-based tasks (a program acts as a policy inside a game/environment defined in
    `libs/`, fitness = accumulated reward)
- Fast enough to run large populations over many generations without the Python layer being the
  bottleneck.
- Inspectable enough to support building a visualizer later: per-generation stats, and the ability
  to see exactly what a given evolved program is doing, step by step.

## Non-goals (for this first version)

- Not designing the games/simulations themselves — that's a separate `libs/` package that this
  engine's simulation-based fitness evaluator will consume via a small interface.
- Not designing the visualization app yet — this doc only needs to make that later app *possible*
  by exposing the right data, not build it.
- Not committing to tree-GP now — just not architecting linear GP in a way that forecloses it.

## Current state (what exists today)

- `Individual` — a single evolved program (register-machine instructions).
- `Population` — a collection of individuals, exposed to Python, with `predict()`.
- `DatasetLoader` / `Utils` — dataset-based helpers, currently Iris-specific in the example.
- No explicit selection/variation strategy abstraction yet, no fitness-evaluator abstraction, no
  simulation-based evaluation, no tracing/introspection API.

## Proposed architecture

### Core abstractions

```
Program            — a single evolved individual's executable representation
Population<Program> — a collection of Programs + GA bookkeeping (generation, stats)
FitnessEvaluator    — Program -> fitness score, pluggable per task
SelectionStrategy   — Population -> parents (tournament, roulette, etc.)
VariationStrategy    — parents -> offspring (crossover/mutation, representation-specific)
```

`Population`/selection/variation operate against these interfaces, not against a concrete
representation — that's what keeps the linear-GP-today, tree-GP-later path open.

**Representation as a compile-time parameter, not a runtime one.** Given the priority on raw
execution speed, `Program` should be a template parameter (policy-based design) rather than a
virtual base class — the hot loop (executing an individual against inputs) must not pay vtable
dispatch per instruction. This means linear-GP and (later) tree-GP are separate instantiations /
separate pybind11 modules or classes (e.g. `LinearPopulation`, `TreePopulation`), sharing the same
generic GA machinery via templates, rather than one polymorphic `Population` handling both at
runtime. Fitness evaluators and selection strategies can be templates too, or ordinary virtual
interfaces — they're called once per individual per generation, not once per instruction, so the
dispatch cost there is negligible.

### Fitness evaluation: dataset vs simulation

Both are just implementations of `FitnessEvaluator`:

- `DatasetFitnessEvaluator` — holds a dataset (features + labels), runs a `Program` over each row,
  scores against the label (error for regression, accuracy for classification). Generalizes the
  current Iris-specific loading/eval code so any dataset can be plugged in.
- `SimulationFitnessEvaluator` — holds/owns a simulation/environment instance (from a future
  `libs/<game>` package), runs a `Program` as a policy for an episode (observation → program →
  action → environment step, repeat), returns accumulated reward. This is the piece that connects
  this engine to the "games/simulations" half of the monorepo's goal — the environment interface
  it depends on (`reset()`, `step(action) -> (observation, reward, done)`) should be specified
  here but implemented by each game/simulation package, not by this engine.

### Reconciling speed and introspectability

You asked for ideas here rather than picking one tradeoff — proposal below; flag if you'd rather
go a different direction.

Rather than building two separate engines (fast/opaque vs. slow/instrumented), split by
**frequency of use**, since introspection is only ever needed for a handful of individuals at a
time, not the whole population every generation:

1. **Hot path (every individual, every generation):** the plain execution loop, no tracing, no
   logging — this is what determines evolution throughput. Stays exactly as lean as a
   speed-only design would be.
2. **Per-generation summary stats (cheap, always on):** best/mean/worst fitness, population
   diversity — collected as a byproduct of the hot-path pass (a handful of extra comparisons/sums,
   not a second pass) and pushed to Python once per generation. This alone is enough to drive a
   live fitness-over-time chart.
3. **On-demand trace/replay (rare, opt-in, off hot path):** given a specific individual (e.g. the
   generation's champion, or one picked by index/id from Python), *re-execute just that program*
   in an instrumented mode — full register/stack state at every step, instruction pointer trace,
   per-step fitness contribution if applicable. Since this replays a single already-evaluated
   program rather than running the whole population, its cost is irrelevant to overall evolution
   speed. This is what a step-through debugger/visualizer in `apps/` would call.

This gets "both" without maintaining two engines: one lean hot loop, plus a replay mode for deep
inspection that's opt-in and out of the critical path. If profiling later shows the hot loop
itself needs a genuinely different data layout for speed (e.g. SoA for SIMD) vs. what's convenient
to trace, that's the point where a second representation would become justified — not before.

### pybind11 API surface (sketch)

```python
pop = LinearPopulation(genome_config, fitness_evaluator, selection, variation)
pop.evolve(generations=100, on_generation=callback)  # callback gets per-gen stats
best = pop.best()
trace = pop.trace(best)      # or pop.trace(individual_id)
trace.steps                  # list of (instruction, registers_before, registers_after, ...)
```

Exact shape TBD as this gets built — the key commitment is that `evolve()` doesn't require
tracing to run fast, and `trace()` is a separate, on-demand call.

## Open questions

- What should the environment interface (`reset`/`step`/`done`) look like concretely, and should
  it live in this doc, its own design doc, or be defined alongside the first game/simulation
  package? Leaning toward: sketch it here, finalize it when the first simulation is built.
- Selection/variation strategies to support first (tournament selection + point
  mutation/crossover are the likely minimum) — not blocking, but worth a short follow-up list.
- Do we want the Python API to be one package with multiple classes (`LinearPopulation`,
  `TreePopulation`, ...), or separate pybind11 modules per representation? Leaning toward one
  package, multiple classes, deferred until tree-GP is actually being built.

## Incremental plan

1. Refactor current linear GP code behind the `Program` / `FitnessEvaluator` /
   `SelectionStrategy` / `VariationStrategy` shape above, with `DatasetFitnessEvaluator`
   generalized off the current Iris-specific logic. No behavior change yet.
2. Add per-generation stats callback to the pybind11 API.
3. Add the on-demand `trace()` API for a single individual.
4. Define the environment interface and build a first toy simulation/game in `libs/` to validate
   `SimulationFitnessEvaluator` end to end.
5. Revisit tree-GP as a second `Program` representation once there's a concrete reason to need it.

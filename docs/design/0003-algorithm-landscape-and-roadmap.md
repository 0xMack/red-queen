# 0003 — Algorithm landscape, and building toward it incrementally

Status: **Draft**
Relates to: [0001](0001-fast-cpp-gp-pybind11.md), [0002](0002-realtime-visualization-architecture.md)

## Context

The long-term aim isn't just a fast GP engine — it's a long runway toward the frontier of GP,
with real experimentation in reinforcement learning and LLM/transformer-assisted search along the
way, framed equally as **education** and **experimentation**: every technique added should be
visualizable, explainable, and comparable to what came before, not just a checkbox. This doc
surveys the algorithm landscape that implies, what that means architecturally for the code being
shaped now, and a phased plan for adding complexity incrementally rather than all at once.

## Landscape survey

### GP frontier

- **Representation variety**: tree GP (classic Koza-style), Cartesian GP/CGP (DAG-based, natural
  node reuse), grammar-based/grammatical evolution (genome maps through a grammar to a valid
  phenotype), stack-based GP (PushGP — the representation behind most current general
  program-synthesis benchmark work).
- **Selection**: lexicase selection — select against a random *sequence* of individual test cases
  rather than aggregate fitness. Close to a default in current GP research because it preserves
  diversity far better than tournament/fitness-proportionate selection.
- **Variation**: semantic/geometric semantic GP — crossover and mutation guided by program
  *behavior* (output distance), not just syntax.
- **Multi-objective**: Pareto-based selection (NSGA-II-style), e.g. trading accuracy against
  program size — parsimony pressure without a hand-tuned penalty term.
- **Strongly-typed GP**: a type system on registers/nodes, needed once programs manipulate more
  than one data type.
- **Quality-diversity**: MAP-Elites, novelty search — search for a diverse archive of behaviors
  instead of a single fittest individual. This is where GP/EA research and RL research have
  genuinely converged, especially in controller/robotics work.

### Reinforcement learning — and where it actually touches GP

- **Gradient-based RL** (DQN, PPO, actor-critic) is a different paradigm entirely: one policy
  improved by gradient descent, not a population improved by selection. Shares the
  environment interface (`reset`/`step`/`reward`) with anything in `libs/`, shares nothing else
  architecturally with a `Population`.
- **Evolution Strategies (ES)** — the case where RL and EA are literally the same algorithm: a
  population of parameter-vector perturbations, selected/weighted by reward, no gradient at all.
- **Neuroevolution** (NEAT/HyperNEAT) — a GA evolving NN weights and topology directly as the
  policy. Same `Population`/selection/variation shape as GP, just a different genome type.
- **Population-Based Training (PBT)** — gradient-trained agents, with evolutionary *selection*
  (exploit/explore, copying weights/hyperparameters between population members) layered on top.
  Selection operates on checkpoints, not the gradient step itself — probably the sharpest existing
  answer to "how do GP/EA and RL fit together."
- **World models / model-based RL** — a learned world model is functionally a **surrogate fitness
  evaluator**, the same idea as surrogate-assisted evolutionary computation under a different name.
  Same slot in the architecture (`FitnessEvaluator`), different domain.

### Transformers/LLMs — three distinct fits, not one

1. **LLM as a variation operator inside an evolutionary outer loop** (FunSearch, AlphaEvolve): an
   LLM proposes code mutations, evolutionary selection decides what survives. Architecturally just
   a `VariationStrategy` that calls an LLM.
2. **Transformer as the policy itself** (Decision Transformer — RL reframed as sequence modeling),
   trained via gradient descent. Same bucket as gradient-based RL: shares the environment
   interface, nothing else.
3. **LLM as fitness/judge, or prompts as genomes** (EvoPrompt-style — the evolved "program" is a
   text prompt, scored by running it against an LLM). Fits the existing pluggable
   `FitnessEvaluator` directly, no change to the GA loop.

## Synthesis: what's shared, what stays separate

Genuinely shared across all of this: the **environment/evaluator interface**, the
**population = (genome, score) + selection/variation** shape, and the **telemetry layer** (0002).
Explicitly *not* shared: pure gradient-based RL (PPO/DQN/Decision Transformer) isn't a population
being selected, and bending `Population` to fit it would be a bad abstraction in both directions —
treat it as a sibling reusing the environment interface and telemetry, never a subclass of the
GP/GA machinery.

**The concrete implication for the code being built now:** the genome type must not be hard-coded
to "sequence of linear-GP instructions" anywhere above the C++ hot loop. ES, neuroevolution, and
prompt-as-genome all need `Population`/selection/variation to be genuinely generic over genome
type — a weight vector and an instruction sequence both need to fit the same shape. `VariationStrategy`
carries the most weight here: it has to accommodate crossover, Gaussian noise (ES), an LLM call,
and eventually a PBT-style exploit/explore step, all as peers behind one interface.

## Incremental, educational, replayable experimentation

Every technique added should ship as a **comparable, replayable experiment** against a fixed
baseline, not just new code landing in a lib. This is exactly why the telemetry stack (0002) was
built generic and decoupled first — it's the substrate that makes "compare tournament vs lexicase,
replay the exact run, show the failure mode" possible without bespoke plumbing per algorithm.

A few things this implies, without over-building them before they're needed:

- **Fixed benchmark problems as an anchor.** Comparisons ("tournament vs lexicase," "linear GP vs
  tree GP," "GA vs ES on the same task") are only meaningful if the problem stays constant while
  the algorithm varies. A small, deliberately reused set of toy problems/environments (start with
  something like the current Iris regression case, plus one toy simulation once
  `SimulationFitnessEvaluator` exists) should exist early and get reused across every subsequent
  experiment, rather than each experiment inventing its own task.
- **Raw runs vs. curated showcases are different concerns** — same "don't conflate" principle as
  `MetricsSink` vs `ArtifactStore` in 0002. `RunRegistry` already gives replay of any run; a
  "showcase" (a narrative explaining what a run or comparison of runs demonstrates, for teaching/
  presenting) is a layer on top, not a replacement. Deliberately **not** designing this now — the
  cheapest starting point is a notebook that reads from `RunRegistry`/`MetricsSource`/
  `ArtifactStore` plus prose, "promoted" to a proper `apps/` showcase gallery only once enough real
  experiments exist to know what curation actually needs (tags? narrative field? grouping multiple
  runs for a side-by-side comparison?).
- **Notebooks are the first home for the educational narrative** — `notebooks/` already exists for
  exactly this (exploration, experiments, write-ups). Each new technique's first outing should be a
  notebook: load a baseline run, load the new technique's run, compare, explain. That comparison
  becomes the template new techniques are checked against.

## Incremental roadmap

Ordered to front-load cheap, high-educational-value, low-architecture-risk additions before
expensive, architecture-testing ones — each phase both proves out a design decision from 0001/0002
and produces a concrete comparison experiment, not just code:

1. **Baseline GA loop** (Python prototype per 0001's plan): tournament selection, standard
   mutation/crossover, linear GP, against a fixed regression benchmark. Wired to telemetry only via
   the `on_generation` callback adapter — first end-to-end validation that evolve → metrics →
   replay actually works, before any algorithm variety exists.
2. **First comparison template**: add lexicase selection as a second `SelectionStrategy` on the
   *same* baseline problem. Cheap (no genome-type change), high payoff (a real frontier technique),
   and produces the first "compare tournament vs lexicase" notebook — the template every later
   comparison follows.
3. **Multi-objective selection** (Pareto, accuracy vs. program size) — another selection-strategy-
   level addition, another comparison experiment (accuracy/complexity tradeoff curves).
4. **A second representation** (tree or stack-based/PushGP) — first real test of the
   genome-genericity decision above; comparison against linear GP on the same benchmark.
5. **Neuroevolution/ES on a toy simulation** — first use of `SimulationFitnessEvaluator` and the
   first genome type that isn't program-shaped at all (a weight vector); tests genome-genericity
   from a completely different angle.
6. **LLM-in-the-loop variation** (FunSearch-style) on an existing evaluator — tests
   `VariationStrategy` pluggability with a fundamentally different kind of operator.
7. **Islands, PBT-style exploit/explore, coevolution** — combination techniques, once the pieces
   they combine already exist individually.

## Open questions

- What the curated-showcase layer needs beyond raw `RunRegistry` — deferred until a few real
  experiments exist to see what's actually missing.
- Exact fixed benchmark set beyond the Iris-derived regression case — decide the first toy
  simulation once `SimulationFitnessEvaluator` is being built (phase 5, or earlier if a simple one
  is useful sooner).

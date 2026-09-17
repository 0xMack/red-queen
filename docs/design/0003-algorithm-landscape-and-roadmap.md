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

**Revised priority (2026-09):** all three above assume an existing, external LLM — call an API, get
a result. That's a real, valid direction eventually, but it skips the thing every other technique
in this doc got built the hard way for: understanding the mechanism from scratch, not just wiring
up someone else's. Before any of the three, build a **small transformer/language model from
scratch** — same spirit as `RedQueenCbind`/`evolve` existing at all instead of just calling a
GP library. Only once that exists and is understood does it make sense to ask which of the three
fits above (or something else entirely — e.g. a small *locally-trained* model standing in for the
"LLM" in fit #1, no external API involved) is worth pursuing for this repo specifically. See the
roadmap below.

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
carries the most weight here: it has to accommodate crossover, Gaussian noise (ES), sampling from a
trained model (once one exists), and eventually a PBT-style exploit/explore step, all as peers
behind one interface.

## Incremental, educational, replayable experimentation

Every technique added should ship as a **comparable, replayable experiment** against a fixed
baseline, not just new code landing in a lib. This is exactly why the telemetry stack (0002) was
built generic and decoupled first — it's the substrate that makes "compare tournament vs lexicase,
replay the exact run, show the failure mode" possible without bespoke plumbing per algorithm.

A few things this implies, without over-building them before they're needed:

- **Fixed benchmark problems as an anchor.** Comparisons ("tournament vs lexicase," "linear GP vs
  tree GP," "GA vs ES on the same task") are only meaningful if the problem stays constant while
  the algorithm varies. A small, deliberately reused set of toy problems/environments — the
  polynomial regression benchmark (`jobs/baseline_gp_run.py`) and `games.reach1d` (`libs/games`)
  for simulation-based fitness — gets reused across every subsequent experiment, rather than each
  one inventing its own task.
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
6. **A small transformer/language model, built from scratch** (own `libs/` package, no external
   API) — tokenization, embeddings, attention, a training loop, generation. Not wired into
   `evolve` yet and not required to be useful for anything else immediately — the point is
   understanding the mechanism, the same reason `RedQueenCbind` and `evolve` exist as from-scratch
   implementations rather than calls to an existing library. Likely warrants its own design doc
   once it starts (probably `0004`), the way the GP engine and telemetry did.
7. **Explore applying it to this project's domain** — deliberately left open until phase 6 exists
   and is understood, rather than committed to now. Candidates, roughly in order of how well they
   fit what's already built: a learned `VariationStrategy` trained on the programs already sitting
   in `ArtifactStore` from prior runs (a self-contained, no-external-API analog to FunSearch's
   LLM-driven mutation); a transformer-based policy representation (Decision-Transformer-style,
   trained via gradient descent — a sibling to the GP/GA machinery per the synthesis above, not a
   subclass of it); a learned fitness/judge. External-API-based LLM-in-the-loop (FunSearch/
   AlphaEvolve-style, or an "LLM as judge" `FitnessEvaluator`) is explicitly **deferred**, not
   planned near-term — it's a real, valid direction eventually, but it's the one place on this
   roadmap that would break from every other technique here being built and understood from
   scratch rather than wired up to something external.
8. **Islands, PBT-style exploit/explore, coevolution** — combination techniques, once the pieces
   they combine already exist individually.

## Datasets and games: near-term concrete plan

Decided (2026-09), after discussion:

- **Datasets**: a small named catalog (`evolve.benchmarks` or `libs/benchmarks`) of known-ground-
  truth synthetic problems (Nguyen/Koza-style symbolic regression) plus Iris, replacing inline
  lambda targets scattered across notebooks/jobs — synthetic-first, real-world datasets expanded
  later rather than prioritized now.
- **Next environment: Snake.** Cheap to simulate (matters — fitness evaluation runs it thousands of
  times per generation), trivially human-playable, obvious visual representation, a natural next
  step up from `games.reach1d`. Same `Environment` protocol, a new module in `libs/games` (one
  shared package per games/simulations, not one `libs/` package per game — reconsidered after
  `reach1d` shipped as its own package; grouping them makes shared utilities have an obvious home
  as more games are added instead of duplicating them per package).
- **Parked for later, not forgotten**: a 2D driving/sensor-casting task (single-agent, more visually
  impressive showcase material) and predator/prey pursuit (the thematic flagship — a literal Red
  Queen coevolutionary arms race, and already `phase 8`'s coevolution item) — both real candidates,
  deliberately not next because of added implementation risk (sensor casting; two-sided reward
  shaping and genuine coevolutionary dynamics) relative to Snake.
- **`apps/`/`apis/` (doc 0002's visualization/live-play layer) stay deferred.** Build out one or two
  more `Environment`s on the existing protocol first, so the eventual rendering contract (what
  state a frontend needs each step) and live-play API are designed against more than one game's
  shape, not guessed at from Snake alone.
- **Two concrete architectural extensions this implies, whenever `apps/`/`apis/` work starts** (not
  yet, per the above): `ArtifactStore` needs a trajectory artifact type (a full episode's
  states/actions/rewards, not just a stored program) for replay/rendering; and a live API (not just
  a replay one) is required for "play against the agent," reusing the control-API slot doc 0002
  already sketched. Recorded human play is also a candidate training corpus for phase 6/7's
  from-scratch model (an imitation-learning baseline, or literal sequence-training data) — noted
  here so it isn't lost by the time `apps/`/`apis/` actually get built.

## Open questions

- What the curated-showcase layer needs beyond raw `RunRegistry` — deferred until a few real
  experiments exist to see what's actually missing.
- ~~Exact fixed benchmark set beyond the Iris-derived regression case~~ — resolved: `games.reach1d`
  (phase 5, `libs/games`) is the first toy simulation; `reach1d.benchmark_environments()` is its
  fixed set.
- Scope for the from-scratch transformer/LM (phase 6): a minimal char-level model (nanoGPT-style)
  vs. something more ambitious; what corpus to start on — deferred until that work actually
  starts, same as every other phase's specifics were decided when reached, not in advance.

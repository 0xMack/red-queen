# 0008 — NEAT, and tracked comparisons between algorithms

## Context

Two asks: **(1)** implement NEAT and test it against Snake; **(2)** explain neuroevolution and NEAT in the
Learn section, interactively. Doc 0003 listed "Neuroevolution (NEAT/HyperNEAT)" as a roadmap item; doc 0007
gave every game run a measured cost and a held-out score. What was missing was a way to say something
*true* about "is algorithm A better than algorithm B" — one training run is an anecdote (rng seeds swing a
Snake policy's score by several points), and nothing recorded which runs belonged to one comparison.

## Decision 1: NEAT is a new genome with its own loop, not a `VariationStrategy`

`libs/evolve/src/evolve/neat.py`: `NeatGenome` (a feed-forward graph of `ConnectionGene`s, each with a global
innovation number from an `InnovationTracker`), the four mutations (perturb weights, add connection, split a
connection into a node, toggle a gene), innovation-aligned `crossover`, `compatibility_distance`, and
`evolve_neat()`.

- **Why not `SelectionStrategy` + `VariationStrategy`.** Those are per-child (`select` a parent, `vary` it).
  Speciation and fitness sharing are *population-level*: every child's parents and the number of offspring each
  species gets depend on the whole generation. Forcing that through the per-child interface would have meant
  hiding population state inside a strategy object. A dedicated loop is honest about it. What *is* shared is the
  reporting contract: `evolve_neat()` emits the same `GenerationSummary` as `evolve()`, so every `jobs/`
  callback (telemetry, held-out monitor, control, cost) works unchanged.
- **Deviations from the paper, on purpose:** feed-forward only (`add_connection` refuses cycles; a network is a
  pure function of its observation, like `WeightVector`); tanh everywhere, for parity and bounded outputs;
  fitness is the mean over the evaluator's per-case values (lexicase has no role once selection is
  species-based); add-node never splits a bias connection.
- **Crossover is acyclic by construction.** A child's genes are always a subset of the fitter parent's, so an acyclic fitter
  parent gives an acyclic child — crossover needs no cycle check, only the mutations that invent structure do.
- **`forward()` compiles once.** A genome is evaluated thousands of times and mutated once, so its evaluation
  order is a `cached_property` (dead-end hidden nodes pruned). This is why `NeatGenome` isn't `slots=True`.

### Wire format and loading

`NeatGenome.to_json()` carries `"type": "neat"`; `WeightVector.to_json()` never had a type field and old
artifacts must keep loading, so *no type* means `WeightVector`. `evolve.network_from_json()` is the one loader
(`jobs/evaluate.py`, the Pyodide worker). Both kinds expose `forward(observation)`, which is all a policy needs.

## Decision 2: the paper's fixed compatibility threshold doesn't survive a 36-gene start

Distance is `c1·E/N + c2·D/N + c3·W̄`, with `N` the larger genome's gene count once it reaches 20. Snake's
minimal genome is 36 genes (11 inputs + bias → 3 outputs), so `N` ≥ 20 from generation 0 and structural
differences barely register against the paper's threshold of 3.0. **The first trial run (6 generations) reported one
species in every generation it recorded** — NEAT silently degenerating into fixed-topology neuroevolution with a growing network. Nothing
failed; only the species count showed it. Fix: `NeatConfig.target_species` steers the threshold (down when there
are too few species, up when too many). Snake uses `target_species=6`.

This is also why `GenerationSummary` gained `extras` and `telemetry.GenerationStats` gained `extras`
(free-form `dict[str, float]`): species count, the champion's hidden nodes and connections, the adaptive
threshold. An algorithm with an internal mechanism that can fail quietly needs that mechanism on a chart
(the run page's "Evolved structure" section). Kept generic so the next algorithm needs no schema change.

## Decision 3: an *experiment* is tagged runs, aggregated per arm

`jobs/snake_experiment.py`: arms × rng seeds, every run an ordinary telemetry run tagged
`config.experiment` / `arm` / `rng_seed`; resumable; arms can run as parallel processes. `report` scores each
run's **final** champion (never the best-looking generation — that's selecting on the test set) on the 200
leaderboard games (`snake.score.v1`) and writes per-arm aggregates to `run-data/experiments/NAME.json`.

- Experiment-tagged runs are **excluded from the leaderboard** (`evaluate.py`), so 20 seeds don't bury every
  other entrant. A single untagged NEAT run (`snake_neat_run.py --seeds resample:5`, seed 0) sits on the
  leaderboard instead. It reproduces the experiment's `neat` seed-0 run exactly (best fitness 11.75, held-out 20.18,
  11 hidden nodes, 73 connections in both), which doubles as a reproducibility check.
- Arms share everything else: interface `snake/features.v1+relative3.v1`, `resample:5` training games (doc 0007:
  nothing to memorize), population 100, 250 generations, 125,000 training episodes.
- `neuro-tournament` exists as a control. NEAT selects on mean fitness while the project's neuroevolution
  default is lexicase; without a tournament arm, "NEAT vs. neuroevolution" would silently be "NEAT vs. lexicase".
- `neat-no-speciation` is the ablation of the one mechanism NEAT is defined by.

## Result: `neat-vs-neuro-v1` (5 seeds per arm, 200 held-out games per run)

| Arm | Held-out score (mean ± sd) | Range | Params | Hidden nodes | Training (pure Python) |
|---|---|---|---|---|---|
| neuroevolution, lexicase | 16.36 ± 1.48 | 14.0–17.6 | 243 | 16 (fixed) | 364 s |
| neuroevolution, tournament | 18.16 ± 1.83 | 16.2–20.5 | 243 | 16 (fixed) | 385 s |
| NEAT, no speciation | 15.57 ± 8.36 | 1.5–21.4 | 41 | 5.6 (1–13) | 79 s |
| **NEAT** | **20.17 ± 0.79** | 19.2–21.1 | 60 | 7.8 (6–11) | 94 s |

Reference: 3-line greedy heuristic 18.47, random 0.12 (same games).

Exact two-sided permutation tests on the difference of means (5 vs 5 → 252 relabelings):

| Comparison | Difference | p |
|---|---|---|
| NEAT − neuro-lexicase | +3.81 | 0.008 |
| NEAT − neuro-tournament | +2.02 | 0.056 |
| neuro-tournament − neuro-lexicase | +1.79 | 0.135 |
| NEAT − NEAT without speciation | +4.60 | 0.349 |

What it supports, and what it doesn't:

- NEAT clearly beats the project's existing neuroevolution setup (lexicase), with ~¼ the parameters and ~¼ the
  training time; all five NEAT runs beat the greedy heuristic, none of the lexicase runs did.
- Against the fairer tournament control the gap is +2.0 but only suggestive at five seeds; ranges overlap.
- Speciation's effect here is **reliability, not typical score**: four of five unspeciated runs scored 14.2–21.4
  (median 20.2 ≈ NEAT's 20.6), one never took off (1.5; its best training fitness never exceeded 2.2 in 250 generations
  and its held-out score never exceeded 2.0, though it grew 7 hidden nodes; the healthy unspeciated run checked
  was above 11 by generation 30). Whether that is the same premature-convergence failure the XOR demo shows was
  not investigated. One failure in five is not distinguishable from chance. (On XOR the Python engine solved 8/8
  speciated vs. 4/8 unspeciated; the TypeScript port, on other seeds, 7–8/8 vs. 3/8.)
- Not shown: any other game, tuned hyperparameters for either side (NEAT's rates and species target are
  first-guess defaults; the fixed network's σ=0.2 / 16 hidden units are earlier choices), more than five seeds.
  The honest claim is "a competitive, cheap way to train a Snake policy here".
- NEAT champions are not minimal (6–11 hidden nodes; the XOR lab's speciated champions carry ~6 for a problem that
  needs 1). Nothing in the loop rewards tidiness (no complexity pressure; `ParetoSelection` is unused).
- Cost caveat: NEAT's advantage in *time* is largely that its evolved graphs have ~60 connections against 243
  weights, in a pure-Python forward pass. Inference on the leaderboard: 4.7 µs/decision vs. 13.3 µs.

## Decision 4: the Learn demos run real algorithms in the browser, via a TypeScript port

Two new chapters (`/learn/neuroevolution`, `/learn/neat`), each with several live demos. Pyodide's ~10 s cold load
is fine for watching a Snake champion but wrong for "click a button, see a mutation", and the demos need
thousands of genomes evolved per second. So `apps/frontend/app/utils/neat.ts` is a hand-kept port of `neat.py`
(and `utils/neuro.ts` a small Evolution-Strategies loop). Divergence is the risk of a port, so it is checked:
`forward()`/`activations()` against the Python on genomes evolved in Python (max |Δ| 2.5e-16), and the
qualitative XOR/speciation behaviour reproduces (8/8 vs. 3–4/8). What the port does *not* replace: **watching a
trained Snake champion** still runs the real `evolve` + `games` in Pyodide, loading the genome with
`network_from_json`; the browser-side TS is only used to *draw* it and light up its activations.

Also: `NeatDiagram.vue` draws any NEAT genome as a graph and is what `WatchChampion` shows for a NEAT run.

## Explicitly out of scope

- HyperNEAT, recurrent/CPPN genomes, NEAT's steepened sigmoid and other activations.
- Complexity pressure for NEAT (e.g. `ParetoSelection` on connection count) — the natural next experiment given how
  large the evolved networks are.
- Tuning either side, other games (checkers via `MatchFitnessEvaluator` is the obvious next test), more seeds.
- A C++ NEAT. `RedQueenCbind` is untouched.

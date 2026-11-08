# 0024 — Copy Cat as Bayesian image parsing: a posterior over layered scenes, sampled by reversible-jump MCMC

Status: **proposed** (2026-11-08). Written against 0022 as of phase 4's first rung (30.8 dB on `copycat.clean.v1`);
its context, Decision 6 (much of that benchmark, `copycat.clean.v2` included, now exists) and Decision 7 (beside
[0023](0023-process-traces-and-research-log.md)'s report runs) still need updating to v23 before it is built.
Relates to: [0022](0022-copy-cat.md) (the task, the benchmark, protocol `copycat.clean.v1`, the structured pipeline this
starts from), [0007](0007-representations-leaderboards-and-tradeoffs.md) (held-out protocols, versioned sets, an entry
as a vector of measurements), [0010](0010-reinforcement-learning.md) (determinism across native and WASM),
[0018](0018-workloads-api-and-cli.md) (runs, experiments, the trainer and the evaluator).

## Context

### Where 0022 stopped

The structured pipeline (0022 phase 2) and its point-by-point refinement (phase 4's first rung) score 30.8 dB of clean
fidelity on the held-out `synthetic` set, against VTracer's 20.2. The gains have come almost entirely from `clean`
(34.0 to 40.7 dB); `medium` (26.5) and `heavy` (22.7) have barely moved. Three things in the design hold them there:

- **Each part is fitted alone, against its own mask, not against the image.** `Target::misfit` compares one shape's
  coverage with that region's "visible fraction", at a fixed `pixel_bits` per wrong pixel. Nothing renders the whole
  SVG and asks how well it explains the input. Most of what has been tuned since (`pixel_bits`, `merge_delta`,
  `background_delta`, `outline_gain`) are settings on that stand-in for the objective.
- **The pipeline is greedy.** Palette, labels, parts, order, shapes: a mistake at one stage is never revisited. That is
  the likely cause of `medium`'s missed parts (0.93x the shapes) and of layer recovery at 0.75 on `heavy`.
- **The corruption is not in the model.** 0022's warp cap (a pixel-perfect copy of the *warped* geometry) is 30.1 dB on
  `medium` and 24.4 on `heavy`. Nothing that copies the input's geometry passes it. Every shape absorbs its share of
  one image-wide warp separately and pays for it in its own bits -- the opposite of what MDL wants, which is one shared
  cause, said once.

Ladder steps 3 and 4 in 0022 (structure moves, gradient refinement) would each tune that design further. This doc
replaces it.

### The idea

Treat tracing as **inference in a generative model**: a scene of stacked shapes, rendered, then warped, blurred,
shaded and composited over a background, then noised, gives the input. Instead of one greedy pass, **sample the
posterior over scenes** given the input, by Markov chain Monte Carlo whose moves can add, remove, split, merge, retype
and re-order shapes (reversible-jump MCMC, Green 1995).

0022's MDL objective is already half of this. A code length is a negative log probability, so
`L(svg) = bits(svg) + bits(input | render(svg))` is a negative log posterior with a particular prior and likelihood.
What changes:

- the likelihood is of the **whole image**, under a forward model that includes the corruption;
- the search is **global and revisable**: any decision can be undone by a move;
- the output is a **distribution**, not one answer: how sure we are that a shape exists, that one part is behind
  another, and what a hidden part looks like.

This is the "image parsing" programme (Tu & Zhu's data-driven MCMC; Tu, Chen, Yuille and Zhu 2005) and the
probabilistic graphics programs line (Mansinghka et al. 2013; Kulkarni et al.'s Picture, 2015), applied to the one
domain where the generative model can be nearly exact: flat-colour layered vector art.

### Why it could be a step change, not a tweak

- It is the only route in reach past the warp cap: a warp in the model is a shared latent, paid for once.
- It removes the hand-tuned thresholds: merging, dissolving thin bands and ignoring clutter become consequences of the
  prior and the noise model, not settings.
- It gives the project a new algorithm family. Evolution, RL, supervised learning and tree search are all here;
  Bayesian inference by sampling is not.
- The posterior is a new product: a designer can be shown where the trace is unsure ("is this one shape or two?",
  "which part is on top?") instead of a confident wrong answer.

## Decision 1: the generative model

A scene `θ` and nuisance factors `ν` generate the input image `x`.

**The scene θ:**
- `K` shapes, bottom to top. Each has a type (circle, ellipse, rect, polygon, path, and -- with `synthetic.v2` and the
  icon sets -- a path with holes), its parameters, and a palette index.
- A palette of `M` colours.
- A background: constant or a linear gradient (0022 phase 2's two models).

**The nuisance factors ν:**
- **A smooth warp:** a displacement field given by a coarse grid of control points (4 x 4, bicubic B-spline, 32
  numbers). It is deliberately *not* `corrupt.v1`'s sinusoid: a model that inverts the generator's exact family is
  fitting the benchmark, not the problem. `corrupt.v2` (Decision 6) uses a different family to check that.
- **A blur width** (Gaussian; one number per image).
- **A shading gradient** across the subject (a direction and a strength, as `corrupt.v1` draws it).
- **The noise scale σ** and, for the robust likelihood, its degrees of freedom.

**Rendering.** The warp is applied to the shapes' *flattened outlines* before rasterizing, not to the image after: the
rasterizer stays exact, and a shape's change stays local. Then each shape's coverage is composited in order, the
shading applied, blur applied to the premultiplied subject, and the result composited over the background -- the same
order `corrupt.v1` uses.

## Decision 2: the prior is 0022's code lengths, made proper

0022's code lengths become a proper prior, `p(θ) ∝ 2^-bits(θ)`, with the changes a probability needs:

- **Shape count and types:** `K` is geometric, from the type bits; a type is drawn from a categorical whose default is
  0022's type costs (a circle is cheaper than a path because it has fewer numbers).
- **Coordinates are continuous** and uniform over the canvas plus a margin; sizes are log-uniform. 0022 needed a fixed
  0.1 px precision to give a coordinate a code length; with densities, the **Bayesian Occam factor** does that work --
  a parameter the data pins down tightly pays for the volume it rules out, automatically.
- **Paths and polygons:** a geometric number of segments or vertices, so a straight run is still cheaper as one
  segment.
- **The palette is a Chinese restaurant process** over shapes: reusing a colour is cheap, a new one is expensive, and
  the palette size is inferred. This replaces 0022's merge rule and its `merge_delta`.
- **Nuisance priors shrink toward none:** zero-mean Gaussians on the warp's control points and on the shading strength,
  a half-normal on blur, a weak prior on σ around the pipeline's own estimate (0022 phase 2, stage 1).

The prior's hyperparameters are tuned only on `train` splits, never on held-out ones (0022's rule), and learning them
from data is experiment E4.

## Decision 3: the likelihood is of the whole image, and robust

- **Every pixel of the input is explained**, by the rendered scene over the background, under per-channel noise in the
  0..1 sRGB space the rasterizer composites in. No region masks, no "visible fractions".
- **Student-t noise, not Gaussian.** JPEG blocks, film grain and clutter specks are outliers under a Gaussian and
  would each demand a shape. A heavy-tailed likelihood lets a pixel be wrong without the whole scene bending to it.
  Implemented as a scale mixture (a per-pixel precision, itself sampled), which keeps the colour update exact (Decision
  4).
- **Background clutter is explained by the background model plus the tails**, or else by shapes the prior must pay for.
  `corrupt.v1`'s clutter level shows which.
- **Transparency is part of the model, as in 0022:** a pixel the background explains is transparent in the output.

## Decision 4: inference is reversible-jump MCMC, with tempering, started from the pipeline

**Started from the pipeline (data-driven MCMC).** The structured pipeline's SVG is the chain's starting state; its
noise and background estimates start `σ` and the background. Experiment E2 also starts from an empty scene, to show
the posterior and not the starting point is doing the work.

**The moves** (each a Metropolis-Hastings proposal, chosen at random with tunable weights):

| Move | Dimension | Proposal |
|---|---|---|
| Nudge a shape | fixed | a random walk on one shape's parameters; step sizes adapted during burn-in only, then frozen |
| Colours | fixed | **exact Gibbs:** with geometry, order and the per-pixel precisions fixed, the render is linear in each palette colour, so its conditional is Gaussian |
| Re-order | fixed | swap two shapes adjacent in the stack whose bounding boxes overlap (symmetric) |
| Nuisance | fixed | random walks on warp control points, blur, shading, background, σ |
| Birth / death | +1 / -1 | birth fits a shape to a blob of the current residual with `copycat_fit`, then adds Gaussian jitter, so the proposal density is computable; death removes a uniformly chosen shape |
| Split / merge | +1 / -1 | a deterministic map between one shape and two, with its Jacobian |
| Retype | changes | circle and ellipse, polygon and path, path to primitive: refit plus jitter, the density computed both ways |
| Recolour | fixed / CRP | move a shape to another palette entry, or a new one; merge or split palette entries |
| Holes | +1 / -1 | add or remove a hole in a compound path (once the hole-bearing sets exist) |
| Swap chains | -- | parallel tempering: adjacent temperatures exchange states |

**Every jump move is paired with its reverse**, and the acceptance ratio carries both proposal densities and the
Jacobian. A data-driven proposal that cannot report its own density is not allowed: that is the classic way a
reversible-jump sampler becomes silently wrong.

**Parallel tempering.** Layer orders and part counts are separated by low-probability valleys; a single chain stays in
whichever it started in. A ladder of chains at increasing temperature (4 by default) swaps states, and the chains also
give the convergence diagnostic (rank-normalized R-hat, Vehtari et al. 2021).

**The output.** Over the chain's post-burn-in samples:
- **The submitted SVG** is the highest-posterior sample by default. E3 compares it with the Bayes-optimal choice (the
  sample with the highest expected fidelity under the posterior) and with the most probable structure.
- **The posterior summary:** each shape's probability of existing (matched across samples by whole-shape IoU, as layer
  recovery matches), each pair's order probability, and the spread of each hidden part.

## Decision 5: speed and determinism

**An incremental likelihood.** A move changes one shape (or two). Only the pixels in its bounding box, grown by three
blur widths, change; the composite there is recomputed from the shapes that touch it, and the likelihood delta is
summed over that box alone. This is the incremental API 0022 Decision 6 planned and never built. A nuisance move (the
warp, blur) is global and re-renders everything; it is proposed less often.

**The budget is 1-10 s per image** at 128 px, about 10k-100k moves, against the pipeline's 152 ms. That is a leaderboard
budget. The browser's live chain (phase P5) runs shorter.

**Determinism, as everywhere else in the games core.** Randomness from `pcg.rs`'s PCG32; transcendentals (the
Student-t's logarithm, the Gaussian draws) from `libm`; no hash-map iteration order in anything that decides. A seed
is the same chain natively and in WASM, pinned by a digest in the style of `libs/rl`'s `determinism.json`.

## Decision 6: the benchmark grows with it

The sampler's claims need sets that can falsify them. All of these are new sets or a new protocol, never edits to
`copycat.clean.v1`'s sets, which stay as they are so 0022's numbers stay comparable.

- **`corrupt.v2`:** a second corruption family with a different warp (a random spline displacement plus a small
  rotation and scale) and a different blur kernel. It exists to check generalization: a warp model that has learned
  `corrupt.v1`'s sinusoid will not transfer.
- **`synthetic.v2`:** compositions from a small grammar: holes (subpaths of opposite winding, which the subset's
  non-zero rule already draws), mirror symmetry, rotational repetition, nested parts. The rasterizer gains a test of
  holes against its supersampled oracle.
- **`icons`:** Twemoji (CC-BY 4.0), as 0022 proposed, with its attribution checked in.
- **A second, more complex icon set:** flat-colour emoji with dozens of parts and many holes. Candidates are Noto Emoji
  (Apache-2.0) and Fluent Emoji Flat (MIT). Licences are confirmed from the source before anything is downloaded.
- **A converter** brings icons into the subset: transforms flattened, files with gradients, masks, text or strokes left
  out (strokes are 0022's v2 question), each set split into `train` and held-out.
- **Protocol `copycat.clean.v2`** scores every set and every corruption family, with breakdowns per set and per
  severity, ranked by mean clean fidelity across sets. `copycat.clean.v1` keeps running beside it.

## Decision 7: a sampler configuration is a run

0022 left open how a searching entrant is recorded. Here:

- **A `copycat.mcmc` run is one sampler and prior configuration applied, in lockstep, to every image of a split**
  (`train` by default). It is a trainer algorithm (`redqueen train copycat.mcmc --game copycat ...`), so it is recorded
  like every other run, with `config.paradigm = "inference"` -- a fourth run vocabulary for the frontend's `runMeta`
  beside evolution, RL and supervised.
- **A generation is a block of moves** on every image. It records:
  - fitness: the mean log-posterior per image (best and mean across the tempering ladder's cold chains);
  - diversity: the mean posterior entropy of the shape count;
  - extras: acceptance rate per move type, swap rate between temperatures, R-hat, the shape count, and the clean-fidelity
    columns on the split (the `train` splits have clean answers too).
- **Artifacts:** each image's submitted SVG and posterior summary.
- **The champion is the configuration** (and, after E4, the learned prior). `copycat.clean.v1` and `.v2` score every
  finished `copycat` run that is not an experiment run as a `run:<id>` entrant, by re-running its configuration on the
  held-out images under the protocol's timing. Experiment runs stay off the leaderboard; a winning arm is promoted as a
  plain run.

## Architecture

| Piece | Where |
|---|---|
| Scene, prior, forward model, incremental likelihood, the warp on outlines | `libs/games/rust/core/src/copycat_scene.rs` |
| Moves, reversible jumps, tempering, the sampler | `libs/games/rust/core/src/copycat_mcmc.rs` |
| Python face | `games.copycat.Sampler`: `step(n)`, `state()`, `stats()`, `samples()` |
| Browser | WASM `CopyCatSampler`: a chain running live on the page |
| New sets, `corrupt.v2`, the converter, `copycat.clean.v2` | `libs/arena/src/arena/copycat_sets.py` and siblings; the evaluator's protocols |
| Runs | the trainer's `copycat.mcmc`, its params model in `jobcore.algorithms` |
| Leaderboard | `copycat.clean.v1`/`.v2` score finished `copycat` runs |
| Experiment report | `experiment.copycat-report`: per image, each arm against its baseline, a paired permutation test (as `.rl-report` does) |

## Phases

| Phase | What | Checked by |
|---|---|---|
| P0 | this doc | review |
| P1 | the scene, prior and forward model; the incremental likelihood | off the books: random move sequences, incremental against full render to 1e-9; the warp on outlines against a warped full render |
| P2 | the sampler and every move | off the books (below) |
| P3 | the Python face, `copycat.mcmc`, run entrants on the leaderboard, the report, golden cases | CI and golden tests; one small smoke run |
| B1-B5 | `corrupt.v2`, `synthetic.v2`, `icons`, the second icon set, `copycat.clean.v2` | alongside P1-P2: independent of the sampler |
| P4 | experiments E0-E4 | registered runs |
| P5 | the page: a flipbook of posterior samples, existence and order probabilities, the live chain | the in-app browser |

### Off the books: proving the sampler correct before trusting a number

Small, unregistered scripts on at most 8 `train` images, for finding bugs, not for deciding anything. Nothing is
registered as a run until the first four pass:

1. **Prior recovery.** With the likelihood switched off, the chain must sample the prior: its marginals of `K`, the
   types, the parameters and the palette size match the prior's. Any wrong Jacobian or proposal density shows up here.
2. **Geweke's joint test** ("getting it right", 2004): scenes and images simulated from the model, compared with scenes
   from the sampler on simulated images.
3. **Toy posteriors with known answers.** One circle in noise: the posterior on its centre and radius matches a grid
   computation. Two overlapping discs: the order probability matches exact enumeration.
4. **Determinism:** native and WASM chains produce the same digest.
5. **Trace plots and acceptance rates** on a handful of images, to set the move weights and burn-in.

## Experiments

Every experiment is an `experiments/copycat-mcmc-vN.yaml`: arms x chain seeds (3-5), recorded as runs, reported by
`experiment.copycat-report`. **Decisions are made on `train` splits**; held-out images are scored only when an arm is
promoted to the leaderboard. Each experiment has a gate: what it must show for the next one to be worth running.

| Exp | Question | Arms (each adds to the one before) | Gate |
|---|---|---|---|
| **E0** structure | Does sampling the posterior beat the greedy pass? | `pipeline` (zero moves: must reproduce the leaderboard's structured entrant exactly), `local` (nudges and Gibbs colours), `+rj` (birth, death, split, merge, retype), `+order` | `+rj` fixes `medium`'s missed parts (0.93x toward 1.0x) or `heavy`'s layer recovery beyond the CI. If not, study mixing before anything else |
| **E1** forward model | Does modelling the corruption pass the warp cap? | `+blur`, `+warp`, `+robust`, `+background` | `medium`/`heavy` gain at least 1 dB, **and** the gain holds on `corrupt.v2`. A gain only on `corrupt.v1` means the warp is fitting the benchmark |
| **E2** compute | What does fidelity cost? | moves per image 1k / 10k / 100k; tempering chains 1 / 4 / 8; started from the pipeline or from empty; per set | from empty comes near from the pipeline. The fidelity-against-time curve is the cost-tradeoff chart, and it checks the 1-10 s budget holds on the icon sets |
| **E3** decision and calibration | Which sample to submit; is the posterior honest? | highest-posterior sample, Bayes-optimal sample, most probable structure | a reliability diagram: shape-existence and order probabilities within about 10% of how often they are right |
| **E4** a learned prior | Learn the prior's hyperparameters on `train` (type frequencies, shape count, warp scale) | fixed prior, prior learned per set, prior learned on `synthetic` and tested on `icons` | at least 0.5 dB on `train` without losing any on `clean`; then promote to the leaderboard |

**Statistics:** arms are compared image by image (every arm sees the same images), mean ± 95% CI over images and chain
seeds.

**Compute:** at about 5 s per image, an arm-seed on 96 `train` images is about 8 minutes; an experiment of 4 arms x 3
seeds is about 2 hours on one scheduler worker.

**What success looks like:**
- Held-out clean fidelity clearly above the structured pipeline's 30.8 dB on `copycat.clean.v1`, and ahead on every set
  of `copycat.clean.v2`.
- Layer recovery on `heavy` of at least 0.85 (the pipeline: 0.75).
- A calibrated posterior, shown on the page.

## The page

No new page (the Copy Cat page is where this belongs):
- **Watch** gains, for the sampler's entrant, a flipbook of posterior samples, each shape shaded by its probability of
  existing, and the ambiguous order pairs called out.
- **Play** runs the sampler live on a dropped-in image (WASM, a shorter budget), the chain's log-posterior and accepted
  moves drawn as it goes.
- **The run page** draws the chain's curves like any run, with the inference vocabulary.

## Risks

- **Mixing.** Reversible-jump samplers on continuous geometry mix slowly, and tempering may not be enough. E0's gate and
  E2's "from empty" arm measure it; data-driven births are the main defence.
- **Silent incorrectness.** A wrong Jacobian still produces plausible SVGs. The off-the-books tests exist for this, and
  are kept as unit tests once they pass.
- **Overfitting the benchmark.** The warp and blur models could learn `corrupt.v1`. Generic families, `corrupt.v2`, and
  E1's gate are the guard.
- **Cost.** 10-100x the pipeline's time. E2 reports it as a tradeoff, and the leaderboard shows time per image beside
  fidelity, as it already does.
- **Many-part icons.** Emoji have 10-40 parts against `synthetic`'s 2-8, so the state space is far larger. E2 per set
  shows whether the budget holds.

## Open questions

- **Which second icon set:** Noto Emoji or Fluent Emoji Flat, once their licences and the share of files the subset can
  hold are checked.
- **Strokes.** Out of the subset still; many icons need them, and a stroke is a natural shape type for the sampler
  later.
- **Gradient fills.** The same, and the sampler could infer one as easily as a flat fill.
- **Amortization.** Once the sampler works, a learned proposal (a network suggesting births from the image) could cut
  the moves needed. Only if E2 shows compute is the limit.

## Explicitly out of scope

- Everything 0022 rules out: photographs, large learned models that write SVG, learned perceptual metrics as a reward,
  animation.
- Variational inference. Sampling first, because its correctness can be tested exactly (above).

## Sources

- P. J. Green, "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination", Biometrika, 1995.
- Z. Tu and S.-C. Zhu, "Image segmentation by data-driven Markov chain Monte Carlo", IEEE TPAMI, 2002.
- Z. Tu, X. Chen, A. Yuille and S.-C. Zhu, "Image parsing: unifying segmentation, detection, and recognition", IJCV,
  2005.
- V. Mansinghka et al., "Approximate Bayesian image interpretation using generative probabilistic graphics programs",
  NeurIPS 2013: <https://arxiv.org/abs/1307.0060>
- T. Kulkarni et al., "Picture: a probabilistic programming language for scene perception", CVPR 2015.
- J. Geweke, "Getting it right: joint distribution tests of posterior simulators", JASA, 2004.
- D. Earl and M. Deem, "Parallel tempering: theory, applications, and new perspectives", PCCP, 2005.
- A. Vehtari et al., "Rank-normalization, folding, and localization: an improved R-hat", Bayesian Analysis, 2021:
  <https://arxiv.org/abs/1903.08008>
- Twemoji: <https://github.com/jdecked/twemoji>; Noto Emoji: <https://github.com/googlefonts/noto-emoji>; Fluent Emoji:
  <https://github.com/microsoft/fluentui-emoji>

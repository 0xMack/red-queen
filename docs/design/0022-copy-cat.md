# 0022 — Copy Cat: a game of drawing an image back as an SVG

Status: **Proposed** (2026-10-24). Nothing is built yet.
Relates to: [0007](0007-representations-leaderboards-and-tradeoffs.md) (interfaces, held-out protocols, an entry as a
vector of measurements), [0009](0009-client-side-inference-at-scale.md) (a Rust core shared by PyO3 and WASM),
[0010](0010-reinforcement-learning.md) (the `libs/rl` core and its determinism rule), [0011](0011-multi-armed-bandits.md)
(the last game added, and the shape of this doc).

## Context

Every game so far is a toy whose result is a number. **Copy Cat** is a game whose result is useful in itself. The
player gets a raster image and draws it back as an SVG. It scores for likeness, and it may only use a fixed number of
shapes. A good player can be used as a vectorizer.

### What exists (surveyed 2026-10-24)

| Family | Examples | Strength | Weakness |
|---|---|---|---|
| Tracers: segment into colour regions, fit curves to their edges | Potrace, VTracer, Illustrator Image Trace, Vectorizer.AI (commercial; generally rated best) | fast; very good on logos and flat art | thousands of fragmented shapes on photos; no shape budget |
| Per-image optimization: render, compare, adjust, repeat | DiffVG (a differentiable rasterizer), LIVE (adds paths layer by layer), Optimize & Reduce (prunes paths), Bézier Splatting (NeurIPS 2025, ~10x faster than DiffVG), LayerPeeler, COVec (CVPR 2026) | near-perfect fidelity given enough paths | slow (seconds to minutes, GPU); compactness and editable layers are the open problems |
| Models that write SVG code | StarVector, OmniSVG, VFig, frontier VLMs; RLRF (NeurIPS 2025) rewards a model for rendered likeness and short code | amortized: one generation per image; semantic structure | large models and GPU training; out of reach at this project's scale |
| Stroke painting | Learning to Paint (RL, rewarded by similarity), `primitive` / geometrize (hill climbing), Alsing's evolved Mona Lisa | the game framing exactly; tiny, legible outputs | rarely measured against tracers |
| Animating an SVG | AniClipart, LiveSVG: fit the SVG's control points to a video from a pretrained video diffusion model | real motion that keeps the drawing's identity | depends on large pretrained video models |

Two conclusions shape this doc:

- **Beating the best vectorizers on general quality is not a realistic goal** for a from-scratch, CPU-scale project.
  Per-image optimization already matches pixels when shapes are unlimited, and commercial tracers are very good.
- **Fidelity per shape, per byte and per second is under-measured.** Papers report fidelity at whatever path count
  their method lands on, and tracers have no budget at all. A leaderboard that holds the budget fixed measures exactly
  what tracers are bad at. It is the same move as Checkers' cost tradeoffs (0007 Decision 3). It also has a real use:
  an SVG placeholder shown while a photo loads (SQIP already does this with `primitive`) must be under ~1 KB, where
  search beats tracing.

## Decision 1: an entrant turns an image into an SVG; the protocol renders and scores it

- **An entrant is any `image -> SVG` function.** Our algorithms, VTracer and a `primitive`-style hill climb are all
  entrants. The protocol never trusts an entrant's own render or its own shape count.
- **The protocol parses the SVG into a restricted subset and renders it with our own rasterizer:**
  - `rect`, `ellipse`, `polygon`, and `path` with `M`/`L`/`C`/`Q`/`Z`;
  - a solid `fill`, with opacity;
  - no strokes, gradients, filters, transforms, masks or text in v1.

  A tracer's output fits the subset. Anything outside it is refused, not approximated.
- **Shapes and bytes are counted from the parsed subset.** The parsed shapes are re-serialized canonically before
  bytes are counted (coordinates rounded to 0.1 px in a `viewBox` the image's size), so formatting can't win.

## Decision 2: a shape budget, not a shape penalty

A score like "likeness - λ x shapes" hides the tradeoff in λ and invites confetti: thousands of sub-pixel shapes. The
protocol instead fixes **budgets** and scores each one separately:

- shape budgets of **10, 50 and 200**;
- a byte budget of **1 KB** for the placeholder use case.

An entrant may submit one SVG per budget. One that can't target a budget, such as a tracer, is tuned
(`filter_speckle`, `color_precision`, ...) and scored at every budget its output fits under. The leaderboard ranks by
the 50-shape score and shows all budgets. The page's headline chart is the Pareto curve: fidelity against shape count,
and against time per image, with every entrant on it.

## Decision 3: the score

- **Fidelity is PSNR in sRGB**, the mean over held-out images. PSNR is plain arithmetic, so it is deterministic and
  cheap. **SSIM** is reported beside it, as a second opinion on structure.
- **Learned perceptual metrics (LPIPS, DreamSim) are report-only, if ever.** They need a pretrained network the
  project didn't build, and making one the reward would mean optimizing against it.
- **The render is scored at two resolutions:** the image's own, and 2x. A shape that only works because of how it
  lands on the pixel grid (sub-pixel slivers, aliasing tricks) loses at the second.
- **The cost columns are time per image and peak memory** (`arena.costs`). An entrant that takes 100x longer for
  +0.5 dB shows it.

## Decision 4: the rasterizer is Rust, in the games core, with no dependencies

- **One rasterizer** (`libs/games/rust/core/src/raster.rs`): scanline with analytic coverage anti-aliasing, curves
  flattened to lines, non-zero fill, `over` compositing in sRGB.
- **The same code serves three uses:** the protocol scores with it, training loops render with it, and the browser
  draws and scores with it through WASM. A score is then the same number everywhere.
- **No dependencies.** `tiny-skia` would be quicker to adopt. But the games core has no dependencies today, and the
  determinism rule (0010 Decision 2: a seed trains bit-identically natively and in WASM) is easier to keep with only
  `+ - * /` and `sqrt`, which IEEE rounds exactly. If a transcendental is needed, it comes from `libm`, as in
  `libs/rl`.
- **An incremental API for search:** add, move or remove one shape, and re-score only its bounding box. Hill climbing
  and evolution spend almost all their time here: about 10^4-10^5 candidate renders per image, at 128 px.
- **A soft, differentiable variant is a later phase (phase 3),** not v1: splatting-style Gaussians along the edges
  (after Bézier Splatting), with gradients written by hand, the way `libs/rl` does backprop.

## Decision 5: the target sets

- **`synthetic`:** images drawn by our own rasterizer from K random shapes. **The answer is known**: K shapes reach
  PSNR = infinity. It is the sanity test every algorithm must pass first (can it recover a 10-triangle picture with
  10 triangles?), and it is free to generate from a seed.
- **`flat`:** icons and emoji from permissively licensed sets, rasterized once and checked in as small PNGs, with their
  source path counts kept as an approximate answer. This is tracer territory, and where the comparison with VTracer is
  fair to VTracer.
- **`photo`:** CC0 photographs, downscaled to 128 px on the long side. This is the placeholder use case, and where the
  shape budget matters most.
- **The protocol fixes the split.** Each set has training images and held-out images, and the leaderboard scores only
  the held-out ones (0007 Decision 2).

## Decision 6: the interface is the shape vocabulary

Following 0007, the pair "what the player sees" + "what an action means" is a versioned interface:

- **Interfaces:** `copycat/rgb128.v1+triangles.v1`, `+ellipses.v1`, `+rects.v1`, `+paths.v1` (closed cubic Béziers
  with up to N segments).
- **Every shape's colour is part of its action.** A search may also set a shape's colour to the best constant colour
  under it, in closed form, as `primitive` does.
- **The observation for a learned player** is the target and the current canvas, stacked, plus the error map.
  Algorithms that search per image don't need one.

## The algorithm ladder

Each rung is an experiment against the one below, under the same protocol:

1. **Baselines:**
   - `primitive`-style hill climbing: random shape, mutate it, keep it if the score improves, repeat;
   - simulated annealing;
   - VTracer (a Rust crate) as an external entrant, at several settings.
2. **Evolution (`evolve`):**
   - a genetic algorithm over shape lists, with crossover that swaps whole shapes;
   - CMA-ES on one shape's parameters at a time;
   - cleanup moves: a "remove the least useful shape" pass (Optimize & Reduce's idea), so a run can trade shapes for
     fidelity.
3. **Gradients:** the soft rasterizer plus Adam (`libs/rl` has one). Hill climbing chooses the topology (which shapes,
   in what order) and gradient descent refines their coordinates.
4. **A learned player (`libs/rl`):** Learning to Paint, rebuilt. A policy sees the target and the canvas and places
   shapes, rewarded by the gain in PSNR per shape.
   - It will be worse than per-image search at the same budget.
   - Its value is speed (one forward pass per shape, in the browser) and a starting point that rungs 1-3 refine.
   - The tracked question: does starting from the policy beat starting from random shapes, at equal wall-clock time?
5. **Animation:** its own design doc, later. A version within reach is fitting one SVG to every frame of a short clip
   while keeping each shape's identity, rewarded for likeness and penalized for jitter, with no diffusion prior. Making
   a still image move by itself needs a generative model of motion. That is a research project, not a phase.

**What success looks like:**
- the Pareto curve has our entrants above VTracer and `primitive` at 10-200 shapes on `photo`;
- they are competitive on `flat`;
- a 1 KB placeholder is visibly better than SQIP's at the same size.

None of these claims "better than SOTA vectorization". All of them are measurable here.

## The game, as a player sees it

- **Watch:** an entrant draws the target one shape at a time, with the error map fading as it goes. Of every game in
  the project, this is the one where what the algorithm is doing is most visible.
- **Play it yourself:** place and drag triangles to match the target, with your PSNR against the leaderboard at your
  shape count.
- **Every result exports:** the SVG downloads, so the page is a working vectorizer for any image a visitor drops in. It
  runs in their browser, with nothing uploaded.

## Architecture (reusing what exists)

| Piece | Where |
|---|---|
| Rules, rasterizer, SVG subset parser and serializer, shape mutations | `libs/games/rust/core/src/copycat.rs`, `raster.rs` |
| Python face | `games.copycat` (PyO3), `games.interfaces` registrations |
| Protocol, budgets, target sets and their splits | `libs/arena/src/arena/copycat.py`; the evaluator protocol `copycat.fidelity.v1` |
| Training | `jobs/trainer`: `copycat.hillclimb`, `copycat.anneal`, `copycat.ga`, later `copycat.gradient` and `rl.*` on `copycat` |
| Browser | WASM `CopyCatGame`; `app/games/copycat.ts` (a `GameModule`), a `CopyCatStage` |

Per-image search is not "training" in the usual sense: the run is the search on one image. How it is recorded (one run
per image, or one run per held-out set) is an open question. Whichever it is, the protocol times it the same way.

## Phases

1. **The game:** rasterizer and parser with tests (coverage against a supersampled oracle; `synthetic` round trips),
   the `synthetic` set, the protocol, and hill-climbing and VTracer entrants. This is enough to rank.
2. **The page:** Watch, Play, export, the Pareto chart; `flat` and `photo` sets.
3. **The ladder:** annealing, GA, CMA-ES and cleanup moves, then the soft rasterizer and gradient refinement. Each one
   is an experiment.
4. **The learned player,** and the question of whether its starting point pays.
5. **Animation:** a separate design doc.

## Open questions

- **Licences for `flat` and `photo`:** which icon and emoji sets, and which CC0 photo source. A rasterizing tool for the
  icons (`resvg` would be a one-off generator dependency, not a runtime one).
- **Whether `paths.v1` should be cubic only.** Quadratic curves are cheaper to flatten, and VTracer emits both.
- **How to record per-image search runs:** see "Architecture".
- **Colour space for the error:** sRGB PSNR is the standard number. A perceptual space (OKLab) may be a better
  *reward* even if it isn't the reported score: sRGB distances track perceived hue and chroma differences poorly, and
  OKLab's track them more closely. OKLab needs a cube root, so it would come from `libm`.

## Explicitly out of scope

- Strokes, gradients and text in the SVG subset (v1).
- Large learned models that write SVG code (StarVector-style), and LPIPS as a reward.
- Animation beyond a design doc.

## Sources

- Optimize & Reduce: <https://arxiv.org/abs/2312.11334>
- Bézier Splatting: <https://arxiv.org/abs/2503.16424>
- LayerPeeler: <https://arxiv.org/abs/2505.23740>
- Layered vectorization via semantic simplification: <https://arxiv.org/abs/2406.05404>
- COVec (Clair Obscur): <https://arxiv.org/abs/2511.20034>
- LayerTracer: <https://arxiv.org/abs/2502.01105>
- RLRF: <https://arxiv.org/abs/2505.20793>
- VFig: <https://arxiv.org/abs/2603.24575>
- VectorGym: <https://arxiv.org/abs/2603.29852>
- AniClipart: <https://arxiv.org/abs/2404.12347>
- LiveSVG: <https://papers.cool/arxiv/2605.30174>

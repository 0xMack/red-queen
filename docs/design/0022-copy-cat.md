# 0022 — Copy Cat: recover the clean, layered SVG behind a generated logo or icon

Status: **Phases 1-2 built, phase 3's page, and phase 4's first rung** (2026-10-24).
- Phase 1 built the rasterizer, the subset, `synthetic` + `corrupt.v1`, protocol `copycat.clean.v1` and the
  baselines; see "Phase 1 results".
- Phase 2 built the structured MDL pipeline; see "Phase 2 results".
- Phase 3 so far is the page, `/games/copycat`; see "The page". The `icons` set and the `wild` side-by-sides are
  still to come.
- Phase 4 so far refines each shape point by point under L, and describes a smooth outline by its knots
  (fitted by least squares), and cuts a part of one colour in two where two shapes overlap. A regular-polygon candidate was measured and not
  kept. See "Phase 4 results".

Phases 3-6 are proposed.
Relates to: [0007](0007-representations-leaderboards-and-tradeoffs.md) (held-out protocols, an entry as a vector of
measurements), [0009](0009-client-side-inference-at-scale.md) (a Rust core shared by PyO3 and WASM; ONNX Runtime Web),
[0010](0010-reinforcement-learning.md) (determinism; the `libs/rl` core), [0011](0011-multi-armed-bandits.md) (the last
game added).

## Context

### The use case

People ideate logos, icons and graphics with image and video generators, because that's where the best ideas come
from. But a generated image can't go on a production page as an icon. It is a raster image, with no transparent
background, and it is full of noise the designer never wanted:
- soft edges;
- faint gradients and texture;
- compression artifacts;
- colour that drifts across what should be one flat fill;
- a background that is almost, but not quite, empty.

What they want is the **design the image depicts**. That means an SVG that:
- has a transparent background and only the subject;
- uses a handful of flat colours, with smooth edges and exact geometry where the design has it (circles, straight
  lines, symmetry);
- is small and loads quickly;
- is built from **whole, stacked parts in layers**, so it can be edited and, later, animated. A circle partly behind a
  square should be a circle, not a crescent.

Animating it is a separate, later problem. This doc only has to produce SVGs that make animation easy.

### Why existing vectorizers fall short here

**Tracers reproduce the pixels faithfully**, and for a generated image that is the wrong goal. Examples: Potrace,
VTracer, Illustrator Image Trace, and Vectorizer.AI (commercial, generally rated best). They reproduce the noise too:
- jagged edges that follow the pixel staircase and the blur;
- speckles;
- dozens of near-identical colours;
- regions cut out of each other instead of stacked.

None of them isolates the subject.

| Family | Examples | Relevant here |
|---|---|---|
| Tracers | the above | the baselines to beat; fast; good on already-clean art |
| Per-image optimization | DiffVG, LIVE, Optimize & Reduce, Bézier Splatting (NeurIPS 2025) | the refinement machinery; LIVE and O&R aim at fewer paths |
| Layered and occlusion-aware vectorization | Semantic simplification (CVPR 2024), LayerPeeler (2025, flat-colour SVG data), AmodalSVG (2026: peel layers, complete hidden parts) | the closest research to this use case; AmodalSVG's code isn't released |
| Models that write SVG | StarVector, OmniSVG, VFig, RLRF; text-to-SVG generators (Recraft V4, LottieFiles) | the competing workflow: skip the raster entirely. Ours keeps the stronger image models for ideation |

**The opening:** recover the design, not the pixels, and measure that directly. Tracers are optimizing a different
objective, so beating them on this task is a realistic goal.

## Decision 1: the benchmark has known clean answers

Likeness to the input is the wrong score: it rewards reproducing the noise. The benchmark therefore starts from clean
SVGs and corrupts them the way a generator would:

- **`synthetic`:** random compositions of stacked primitives and Bézier blobs, in 2-6 palette colours, generated from a
  seed. Every part is known *whole*, including the parts hidden behind others. So layer recovery is scored exactly.
- **`icons`:** flat-colour, multi-part icons and emoji from permissively licensed sets (Twemoji, CC-BY 4.0, is the
  leading candidate). They are converted once into the SVG subset (Decision 2) and checked in.
- **The corruption model `corrupt.v1`** is versioned and seeded, like an observer. Each image gets a mix of:
  - blur and soft anti-aliasing;
  - Gaussian noise and film grain;
  - JPEG at quality 50-90;
  - per-region colour drift and a faint shading gradient;
  - a slight warp;
  - a background: near-flat, a soft gradient, or clutter.

  Images are generated once by a script (Pillow for the JPEG step) and checked in beside their clean SVGs. The levels
  are recorded per image, so results can be broken down by corruption.
- **`wild`:** real AI-generated logos and icons, with no clean answer. It is scored by blind pairwise preference against
  the baselines (Decision 3).
- **Splits:** each set has training and held-out images, and the leaderboard scores only the held-out ones (0007
  Decision 2).

## Decision 2: an entrant turns an image into an SVG; the protocol renders and scores it

- **An entrant is any `image -> SVG` function.** Our pipeline, VTracer and a `primitive`-style hill climb are all
  entrants.
- **The protocol parses a restricted subset and renders it with our own rasterizer:**
  - `rect`, `circle`, `ellipse`, `polygon`, and `path` with `M`/`L`/`C`/`Q`/`Z`;
  - a solid `fill`, with opacity;
  - `<g>` for layers (no transforms).

  No strokes, gradients, filters, masks or text in v1. Anything outside the subset is refused, not approximated. The
  output must be **transparent wherever it draws nothing**.
- **Shapes, path nodes and bytes are counted from the parsed shapes,** re-serialized canonically (coordinates to 0.1 px
  in a `viewBox` the image's size), so formatting can't win.

## Decision 3: the score is a vector, led by fidelity to the clean design

Each column compares against the clean SVG, never the corrupted input:

| Measure | How |
|---|---|
| **Clean fidelity** (the rank) | PSNR of the output against the clean render, both composited over black *and* over white. Averaging the two scores colour and transparency together: a leftover background, a halo or a hole all lose. Reproducing the noise loses too, because the clean render has none |
| Alpha IoU | intersection-over-union of the output's coverage and the clean coverage |
| Layer recovery | each clean part is matched to an output shape by *whole-shape* IoU: each one rendered alone, occlusion ignored. The mean over parts: high only if the parts are separated **and** their hidden portions completed |
| Structure | shapes, path nodes and palette size, each as a ratio to the clean SVG's |
| Smoothness | total absolute turning angle along the flattened outlines, as a ratio to the clean SVG's. A jagged edge turns far more than a smooth one |
| Size and cost | canonical bytes, time per image, peak memory (`arena.costs`) |

- **The leaderboard ranks by clean fidelity and shows every column.**
- **The page's headline chart** is clean fidelity against shapes, with every entrant on it.
- **`wild` is reported apart:** win rate in blind side-by-side pairs against each baseline. The judges are people
  first; a vision-model judge only if it agrees with the people on a calibration set.

## Decision 4: our entrants minimize description length

**Minimum description length (MDL):** find the shortest SVG that explains the image, where whatever the shapes leave
unexplained costs bits too. The objective:

```
L(svg) = bits(svg) + bits(input | render(svg))
```

- **`bits(svg)` is a code length per element:**
  - a shape type;
  - a colour as an index into the palette (cheap) or a new colour (expensive);
  - each coordinate at a fixed precision.

  So the design priors come out as code lengths:
  - **a circle (3 numbers) is cheaper than the four Bézier segments that draw it**, so near-circles snap to circles;
  - a straight run is one segment;
  - a shape that is nearly symmetric is cheaper encoded as half and a mirror;
  - a reused colour is cheaper than a new one, so the palette stays small.
- **`bits(input | render)` is the residual under a noise model.** Its scale σ is estimated from the image itself (from
  flat regions). Errors at the noise level then cost almost nothing, and only real structure is worth a shape.
  - This is what removes the noise. Nothing has to say "denoise".
  - It replaces a hand-tuned "likeness - λ x shapes": λ comes from σ.
- **Background and transparency are part of the model.** The input is explained as the SVG composited over a
  background model: a constant colour, or a low-order gradient, fitted from the image border. Pixels the background
  explains are transparent.
  - Clutter breaks this simple model. `corrupt.v1`'s clutter level measures exactly where.
  - A pretrained matting or segmentation model (run through ONNX, 0009) may be an optional stage. It is the one part
    not built here, so an entrant that uses one is labelled as such on the leaderboard.

## Decision 5: parts are stacked, not cut out

- **Shapes have a paint order and may extend under the shapes above them.**
- **Under MDL, the hidden parts complete themselves.** Pixels behind an occluder cost nothing either way, so the
  cheapest description of "a circle behind a square" is the whole circle, not a crescent. Layer recovery (Decision 3)
  checks that it happened.
- **The output groups shapes into `<g>` layers by part,** bottom to top, so each part can be moved, recoloured or
  animated on its own. This is the hand-off to a later animation phase.

## Decision 6: the rasterizer is Rust, in the games core, with no dependencies

- **One rasterizer** (`libs/games/rust/core/src/raster.rs`): scanline with analytic coverage anti-aliasing, curves
  flattened to lines, non-zero fill, `over` compositing, and RGBA output.
- **The same code serves three uses:** the protocol scores with it, the entrants optimize with it, and the browser
  draws with it through WASM. A score is the same number everywhere.
- **It also renders one shape alone,** which layer recovery needs.
- **No dependencies.** The determinism rule (0010 Decision 2) is easier to keep with `+ - * /` and `sqrt`. Anything
  transcendental (OKLab's cube root) comes from `libm`, as in `libs/rl`.
- **An incremental API for search:** change one shape and re-score only its bounding box.
- **A soft, differentiable variant** (splatting-style, after Bézier Splatting, gradients written by hand) is the
  refinement phase's.

## The ladder

Each rung is an experiment against the one below, on the same held-out sets:

1. **Baselines:** VTracer at several presets, Potrace run once per palette colour, and a `primitive`-style hill climb.
   Vectorizer.AI and Recraft join only the `wild` side-by-sides, run by hand (their terms and cost; see "Open
   questions").
2. **The structured pipeline**, which is likely where most of the win is:
   1. a palette: k-means in OKLab, with clusters merged while that shortens L;
   2. regions;
   3. contours fitted by model selection under L: primitive, line run or Bézier segments, with corners kept where the
      code pays for them;
   4. layer order from how regions meet (the region whose boundary bulges into the other is usually behind it);
   5. hidden parts completed: the simplest shape that covers the visible region and stays inside the visible region
      plus its occluders.
3. **Search over the whole structure under L:** merge, split, snap, re-order and delete moves, by hill climbing,
   annealing or a GA (`evolve`). Optimize & Reduce's pruning becomes one move among them.
4. **Gradient refinement:** the soft rasterizer plus Adam (`libs/rl` has one), refining coordinates and colours with
   the structure held fixed.
5. **Learned parts, only where an experiment shows they pay.** Candidates:
   - a learned noise estimate;
   - a corner detector;
   - an RL policy (`libs/rl`) proposing shapes as a starting point for search.

**What success looks like:**
- On `synthetic` and `icons`, at every corruption level, our pipeline beats VTracer on:
  - clean fidelity;
  - alpha IoU;
  - layer recovery;
  - smoothness;
  - shape count.
- On `wild`, people prefer it to VTracer in most blind pairs.
- Preferred to Vectorizer.AI and Recraft is the stretch goal.

## The game, as a player sees it

- **Watch:** the pipeline's stages, one after another:
  1. the palette;
  2. the regions;
  3. contours snapping to circles and lines;
  4. an **exploded view of the layers**, the stack pulled apart in depth, with hidden parts completed.
- **Drop in your own image,** compare against VTracer side by side, and download the layered SVG. It all runs in the
  visitor's browser, with nothing uploaded.
- **Play it yourself:** rebuild a `synthetic` target with as few shapes as you can, scored like any entrant.

## Architecture (reusing what exists)

| Piece | Where |
|---|---|
| Rasterizer, SVG subset parser and serializer, code lengths, shape moves | `libs/games/rust/core/src/copycat.rs`, `raster.rs` |
| Python face | `games.copycat` (PyO3) |
| Target sets, `corrupt.v1` (a generator script), protocol `copycat.clean.v1` | `libs/arena/src/arena/copycat.py`; the evaluator's `copycat.clean.v1` |
| Entrants | the evaluator's `copycat.clean.v1` entrants (the structured pipeline is `copycat_trace.rs`), then its search and gradient stages -- as `jobs/trainer` runs once they search; `rl.*` on `copycat` later |
| Browser | WASM `CopyCatGame`; `app/games/copycat.ts` (a `GameModule`) and a `CopyCatStage` |

Our entrants mostly *search per image* rather than train. How a run is recorded (one run per held-out set, with
per-image results as artifacts) is an open question. Whichever it is, the protocol times every entrant the same way.

## Phases

1. **Measurement first:** the rasterizer and parser, with tests (coverage against a supersampled oracle; subset round
   trips), `synthetic`, `corrupt.v1`, the protocol and its columns, and VTracer and hill-climb entrants. This alone
   shows how far tracers fall short, before we build anything to beat them.
2. **The structured pipeline under MDL,** with transparency and stacked layers. The first real entrant.
3. **The `icons` set, the page, and the first `wild` side-by-sides.**
4. **Search and gradient refinement,** each one an experiment.
5. **Learned parts,** where they pay.
6. **Animation:** its own design doc, starting from the layered output.

## Phase 1 results

Built as specified above, with these choices made along the way:

- **The rasterizer** (`raster.rs`) is exact area coverage (the font-rs accumulation scheme), checked against a 32 x 32
  supersampled non-zero oracle. Its one approximation is a pixel holding windings of both signs (where a
  self-intersecting outline crosses itself). The games core gained one dependency, `libm`, for the turning angles.
- **Layer recovery** takes each clean part's *best* match, not a one-to-one assignment.
- **Ratios** are summarized by geometric mean: a 10x and a 0.1x average to 1x.
- **The held-out set** is 24 `synthetic` images, 128 px, six per severity. `clean` is the control: the bare render on a
  flat background. Then `light`, `medium` and `heavy`. They are checked in under `libs/arena/src/arena/data/copycat/`.
- **Two "background deleted" entrants** are added, VTracer's and the hill climb's, each with its bottom layer removed.
  This is the fix a person would make by hand, so it separates "kept the background" from everything else.

Measured with `copycat.clean.v1` on all 24 held-out images:

| Entrant | Clean fidelity (dB) | Over black / white | Alpha IoU | Layer recovery | Shapes | Palette | Smoothness | ms / image |
|---|---|---|---|---|---|---|---|---|
| VTracer (defaults), background deleted | **20.2** ± 2.6 | 16.3 / 24.0 | 0.73 | 0.73 | 6.7x | 9.0x | 13.3x | 8 |
| Hill climb (64 shapes) | 11.4 ± 0.5 | 2.2 / 20.7 | 0.27 | 0.72 | 15.4x | 17.3x | 15.3x | 89 |
| Hill climb (64 shapes), background deleted | 11.4 ± 0.4 | 2.2 / 20.6 | 0.26 | 0.72 | 15.1x | 17.0x | 15.1x | 89 |
| VTracer (polygons) | 11.4 ± 0.4 | 2.2 / 20.6 | 0.27 | 0.74 | 7.2x | 9.7x | 13.6x | 5 |
| VTracer (defaults) | 11.3 ± 0.4 | 2.2 / 20.4 | 0.27 | 0.73 | 7.2x | 9.7x | 13.8x | 7 |
| VTracer (flat art) | 10.7 ± 0.5 | 2.2 / 19.2 | 0.27 | 0.61 | 1.0x | 1.3x | 2.0x | 2 |
| Hill climb (16 shapes) | 10.3 ± 0.3 | 2.3 / 18.3 | 0.27 | 0.70 | 4.0x | 5.0x | 4.0x | 39 |

The ratios are against the clean SVG, so 1x is right. Peak memory never rose above each entrant process's start-up peak
(~79 MiB) at this image size, so it doesn't separate anyone yet.

By severity, for the best tracer (VTracer, background deleted):

| Severity | Clean fidelity | Alpha IoU | Layer recovery | Shapes | Palette | Smoothness |
|---|---|---|---|---|---|---|
| clean | 28.3 | 0.97 | 0.74 | 1.0x | 1.3x | 1.7x |
| light | 22.4 | 0.88 | 0.78 | 6.1x | 8.2x | 9.6x |
| medium | 18.4 | 0.74 | 0.76 | 12.1x | 14.1x | 25.1x |
| heavy | 11.7 | 0.33 | 0.65 | 26.4x | 42.8x | 78x |

**What it shows:**

- **No baseline makes anything transparent.** Every one keeps an opaque background, so over black they score ~2 dB, the
  same as a blank canvas. Ranked as they come, the field is a tie at 10-11 dB.
- **Deleting VTracer's bottom layer is worth 9 dB**, but only while the background is one flat region. On `heavy`
  images (gradients, clutter), VTracer's background is many paths, and alpha IoU falls to 0.33.
- **Deleting the hill climb's background changes nothing.** Its later shapes repaint the background too.
- **The tracers reproduce the noise.** On clean art, VTracer's structure is nearly right: 1.0x shapes, 1.3x palette and
  1.7x turning. Every corruption level multiplies all three. At `heavy` it is 26x the shapes, 43x the colours and 78x the
  turning, all of it noise traced faithfully. The flat-art preset avoids that (1.0x shapes, 2.0x smoothness) by
  discarding detail, and its layer recovery drops to 0.51 at `heavy`.
- **Layer recovery is capped near 0.75, even on clean art.** VTracer leaves occluded parts as the crescents they look
  like. The hill climb's ellipses sometimes fit a whole part, but it has no notion of a part.
- **The hill climb is the worst on structure.** At 64 shapes, it has 15x the shapes and 17x the colours for the same
  fidelity as VTracer.

So phase 2's targets are concrete. On every severity, the pipeline needs to:
- reach `clean`-level fidelity (VTracer reaches 28 dB only on already-clean art, with its background deleted by hand);
- keep shapes, palette and smoothness near 1x as corruption rises;
- push layer recovery past 0.75.

Re-run with `uv run redqueen evaluate copycat` (or `python -m evaluator jobs/evaluator/specs/copycat.clean.v1.yaml`).

## Phase 2 results

The structured pipeline is `libs/games/rust/core/src/copycat_trace.rs` (the stages) and `copycat_fit.rs` (fitting one
region by code length), exposed as `games.copycat.trace` and the WASM `CopyCatGame.trace`. It is the evaluator's
`structured` entrant. It is untrained and searches nothing beyond each region's candidates, so it is recorded like the
baselines, with no run. Recording our later, searching entrants as runs is still the open question above. Its stages,
in order:

1. **Noise σ:** the median OKLab difference between neighbouring pixels, as a robust estimate.
2. **Background:** fitted to a 2-pixel border, either constant or a gradient, whichever is shorter. It is a cluster of
   its own whose centre varies per pixel.
3. **Palette:** k-means++ in OKLab from 16 seeds. Clusters merge while the residual the merge adds, in bits, is less
   than the 24-bit palette entry plus the shared boundary it saves.
4. **Labels:** a majority filter, then dissolving components that are too small or too thin (anti-aliased bands).
5. **Parts:** one per component, filled with its label's interior mean. Two components of one label join when one shape
   says them more briefly.
6. **Order:** each part is fitted with every other part free. Of two touching parts, the one whose shape reaches
   further under the other goes below.
7. **Shapes:** each part is fitted with the parts above it free. The candidates are circle, ellipse, rect, polygon and
   Bézier path (Schneider's fit, cut at corners), and the shortest `model bits + pixel_bits x misfit` wins.

Choices made along the way, each measured on the `train` split (96 images, `python -m arena.copycat_sets --split
train`; the held-out split was scored once, at the end):

- **A soft target.** An edge pixel counts as the fraction of the part's colour in it: its colour projected between the
  part's colour and what is across the edge. So contours and fits land between pixels. At the same settings this
  was worth +4.3 dB on `clean` (29.7 to 34.0). Before it, all of `clean`'s remaining error was within 2 px of an edge, and none in interiors.
- **Two thresholds where MDL alone wasn't enough.**
  - `merge_delta` (0.08): a part's own shading or drift, which a flat fill can't say anyway, would otherwise survive as
    a second colour.
  - `background_delta` (0.1): the generator's faint clutter is close to the background colour, but not close enough for
    the noise model to absorb it. Without this, clutter images had 2-3x the shapes.
- **Order by reach, not pairwise code length.** Comparing parts two at a time misses a part hidden under several
  others, because only one occluder at a time is free.
- **`pixel_bits` = 12** per wrongly covered pixel. It trades fidelity against smoothness: 8 was 0.6 dB worse, and 16
  added turning.

Measured with `copycat.clean.v1` on the same 24 held-out images as phase 1:

| Entrant | Clean fidelity (dB) | Over black / white | Alpha IoU | Layer recovery | Shapes | Palette | Smoothness | ms / image |
|---|---|---|---|---|---|---|---|---|
| **Structured pipeline** | **29.0** ± 2.4 | 29.9 / 28.1 | **0.95** | **0.86** | 1.0x | 1.0x | 1.1x | 76 |
| VTracer (defaults), background deleted | 20.2 ± 2.6 | 16.3 / 24.0 | 0.73 | 0.73 | 6.7x | 9.0x | 13.3x | 7 |
| Hill climb (64 shapes) | 11.4 ± 0.4 | 2.2 / 20.7 | 0.26 | 0.72 | 15.4x | 17.3x | 15.3x | 88 |

The rest of the field is as in phase 1. By severity, against phase 1's best:

| Severity | Clean fidelity | Alpha IoU | Layer recovery | Shapes | Palette | Smoothness |
|---|---|---|---|---|---|---|
| clean | 34.0 (VTracer 28.3) | 0.99 (0.97) | 0.91 (0.74) | 0.96x (1.02x) | 1.00x (1.3x) | 1.04x (1.65x) |
| light | 32.7 (22.4) | 0.97 (0.88) | 0.91 (0.78) | 1.00x (6.1x) | 1.00x (8.2x) | 1.01x (9.6x) |
| medium | 26.6 (18.4) | 0.95 (0.74) | 0.86 (0.76) | 0.93x (12.1x) | 1.00x (14.1x) | 1.05x (25.1x) |
| heavy | 22.6 (11.7) | 0.89 (0.33) | 0.76 (0.65) | 0.97x (26.4x) | 1.20x (42.8x) | 1.20x (78x) |

**What it shows:**

- **Phase 2's targets are met.**
  - Fidelity: `light` now beats what VTracer reaches on clean art.
  - Structure stays within 1.2x of the clean SVG at every severity.
  - Layer recovery passes 0.75 everywhere, but only just on `heavy` (0.76).
- **It beats VTracer at every severity** on clean fidelity, alpha IoU, layer recovery and smoothness. On shape count it
  is about a tie on `clean`: 0.96x, a part it missed, against VTracer's 1.02x.
- **`heavy`'s 22.6 dB is mostly the warp, not colour drift.** We first blamed drift, but phase 4 measured it (see
  there): with every shape given its true colour, `heavy` gains only 0.7 dB.
- **It is 10x slower than VTracer,** at 76 ms against 7 ms per image, but still interactive.

**Not done yet:**

- Symmetry (half and a mirror).
- A region with a hole: only its outer contour is fitted, so a hole costs misfit.
- The incremental re-scoring API (Decision 6). Every candidate rasterizes its whole bounding box.

These are phase 4's search moves, or later.

## Phase 4 results

**Where the error was.** On the `train` split, the structured pipeline scored 29.2 dB. Two oracles split that error
by cause:
- **Oracle colours:** each output shape takes the fill of the clean part it overlaps most.
- **Warp cap:** the clean render, warped exactly as `corrupt.v1` warped that image (its random draws replayed), scored
  against the clean render. This is what a pixel-perfect copy of the input's geometry would score.

| Severity | Traced | Oracle colours | Warp cap | Warp + blur cap |
|---|---|---|---|---|
| clean | 35.3 | 35.3 | 60 | 60 |
| light | 31.6 | 31.9 | 42.1 | 37.8 |
| medium | 26.8 | 27.2 | 30.1 | 29.0 |
| heavy | 22.9 | 23.6 | 24.4 | 25.1 |

So:
- Colour was never the problem: oracle colours add 0.3 dB in all.
- On `medium` and `heavy`, the warp caps anything that fits the input better. Only priors that undo the warp can pass
  that cap: simpler shapes, symmetry, snapping.
- On `clean` and `light`, the error is boundary precision, and fitting the input harder fixes it.

**The first rung: refinement under L** (`copycat_fit.rs`, `Target::refine`). Phase 2 refined only primitives. Now:
- The two shortest outline candidates, polygons and paths, also get coordinate descent, on every vertex or control
  point, before the shortest candidate is chosen.
- A polygon then drops the vertex that costs least to lose, whenever the descent from there is shorter.
- A move of an outline's point must save `outline_gain` x (σ - 0.002) x 1000 bits, with `outline_gain` = 12. That is
  nothing on a clean image, where σ sits at 8-bit quantization's floor. On a noisy one, an outline doesn't chase the
  noise.
- Every candidate's misfit starts from one precomputed "nothing covered" sum, instead of summing the image again.

Ablation on `train`, each row adding one step:

| Variant | Fidelity | clean | light | medium | heavy | Smoothness | ms / image |
|---|---|---|---|---|---|---|---|
| Phase 2 | 29.18 | 35.3 | 31.6 | 26.8 | 22.9 | 1.105x | 50 |
| + refine every outline candidate | 30.31 | 38.3 | 33.0 | 27.0 | 22.9 | 1.108x | 742 |
| + only the 2 shortest, refined before choosing | 30.87 | 40.1 | 33.2 | 27.2 | 23.0 | 1.172x | 99 |
| + noise-scaled gain (12) | 30.80 | 40.1 | 33.1 | 27.1 | 22.9 | 1.140x | 81 |
| + vertex deletion | 30.59 | 39.5 | 33.2 | 26.8 | 22.9 | **1.091x** | 137 |

- Refining, then choosing, beats choosing, then refining. A refined outline often wins against a primitive.
- Refined outlines on their own add turning, and the gain alone hardly reduces it.
- Deleting vertices costs 0.2 dB, but it is what minimizing L chooses, and it brings smoothness below phase 2's.
- The noise-scaled gain was tried at 2, 5 and 12; fixed minimum gains of 2, 6 and 12 bits hurt `clean` for nothing.
- Refining 1 outline scored 30.81 dB at 1.153x, and refining 3 scored 30.75 dB at 1.136x.

On the 24 held-out images, scored once:

| Severity | Clean fidelity (phase 2) | Alpha IoU | Layer recovery | Shapes | Palette | Smoothness |
|---|---|---|---|---|---|---|
| all | **30.8** (29.0) | 0.95 (0.95) | 0.87 (0.86) | 0.98x | 1.05x | 1.05x (1.1x) |
| clean | **40.7** (34.0) | 0.99 | 0.95 (0.91) | 1.00x | 1.00x | 1.04x |
| light | 33.3 (32.7) | 0.98 | 0.91 | 1.00x | 1.00x | 1.00x |
| medium | 26.5 (26.6) | 0.95 | 0.86 | 0.93x | 1.00x | 1.02x |
| heavy | 22.7 (22.6) | 0.89 | 0.75 (0.76) | 0.97x | 1.20x | 1.16x (1.20x) |

It takes 152 ms per image, against phase 2's 76.

### The second rung: smooth splines

**Which kind's geometry costs the most?** We put each clean part's exact geometry in place of the output shape that
overlaps it most, keeping our fill and order, and measured the fidelity gained per kind on `train`. The tool is noisy:
one output shape can stand in for two clean parts. One kind dominated: **smooth blobs**, worth +7.5 dB on `clean`, +2.0
on `light`, +1.1 on `medium` and +0.8 on `heavy`. On `clean`, 24 of 46 blobs came back as polygons, and most of the rest
as Schneider cubics cut at false corners.

**The candidate** (`copycat_fit.rs`, `spline`): a closed Catmull-Rom spline through k knots.
- Each knot's tangent is a sixth of the chord between its neighbours, so the curve is smooth everywhere.
- It is described by its knots alone: 2k + 1 coordinates and two type codes, where k cubics cost 6k. A smooth outline
  is now the shorter description.
- It is still written out as cubics, in the same SVG subset.
- Knots start evenly spaced along the contour (4, 5, 6 or 8 of them). The shortest spline is always refined, beside the
  two shortest outlines: its first knots are rough, so it rarely leads before refinement.
- Refinement nudges the knots, and deletes a knot whenever a polygon would lose a vertex.

On `train`:

| Variant | Fidelity | clean | light | medium | heavy | Layer recovery | Smoothness | ms / image |
|---|---|---|---|---|---|---|---|---|
| First rung | 30.59 | 39.5 | 33.2 | 26.8 | 22.9 | 0.861 | 1.091x | 137 |
| + splines | **31.92** | 42.8 | 35.0 | 27.0 | 22.9 | 0.874 | 1.061x | 189 |
| + 6 knot counts (4-8, 10) | 31.99 | 42.9 | 35.1 | 27.1 | 22.9 | 0.873 | 1.066x | 208 |

- Blobs fitted as splines: 43 of 46 on `clean` (before, 14 as paths of any kind) and 34 of 36 on `light`.
- Six knot counts add 0.07 dB for 10% more time, so we kept four.
- A cheaper wrong pixel on noisy images was tried twice: scaling `pixel_bits` down with σ above its floor, by
  0.5 to 2 per 0.001. It improves `heavy`'s smoothness (1.20x to 1.09x) and its layer recovery (0.78 to 0.80). But it
  costs `medium` 0.1-0.5 dB, because a simpler shape fitted to the warped input is still the wrong shape.

Held-out, scored once:

| Severity | Clean fidelity (first rung) | Layer recovery | Smoothness |
|---|---|---|---|
| all | **31.4** (30.8) | 0.88 (0.87) | 1.04x (1.05x) |
| clean | **42.4** (40.7) | 0.96 (0.95) | 1.03x |
| light | **34.1** (33.3) | 0.93 (0.91) | 1.00x |
| medium | 26.3 (26.5) | 0.86 | 0.99x |
| heavy | 22.7 (22.7) | 0.76 | 1.14x |

It takes 219 ms per image. `medium`'s -0.15 dB is on six images; on `train` it was +0.2.

### The third rung: regular polygons (a negative result, not kept)

`synthetic`'s polygons are regular, and after the second rung they recovered worst (IoU 0.73 on `medium`). So we tried
a regular-polygon candidate: n sides, a centre, a radius and a turn, for three type codes and 4 coordinates whatever n
is. It was drawn as its vertices and refined by coordinate descent on its four numbers (the turn stepped as arc length).
The shortest one went into refinement beside the other candidates. Sides ran from 3 to 8. It was guessed three ways:
- from the vertices of each simplified polygon: their centroid, their mean distance from it, and the phase of their
  angles taken n times round;
- from the contour: the n-th harmonic of its distance from the centroid against angle, with r = 2 x mean / (1 + cos(π/n));
- from the corners: a circle fitted through the simplified polygon's vertices, with the same phase.

On `train`, each guess added a little. A test (a hexagon half hidden under a shape above) completed itself to 6 sides,
its centre and radius within 0.3 px.

| Variant | Fidelity |
|---|---|
| Second rung | 31.92 |
| + regular, from polygon vertices | 32.02 |
| + the radial-harmonic guess | 32.08 |
| + the circumcircle guess | **32.14** |

Polygon IoU on `train` rose from 0.909 to 0.926 on `clean` and from 0.73 to 0.80 on `medium`. **Held-out, scored once,
it fell: 31.38 to 31.32 dB.** By severity, `clean` went 42.35 to 42.09, `light` 34.12 to 34.14, and `medium` (26.32) and
`heavy` (22.70 to 22.71) were level. Layer recovery went 0.8765 to 0.878, at 207 ms per image. The +0.22 on `train` did
not carry over, so the pipeline is unchanged and the code was not kept.

The worst `train` polygons say where the error actually is:
- seeds 20 and 60: polygons hidden to 52% and 39% visible (IoU 0.63 and 0.42). Too little outline shows to pin a regular
  shape, and a wrong n is still cheap;
- seed 68: a fully visible hexagon merged with a same-colour neighbour into one 12-vertex region. That is segmentation,
  not shape: no shape prior can fit two parts as one, and only a structure move (a split) can.

### The fourth rung: cutting a part in two

**Where the structure goes wrong.** On `train`, we counted the output shapes that are the best match for two or more
clean parts. There were 20 such merges: 3 on `clean`, 5 on `light`, 5 on `medium` and 7 on `heavy`. Twelve were two
touching parts of one colour, which segmentation can only see as one component. Images with a merge scored 4.7 dB
lower on `clean` and 1.3 on `light`. Symmetry was not tried: `synthetic`'s primitives are already symmetric, and its
blobs aren't.

**The move** (`copycat_trace.rs`, `shapes_of`). When two convex shapes overlap, their union's outline has a notch
wherever the two outlines cross.
- A notch is the deepest point of a run of the outline more than 1.5 px inside its convex hull (`copycat_fit::notches`).
- Each pair of the 4 deepest notches is tried as a cut: the part's pixels are split by the outline between them, closed
  by the chord.
- Each half is fitted with the other half's pixels free, as well as everything above. Both have one fill, so each
  completes under the other, the same way a hidden part does.
- The cut is kept if the two halves are shorter to describe than the whole, by 3 outline moves' worth of bits
  (`3 x Costs::outline_gain`, which is zero on a clean image). Each half is then cut again the same way.
- It runs only where a part has two notches, so a convex part costs nothing.

On `train`:

| Variant | Fidelity | clean | light | medium | heavy | Layer recovery | Smoothness |
|---|---|---|---|---|---|---|---|
| Second rung | 31.92 | 42.8 | 35.0 | 27.0 | 22.9 | 0.874 | 1.061x |
| + cuts, no margin | 32.10 | 43.1 | 35.2 | 27.1 | 23.0 | 0.885 | 1.101x |
| + cuts, margin 1 | 32.09 | 43.1 | 35.2 | 27.1 | 22.9 | 0.884 | 1.078x |
| + cuts, margin 3 | **32.08** | 43.1 | 35.2 | 27.1 | 22.9 | 0.883 | 1.068x |
| + cuts, margin 3, notches > 1.0 px | 32.08 | 43.1 | 35.2 | 27.1 | 22.9 | 0.883 | 1.073x |
| + cuts, margin 3, notches > 0.75 px | 32.03 | 42.9 | 35.2 | 27.1 | 22.9 | 0.883 | 1.085x |

- Without a margin, noise makes notches on `medium` and `heavy`, and the cuts there cost smoothness (`heavy` 1.17x to
  1.28x) for almost no fidelity. A margin of 3 keeps most of the layer recovery.
- Shallower notches let in more cuts that MDL accepts but that lower `clean` fidelity, and they take up to four times as
  long.
- Six of the twelve same-colour merges are now cut apart (seed 68's hexagon among them). Of the rest:
  - seed 56's second circle lies inside a same-colour circle, so no image could show it;
  - for seed 37, one end of the cut is hidden under the part above, so its notch is too shallow;
  - four are on `heavy`.

Held-out, scored once:

| Severity | Clean fidelity (second rung) | Layer recovery | Smoothness |
|---|---|---|---|
| all | **31.43** (31.38) | 0.883 (0.877) | 1.04x (1.04x) |
| clean | 42.35 (42.35) | 0.96 | 1.03x |
| light | 34.12 (34.12) | 0.93 | 1.00x |
| medium | 26.32 (26.32) | 0.86 | 0.99x |
| heavy | **22.93** (22.70) | 0.78 (0.76) | 1.16x (1.14x) |

It takes 223 ms per image. The held-out set has fewer same-colour merges on clean images, so the gain there is on
`heavy` alone. The gain is small, but it points the same way as on `train`, and it adds no cost to convex parts.

### The fifth rung: knots fitted by least squares

**Where the error was.** We re-ran the per-kind geometry oracle after the cuts. Blobs still cost the most: +5.7 dB on
`clean`, +1.6 on `light` and `medium`, +0.9 on `heavy`. Colour cost +0.5 on `light` and +0.7 on `heavy` (the colour
oracle), and nothing on `clean`. On `clean`, 43 of 46 blobs were already splines, at a mean IoU of 0.929. `synthetic`'s
blobs are themselves Catmull-Rom splines with this tangent rule, so the family held the answer. What fell short was the
search: knots started evenly spaced along the contour, and coordinate descent moved them only locally.

**The candidate** (`copycat_fit.rs`, `fit_knots`).
- A point of the spline at a fixed segment and `t` is a fixed mix of four knots (`spline_weights`). So once every edge
  point has been given its nearest point of the current spline, the best knots solve a linear system.
- The distance is measured mostly *across* the curve (tangent-distance minimization, the tangent weighted 0.05).
  Plain projection lets the knots creep along the curve: after 60 rounds it was still 0.15 px off a perfect spline.
  Tangent distance recovers one to under 0.1 px in 6 rounds (a test).
- Each point's nearest `t` is searched for among 12 samples per segment, then by ternary search between them.
- Points are fitted to the *true* edge, where there are at least 12: the completed loop's stretches along occluders
  and its hull are not the part's own. The loop only places the starting knots.
- 4 to 8 knots are fitted. The shortest as fitted is refined beside the evenly spaced spline, not instead of it.
- On a noisy image, it must beat every other candidate by 3 outline moves' worth of bits, because it follows noise
  too.

On `train`:

| Variant | Fidelity | clean | light | medium | heavy | Smoothness | ms / image |
|---|---|---|---|---|---|---|---|
| Fourth rung | 32.08 | 43.1 | 35.2 | 27.1 | 22.9 | 1.068x | 198 |
| + LS knots (projection, to the loop) *instead of* even ones | 31.91 | 42.3 | 35.1 | 27.4 | 22.8 | 1.064x | 206 |
| + LS knots *beside* even ones | 32.22 | 43.4 | 35.3 | 27.4 | 22.8 | 1.063x | 252 |
| + to the true edge | 32.24 | 43.4 | 35.4 | 27.4 | 22.9 | 1.067x | 266 |
| + tangent distance | 32.39 | 43.8 | 35.4 | 27.5 | 22.8 | 1.079x | 276 |
| + margin 3 on noisy images | 32.40 | 43.8 | 35.4 | 27.5 | 22.9 | 1.076x | 276 |
| + 12 samples per segment (was 24) | **32.43** | 43.9 | 35.4 | 27.5 | 22.9 | 1.075x | 268 |

- With a margin of 1, `heavy` was 0.02 dB below the fourth rung; with 10, `medium` lost most of its gain.
- 12 projection rounds instead of 6 added 0.006 dB.

Held-out, scored once:

| Severity | Clean fidelity (fourth rung) | Layer recovery | Smoothness |
|---|---|---|---|
| all | **31.99** (31.43) | 0.884 (0.883) | 1.06x (1.04x) |
| clean | **43.95** (42.35) | 0.96 | 1.03x |
| light | 34.23 (34.12) | 0.92 | 1.00x |
| medium | **26.88** (26.32) | 0.86 | 0.99x |
| heavy | 22.91 (22.93) | 0.78 | **1.25x** (1.16x) |

It takes 293 ms per image (it was 223). `clean` gained 1.6 dB, more than on `train`, and `medium` 0.55. The cost is on
`heavy`: fidelity is level, but its outlines are less smooth. The margin keeps a fitted spline from winning on noise
alone, but where one does win on a noisy image, it has followed some of that noise.

**Next:** past the warp cap only through priors that pick the *right* shape, not just a simpler one. Candidates:
- `heavy`'s smoothness: fit knots to a smoothed edge on a noisy image, or let the margin grow with σ;
- cut ends hidden under a part above: find notches on the part's outline with the pixels above it filled in;
- symmetry, for a set whose blobs are symmetric;
- the other structure moves from the ladder's step 3 (merge, re-order) on whole parts.

## The page

`/games/copycat` is the shared game page (docs/design/0007) with a `GameModule` (`app/games/copycat.ts`):

- **The leaderboard** ranks by clean fidelity. Its columns are alpha IoU, layer recovery, and the shape, colour and
  turning ratios.
- **Watch** shows one held-out image at a time: the input, the entrant's SVG, and the clean SVG, with that image's
  scores.
  - Each entrant's SVG per image is stored in its evaluation record, so what you see is exactly what was scored.
  - `GET /leaderboard` strips the SVGs. `GET /games/copycat/targets/{seed}` serves one image with every entrant's
    output, and the static site copies both routes.
  - For the structured pipeline, the page also re-runs it in the browser (WASM `CopyCatTrace`) and shows its stages:
    the k-means palette, the cleaned regions, and the layers pulled apart. A badge says whether the result is the same
    SVG that was scored.
- **Play** traces your own image the same way, at up to 192 px a side, and offers the result as a download. Nothing is
  uploaded.

Two items from "The game, as a player sees it" are not built yet:

- comparing against VTracer on your own image, since VTracer has no WASM build here;
- rebuilding a target by hand.

## Open questions

- **Licences:** which icon sets (Twemoji needs attribution), and which AI-generated images can go in `wild` and be
  shown on the site.
- **Commercial comparisons:** whether Vectorizer.AI's and Recraft's terms allow publishing a comparison, and at what
  cost.
- **Strokes in v2:** many icons are drawn with strokes, and a stroke is the natural thing to animate as a line drawing.
- **Gradient fills in v2:** some logos genuinely have them, and v1 would have to fake them with bands.
- **Text:** recovering a font is out of reach. Text will come out as filled shapes. Detecting that a region is text, so
  it can be flagged, may be worth it.
- **The code lengths themselves:** the precision per coordinate and the primitive costs are a prior. They will need
  tuning on the training split, and the tuning must never see the held-out split.

## Explicitly out of scope

- Photographs and painterly illustrations.
- Large learned models that write SVG code.
- LPIPS-style learned metrics as a reward.
- Animation, beyond keeping the output easy to animate.

## Sources

- Optimize & Reduce: <https://arxiv.org/abs/2312.11334>
- Bézier Splatting: <https://arxiv.org/abs/2503.16424>
- Layered vectorization via semantic simplification: <https://arxiv.org/abs/2406.05404>
- LayerPeeler: <https://arxiv.org/abs/2505.23740>
- AmodalSVG: <https://arxiv.org/abs/2604.10940>
- RLRF: <https://arxiv.org/abs/2505.20793>
- VFig: <https://arxiv.org/abs/2603.24575>
- VectorGym: <https://arxiv.org/abs/2603.29852>
- Recraft, vectorizing: <https://www.recraft.ai/blog/how-to-vectorize-file>
- AniClipart: <https://arxiv.org/abs/2404.12347>

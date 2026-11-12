# 0022 — Copy Cat: recover the clean, layered SVG behind a generated logo or icon

Status: **Phases 1-2 built, phase 3's page, and phase 4's first rung** (2026-10-24).
- Phase 1 built the rasterizer, the subset, `synthetic` + `corrupt.v1`, protocol `copycat.clean.v1` and the
  baselines; see "Phase 1 results".
- Phase 2 built the structured MDL pipeline; see "Phase 2 results".
- Phase 3 so far is the page, `/games/copycat`, and the `icons` set (152 Twemoji, measured on `train` only); see
  "The page" and "The `icons` set". The `wild` side-by-sides are still to come: they need real AI-generated images
  and blind judges, neither of which we have yet.
- Phase 4 so far:
  - refines each shape point by point under L, and on a noisy image keeps a point only if it earns its keep;
  - describes a smooth outline by its knots, fitted by least squares;
  - cuts a part of one colour in two where two shapes overlap;
  - never joins a piece another part lies beneath (a detail on it), measured on `icons`;
  - takes the background from the corners when the border's fit misses them (a tile), and keeps a near-background
    colour enclosed by parts (a white letter) as a part;
  - orders two touching parts by the fraction of each the other covers, not the pixel count;
  - fits the background from the three corners that agree when noise or clutter spoils the fourth, and keeps an
    enclosed gap as a part only if its interior, clear of blurred edges, differs from the background;
  - gives back a light part the background took for faint clutter, if it stands out and touches the subject, and never
    dissolves the background where it reaches the image's edge (a tile's rounded corner);
  - cuts holes: each gap a part's shape encloses is fitted as a shape of its own and drawn as a reversed subpath;
  - fits rounded rectangles, as a path described by its box and radius;
  - redraws a part's final shape as a tight spline when that is shorter and no further off;
  - polishes every shape against the whole image as drawn, with the structure fixed (see "Polish").

  A regular-polygon candidate was measured and not kept. See "Phase 4 results".

Phases 3-6 are proposed. [0024](0024-copycat-learned-inverse.md) proposes a learned entrant beside the MDL tracer: a
small transformer, trained from scratch, that reads the image and writes the SVG, then fine-tuned by RL against the
rendered result (it narrows "Explicitly out of scope" below to size and provenance).
[0025](0025-copycat-prior-art-and-direction.md) surveys the prior art and sets the learned work's next steps: a denoiser in
front of the MDL tracer, then a model that sees its own canvas.
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
| Layered and occlusion-aware vectorization | Semantic simplification (CVPR 2025), LayerPeeler (2025, flat-colour SVG data), AmodalSVG (2026: peel layers, complete hidden parts) | the closest research to this use case; AmodalSVG's code isn't released |
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
  refinement phase's. Exact-coverage renderers (Windfoil) now look like a better target for flat art, and this rasterizer is
  already exact coverage; to be measured (0023). Bézier Splatting's 30x and 150x speed-ups are for open curves, and flat
  fills use closed ones (6x and 18x).

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
   annealing or a GA (`evolve`). Optimize & Reduce's pruning becomes one move among them. Measured before building:
   even judged by the answer, these moves win under 0.2 dB (see "Search over the structure").
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
2. **Background:** fitted to a 2-pixel border, either constant or a gradient, whichever is shorter; or, when that fit
   misses the corners, started again from the three corners that agree best (see "A background that reaches the
   corners" and "What limits `medium` and `heavy` icons"). It is a cluster of its own whose centre varies per pixel.
3. **Palette:** k-means++ in OKLab from 16 seeds. Clusters merge while the residual the merge adds, in bits, is less
   than the 24-bit palette entry plus the shared boundary it saves.
4. **Labels:** a majority filter, then dissolving components that are too small or too thin (anti-aliased bands),
   except the background where it reaches the image's edge. A background component enclosed by parts becomes a colour
   of its own if its interior, 2 pixels clear of its edge, differs from the background by more than one robust
   deviation. A light part that step 2 merged into the background only for being near its colour is a part again if
   it stands out by three deviations and a quarter of its boundary is with parts (see "The `icons` clean render").
5. **Parts:** one per component, filled with its label's interior mean. Two components of one label join when one shape
   says them more briefly, unless another part lies beneath either of them (see "Join only what an occluder cut").
6. **Order:** each part is fitted with every other part free. Of two touching parts, the one whose shape covers more
   of the other, as a fraction of it, goes below (see "Paint order by the fraction covered").
7. **Shapes:** each part is fitted with the parts above it free. The candidates are circle, ellipse, rect, polygon and
   Bézier path (Schneider's fit, cut at corners), rounded rectangle (a path, said as its box and radius), and the
   shortest `model bits + pixel_bits x misfit` wins. Then
   the final shape is redrawn as a tight spline if that is shorter and no further off (see "Tight splines"), and
   each gap the shape encloses is cut out as a hole while that is shorter (see "Holes").

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
- A region with a hole: only its outer contour is fitted, so a hole costs misfit. (Since fixed: see "Holes".)
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

### The sixth rung: a point must earn its keep

**Where `heavy`'s turning comes from.** We split the output's total turning, beyond the clean SVG's, on `train`'s
`heavy` images (clean: 775 rad):
- 87 rad of non-convex wiggle in polygons;
- 57 from extra shapes (2π each);
- 19 of wiggle in paths.

The wiggliest polygons were mostly rects (11 of 19 `heavy` rects came back as polygons), circles and polygons. They
were traced as 10-16 vertex outlines with IoU 0.85-0.94, following the warp's bend.

**An inconsistency.** On a noisy image, a vertex or knot *move* had to save `outline_gain` bits. But a vertex or knot
*deletion* was refused whenever it lengthened the description at all. So a point that paid for itself by less than
one move's worth stayed: it was following the noise. Now a point is deleted unless keeping it saves `outline_gain`
bits, the same bar as a move (zero on a clean image, so `clean` is unchanged).

Two candidates were tried first and not kept: a convex-hull polygon candidate (smoothness 1.192x to 1.190x, fidelity
-0.02) and a higher bar.

| Bar to keep a point (x `outline_gain`) | Fidelity | medium | heavy | Layer recovery | Smoothness | heavy smoothness |
|---|---|---|---|---|---|---|
| 0 (fifth rung) | 32.43 | 27.48 | 22.91 | 0.884 | 1.075x | 1.192x |
| **1** | **32.45** | 27.51 | 22.93 | 0.886 | 1.064x | **1.159x** |
| 3 | 32.43 | 27.53 | 22.84 | 0.888 | 1.063x | 1.160x |
| 10 | 32.38 | 27.48 | 22.84 | 0.881 | 1.074x | 1.224x |

At a bar of 1, `heavy`'s polygon wiggle halves (87 to 42 rad) and `medium`'s falls from 36 to 22. Path wiggle rises a
little (19 to 27).

Held-out, scored once:

| Severity | Clean fidelity (fifth rung) | Layer recovery | Smoothness |
|---|---|---|---|
| all | 32.00 (31.99) | 0.883 (0.884) | **1.05x** (1.06x) |
| clean | 43.95 (43.95) | 0.96 | 1.03x |
| light | 34.23 (34.23) | 0.92 | 1.00x |
| medium | 26.86 (26.88) | 0.86 | 0.99x |
| heavy | 22.96 (22.91) | 0.78 | **1.18x** (1.25x) |

This wins back most of the fifth rung's cost to `heavy`'s smoothness, with fidelity level.

### What is left, and three measured non-changes

After the sixth rung we re-ran every attribution on `train`.

| Where | What exact clean answers would gain | Why it is hard to get |
|---|---|---|
| Blobs on `clean` | +5.0 dB (geometry oracle) | Fully visible blobs are already at IoU 0.987. The loss is blobs 35-67% hidden (IoU 0.42-0.86), whose hidden outline is not in the image. |
| Colour on `light` / `heavy` | +0.55 / +0.64 dB (colour oracle) | Mostly `corrupt.v1`'s per-part colour drift: the fill we see is not the fill that was drawn. |
| Rects on `medium` / `heavy` | +1.7 / +1.0 dB (geometry oracle) | The warp cap: a rect fitted to warped edges is not the clean rect either. |

Three changes were tried on `train` and are not kept:
- **Deleting the extra shapes.** On noisy images, output shapes that are no clean part's best match make up 63 rad
  of `heavy`'s excess turning. But deleting them costs `medium` 0.50 dB (and gains `heavy` 0.004): they carry real
  pixels, mostly shading or drift over a part, or a same-colour piece of it.
- **Merging colours more readily** (`merge_delta`), to average out the drift of parts that share a palette colour.
  Fidelity is flat from 0.08 to 0.10 (32.45, 32.45, 32.47), and falls beyond: 32.26 at 0.11, 32.03 at 0.12, and 31.24
  at 0.13, where `clean` loses 2.8 dB to real colours merged. 0.10 improves `heavy`'s smoothness (1.16x to 1.09x), but
  a +0.03 dB gain beside a cliff is a threshold tuned to this set, not a prior.
- **Making outlines beat primitives by a margin on noisy images** (`outline_gain` x 3, 10 or 30), to pick the rect
  over a polygon that follows the warp. Fidelity was 32.45, 32.44 and 32.21, against 32.45.

So the next gains on `synthetic` would need what the image doesn't hold: a prior on how hidden blobs continue, or on
drift. More search over shapes it does show won't supply either.

**Next:** past the warp cap only through priors that pick the *right* shape, not just a simpler one. Candidates:
- cut ends hidden under a part above: find notches on the part's outline with the pixels above it filled in (small:
  one `train` image);
- symmetry, for a set whose blobs are symmetric;
- the other structure moves from the ladder's step 3 (merge, re-order) on whole parts;
- measuring on the `icons` and `wild` sets, where hidden parts, drift and the warp are not `synthetic`'s, before
  tuning further to `synthetic`.

## The `icons` set

`icons` is 152 Twemoji (v17.0.3, graphics copyright 2019 Twitter, Inc and other contributors, CC-BY 4.0), downloaded on first use.
They were drawn at random, by a fixed seed, from the emoji whose SVG fits the subset after conversion:
- `arena.twemoji` scales the 36-unit viewBox to 128 px;
- it turns relative commands, H/V/S/T and arcs into M/L/C/Q/Z;
- it pushes a group's fill down onto its shapes;
- it refuses anything with a transform, stroke, gradient or opacity.

`copycat_sets.icon_split` shuffles the icons by a fixed seed. 24 are held out (6 per severity), the other 128 are
`train`, and both are built on demand (`arena.copycat_artwork`).
Only `train` has been scored; held-out waits until a change made for icons is kept.

### Results on `train`

| | Structured | VTracer (background deleted) |
|---|---|---|
| Fidelity | 18.10 dB (clean 19.99, light 21.09, medium 16.42, heavy 14.88) | 11.93 dB |
| Alpha IoU | 0.72 | |
| Palette ratio | 0.76 | |
| Time per image | 5.7 s | |

On `synthetic`, by comparison, the pipeline scores 32.45 dB. What limits it here comes down to three causes, in order
of cost:

1. **The background is taken to be the border colour.** 34 icons cover more than 25% of the border (tiles, flags,
   squares); they score 9.5 dB with alpha IoU 0.39.
   - The tile's colour is fitted as the background and deleted.
   - What is left is its rounded corners, which then show the real background.
2. **Fills near the background are lost.** 29 icons have more than 10% of their area within OKLab 0.1 of the
   background (mostly white parts); they score 15.6 dB with alpha IoU 0.61. The other 65 icons score 23.7 dB with alpha
   IoU 0.94. Bringing causes 1 and 2 up to that level would be worth about +5.6 dB on the mean.
3. **`join` buries details.** Twemoji is full of disjoint same-colour details: eyes, letters, keys.
   - `join` fuses them into one shape, judged with every other part free.
   - That shape is then ordered below the part that contains them, so they vanish.
   - With `join` off, single icons gain a lot: keyboard 15.5 to 31.1 dB, ATM 14.6 to 19.7, zipper-mouth face 18.4 to
     24.6, crying face 16.4 to 18.8.
   - Over all of `train`, icons go from 18.10 to 18.89 dB (clean +1.35, light +1.2), and the pipeline is 2.6x faster
     (2.2 s per image; `join`'s pairwise fits are O(n²) in parts). `synthetic` goes from 32.45 to 32.42.

Ruled out:
- `min_area` 10: +0.04 dB.
- `colors` 32: no change on `clean`.
- `merge_delta`, `edge_bits` and `background_delta`: none restores the lost details, except `merge_delta` 0.02 on
  the keyboard alone. `background_delta` 0.02 makes the near-background icons worse (crying face 16.4 to 11.6),
  because it then keeps the background as a shape too.

### Join only what an occluder cut

`join` assumed every other part could be above the two pieces. A detail on a part breaks that: two eyes on a face,
joined across the face, make one shape that `order` then puts *under* the face, and both eyes vanish. Now a piece is
never joined if another part, completed under its own occluders, covers at least half of it. That part lies beneath the
piece, so the piece is a detail on it, not a glimpse of something it hides. Every other part stays free, so a bar an
occluder cuts in two is still one shape. The completed shapes are fitted once and shared with `order`; after a join only
the joined part's is refitted, since every other part's free pixels are unchanged. Tests:
`a_part_an_occluder_cuts_in_two_is_joined` and `details_of_one_colour_are_not_joined_under_what_they_sit_on` (eyes on
a face; marks on a band across a frame of their colour).

Measured on `train` (fidelity, dB; time per icon):

| `join` | `synthetic` | `icons` | per icon |
|---|---|---|---|
| as before (every other part free) | 32.45 | 18.10 | 5.7 s |
| off | 32.42 | 18.89 | 2.2 s |
| guard: no part in the gap may lie beneath both pieces | 32.45 | 18.63 | 4.9 s |
| only parts the paint order puts above are free | 32.45 | 18.65 | 5.4 s |
| the same, and no part above may lie beneath either piece | 32.45 | 18.92 | 6.8 s (3 runs at once) |
| a part beneath either piece is not free | 32.45 | 18.81 | 4.3 s |
| **kept:** a piece with a part beneath it is never joined | **32.45** | **18.84** | **3.2 s** |

- The paint-order versions fail the occluder test: with nothing deciding which of two parts goes on top, the size
  tie-break puts the larger occluder below, and it is then not free.
- The guard still joins thin lines across a part beneath them (the fist's finger lines go under the yellow).
- Making a covering part non-free, instead of refusing the join, still joins the ATM's letters to its frame.
- Off is 0.05 dB better on `icons` and faster, but gives up `synthetic`'s occluded parts (three `train` images,
  +0.03 dB) and the layer they recover.

Held-out, scored once: `synthetic` is unchanged (31.999 dB, the same SVGs); `icons` goes from 18.52 to 18.77 dB and
from 4.8 to 3.1 s per icon.

### A background that reaches the corners

The pixels can't tell a tile filling most of the border from a background: either describes the image equally well,
and only which one is transparent differs. So a prior decides, and the prior is that the background reaches the
corners.
- The border's robust fit is kept if it explains each corner's 3x3 mean. The yardstick is how much the corners vary
  among themselves: tight when they agree, loose under a gradient or clutter.
- Otherwise the fit starts again from the corners. Its outlier cut is set by the pixels it keeps, not by the whole
  border, where the tile would set it. It is used if it explains all four corners.

With the tile now a part, its letter needs saying. The letter's white is within `background_delta` of an off-white
background, so it is labelled background, and a filled tile can't show a hole. So a background component enclosed by
parts becomes a colour of its own, unless it is the background's colour (the gap between four bars, a gear's hole). The
test: its mean is further from the background's than the background fit's own outlier cut. On `synthetic`, a 6σ test
filled real gaps (image 30, -4.1 dB); the fit's cut, which grows with JPEG, drift and gradients, changes no `synthetic`
image.

Measured on `train` (fidelity, dB):

| | `synthetic` | `icons` | `icons` alpha IoU |
|---|---|---|---|
| before | 32.446 | 18.84 | 0.72 |
| corners only | 32.444 | 19.25 | 0.76 |
| enclosed regions only, no colour test | 32.401 | 19.26 | |
| corners + enclosed, 6σ test | 32.401 | 21.53 | |
| **kept: corners + enclosed, the fit's cut** | **32.444** | **20.86** | **0.82** |

- Per severity on `icons`: clean 21.35 to 25.83, light 22.09 to 23.84, medium 16.92 to 18.50, heavy 15.00 to 15.28.
- The prior's failure mode: a subject touching a corner. The clamp (train seed 77) loses 3.2 dB.
- Held-out, scored once: `synthetic` 31.999 to 32.003 dB; `icons` 18.77 to 19.41 dB, clean 15.94 to 18.36.

### Paint order by the fraction covered

Some enclosed white parts still came out under their tile (the refresh arrows, train seed 62). Fitted with the tile's
pixels free, each arrow grows into a blob that covers more of the tile's pixels than the tile, completed, covers of the
arrow's. `order` compared those counts, so the arrow went below. Now it compares fractions: the tile covers all of the
arrow, and the arrow a sliver of the tile. A count favours whichever part is larger; a fraction asks which part the
other one's shape is really under. Test: `the_part_that_covers_more_of_the_other_goes_below`.

Measured on `train` (fidelity, dB):

| | `synthetic` | `icons` |
|---|---|---|
| by count | 32.444 (layer recovery 0.8852) | 20.86 |
| **by fraction** | **32.457** (0.8847) | **22.21** |

- `icons`: 60 change, 49 up, 11 down by at most 0.7 dB. The tiles with white symbols gain the most: refresh arrows 11.6
  to 40.5, seed 143 15.1 to 38.8, seed 93 12.2 to 32.5. Per severity: clean 25.83 to 29.90, light 23.84 to 24.83,
  medium 18.50 to 18.82, heavy 15.28 to 15.31.
- `synthetic`: 31 change, 14 up and 17 down, a wash. Where two blobs overlap, both orders are mostly right, and the
  difference is how a hidden part completes.
- Held-out, scored once: `synthetic` 32.003 to 32.230 dB (clean 43.95 to 44.80, layer recovery 0.883 to 0.887);
  `icons` 19.41 to 20.88.

**Next, on `icons`:** `heavy` (15.3 dB) and `medium` (18.8 dB) are now furthest behind `synthetic`'s 22.9 and 27.5.

### What limits `medium` and `heavy` icons

To find out, each `medium` and `heavy` train image was corrupted again with every random draw kept and one `corrupt.v1`
step changed, then traced and scored. There were 64 icons and 48 synthetic images. Leaving no step out reproduces the
recorded scores to within 0.005 dB. Two experiments:

- **Every step but one.** Leaving out any single step gains at most about 1.2 dB, because the others still do the
  damage.
- **One step only.** This is the clean image plus that one step, and it shows each step's cost on its own.

Fidelity in dB. "None" is the clean render: no step at all.

| one step only | icons `medium` | icons `heavy` | synthetic `medium` | synthetic `heavy` |
|---|---|---|---|---|
| none | 29.1 | 28.0 | 42.4 | 41.8 |
| drift | -2.4 | -5.0 | -4.8 | -10.8 |
| shading | -0.5 | -3.0 | -2.4 | -4.3 |
| warp | -6.4 | -8.6 | -12.9 | -17.8 |
| blur | -2.4 | -5.0 | -6.6 | -9.7 |
| **background** | **-3.0** | **-6.4** | -0.5 | -1.0 |
| noise | -6.9 | -6.9 | -5.1 | -9.0 |
| jpeg | -5.4 | -5.6 | -3.6 | -5.0 |

Two findings:

- **Most of the gap is the clean render.** On icons with nothing corrupted, the pipeline already scores 13 dB below
  synthetic. Twemoji drawings have more parts, thin details and overlaps. That is the rungs' work (shapes, joins,
  order), not robustness. Apart from the background step, the corruption costs icons no more than synthetic.
- **The background step is the icon-specific loss.** It costs synthetic almost nothing and icons up to 6.4 dB: gradient
  backgrounds -6.4, clutter -5.1, flat 0.0. Under noise, a tile whose border fit misses the corners fell back to the
  border as the background, and the whole tile came out at about 6 dB (train seeds 2, 3, 79 and 133).

Two fixes, in `Background::fit` and `enclose`:

1. **A corner fit that noise and clutter don't break.**
   - The corner fit starts from the three corner patches that agree best. One corner can be a speck of clutter, or a
     tile's blurred edge.
   - It measures the kept pixels' deviations from the gradient, not from a flat colour, so a gradient background
     doesn't inflate its cut.
   - It is kept if it explains three of the four corners.

   Test: `a_speck_on_one_corner_does_not_hide_the_background`.
2. **Enclosed gaps judged by their interior.** On a noisy image the corner fit's cut is large, so the old test (6σ
   from the background) dropped a white symbol on its tile (train seed 141, 23.1 to 12.3 dB with the first fix alone).
   The test now uses the gap's interior: pixels at least 2 clear of its edge, of which there must be at least 16. Its
   mean must differ from the background's by more than one robust deviation (the cut / 6).
   - On train, symbols differ by 0.03 to 0.06 and holes by under 0.002 (the gear's, seed 144).
   - Synthetic seed 30's real gap has a single interior pixel, so it stays a hole.

Measured on `train` (fidelity, dB):

| | `synthetic` | `icons` | `icons` `clean` | `light` | `medium` | `heavy` |
|---|---|---|---|---|---|---|
| before | 32.457 | 22.21 | 29.90 | 24.83 | 18.82 | 15.31 |
| corner fit only | 32.457 | 23.00 | 30.64 | 25.87 | 18.47 | 17.03 |
| **both** | **32.457** | **23.96** | **30.61** | **27.03** | **20.68** | **17.50** |

- `synthetic`: no image changes.
- `icons`: 26 up and 8 down. The tiles gain the most: seed 71 10.2 to 37.2, seed 89 8.1 to 31.8, seed 2 5.6 to 22.6.
- The one large loss: the locomotive, seed 87, 21.0 to 15.8. Its subject reaches both bottom corners, so the fit
  keeps a background that is half the drawing. The corners are the only prior about the background, and when the
  subject owns two of them the prior is wrong.
- Held-out, scored once: `synthetic` 32.230 to 32.230; `icons` 20.88 to 21.37 (clean 23.10, light 26.50, medium
  19.37, heavy 16.53).

**Next, on `icons`:** the clean render, 29 dB against synthetic's 42. With the background fixed, the rest of the gap is
in the drawings themselves.

### The `icons` clean render

"Clean render" here is every train image traced with no `corrupt.v1` step at all, on a flat background. Icons score
29.5 dB on it, synthetic 43.0. The scores are bimodal: most icons are at 30 to 40 dB, and 28 of the 128 are under 17.
The 28 dominate the mean (median 32.5).

The error was split per pixel, over white, by what the clean render has there. The share of the squared error, and
the mean gain in dB if that error were gone:

| where | share of error | dB if fixed |
|---|---|---|
| edges of other parts (within a pixel or two of a boundary) | 18% | 3.1 |
| interiors of other parts | 27% | 0.25 |
| light parts (no channel under 75%), interior and edge | 23% | 1.3 |
| drawn outside the subject, where the clean render is empty | 8% | 1.4 |
| holes in the subject, drawn filled | 18% | 0.44 |

- **Edges** are spread over the good images. They are the shapes' precision, and in dB they move the 35 dB images,
  not the mean.
- **Holes** are concentrated in a few icons (the sun's ring 24 dB, a racket 9). Each part is fitted by its outer
  contour only. The subset can say a hole (a reversed inner subpath), but the fitter doesn't yet.
- **Light parts** are most of the worst images: a plate, a white chicken, a panda's face, an ice skate, a grey speaker
  cone. The backgrounds are near-white (0.86 to 1.0 per channel), and Twemoji's light greys (`#e1e8ed`, `#ccd6dd`) are
  within `background_delta` (0.1) of them. So `segment` merges them into the background, with the faint clutter that
  rule is for.
- **Outside**: a tile's rounded corners came back square. The background in each corner is a thin sliver, and
  `clean_labels` dissolved it into the tile.

`background_delta` itself can't be the fix. Lowered, it gives clutter back too:

| `background_delta` | `synthetic` | `icons` | `icons` `clean` |
|---|---|---|---|
| **0.1** (as before) | **32.457** | **23.96** | **30.61** |
| 0.06 | 31.66 | 24.35 | 31.53 |
| 0.03 | 30.46 | 23.53 | 31.43 |
| 0 | 28.88 | 23.28 | 32.38 |

Two changes:

1. **The background never dissolves where it reaches the image's edge.** Test: `a_tile_keeps_its_rounded_corners`.
2. **A light part is given back.**
   - `segment` records the pixels a cluster had when it joined the background only for being within
     `background_delta`, a merge the description length alone wouldn't make (`Segmentation::faint`).
   - After `clean_labels` and `enclose`, each region of such pixels still labelled background is a part again, with a
     colour of its own, if three things hold:
     - it is big and thick enough to be a part;
     - its interior is three robust deviations from the background;
     - at least a quarter of its boundary is with parts.
   - On a clean image any light part stands out, and noisy clutter barely does. A part of the subject touches the rest
     of it, while clutter mostly lies alone in the background.

   Test: `a_light_part_beside_the_subject_is_not_the_background`.

Neither test alone was enough, and the order matters:

- **Without the colour test** (one deviation, as `enclose` uses), clutter beside the subject comes back as ghost
  blobs. Synthetic falls 0.11 at a share of 0.3, and two noisy images lose 3.7 and 6.7 dB.
- **Without the share test**, the same happens to clutter anywhere: synthetic 32.27.
- **Run before `enclose`**, it claims only the faint pixels of a white letter's region and thins the letter: train seed
  62 lost 8.8 dB.

Measured on `train` (fidelity, dB):

| | `synthetic` | `icons` | `clean` | `light` | `medium` | `heavy` | no corruption |
|---|---|---|---|---|---|---|---|
| before | 32.457 | 23.96 | 30.61 | 27.03 | 20.68 | 17.50 | 29.49 |
| edge background kept | 32.454 | 24.03 | 30.68 | 27.03 | 20.92 | 17.49 | |
| **and light parts given back** | **32.454** | **24.82** | **32.61** | **27.85** | **21.28** | **17.54** | **31.51** |

How sensitive the two thresholds are:

- **Share** (at three deviations): 0, 0.15, 0.25, 0.3, 0.4 and 0.5 give icons 24.98, 24.88, 24.82, 24.72, 24.36 and
  24.08.
- Synthetic is untouched from 0.15 up and falls to 32.27 at 0.
- **Deviations** (at a share of 0.3): 1, 2, 3 and 4 give synthetic 32.35, 32.35, 32.454 and 32.454.

- `synthetic`: one image changes, seed 30, by -0.25 dB (the edge rule; no corruption: unchanged at 42.95).
- `icons`: 35 change, 24 up. The light parts come back: the hatching chick's shell 19.4 dB up, the satellite 15.4,
  the custard's plate 14.5, the axe's blade 13.9, the panda 12.8. The largest loss is the locomotive (seed 87, -2.9).
  Its subject owns two corners and is already mistaken for background.
- Held-out, scored once: `synthetic` 32.230 to 32.230; `icons` 21.37 to 22.63 (clean 23.10 to 28.09; light, medium
  and heavy within 0.04).

**Next, on `icons`:** holes (the fitter only fits a part's outer contour; the subset can already say a hole), then the
shapes' precision at edges.

### Holes

Each part was fitted by its outer contour, so a ring came back as a disc. The subset can already say a hole: a path
whose inner subpath winds the other way, which the non-zero rule leaves empty. `Target::with_holes` now cuts them.

1. **Find the gaps.** After a part's final shape is fitted, a gap is a 4-connected run of pixels that the shape
   covers but that are neither the part's nor free under a part above. It counts only if:
   - the shape encloses it (it touches no uncovered pixel and not the image's edge);
   - it has 6 pixels or more;
   - it has an inside: a pixel whose four neighbours are all in the gap.
2. **Fit each gap**, largest first, as a region of its own. What it must cover is the fraction of each pixel that
   isn't the part's, within 2 px of the gap and inside the shape.
3. **Keep it while the whole is shorter.** The shape becomes a path, with the hole reversed if it winds the same way.
   It is described as the outer shape's description plus the hole's, so a circle with a hole isn't charged as four
   Béziers. Test: `a_ring_keeps_its_hole`.

Two things the first version got wrong, both visible only in the score:

- **A hole fitted with the pixels under parts above it free** reached under them. Those parts cover their pixels
  only as well as their own shapes fit, so the background showed through at their edges. A hole now has to keep off
  those pixels.
- **A gap with no inside is a blurred band**, along the edge of a white symbol on its tile. Without the inside test
  it became a 0.6 px hole the length of the edge: the tile with two white triangles lost 0.8 dB.

Measured on `train` (fidelity, dB):

| | `synthetic` | `icons` | `clean` | `light` | `medium` | `heavy` | no corruption |
|---|---|---|---|---|---|---|---|
| before | 32.454 | 24.82 | 32.61 | 27.85 | 21.28 | 17.54 | 31.51 |
| holes, first version | 32.454 | 24.91 | 32.91 | 27.92 | 21.26 | 17.54 | 31.74 |
| kept off parts above | 32.454 | 24.91 | 32.93 | 27.91 | 21.27 | 17.54 | |
| **and an inside** | **32.454** | **24.94** | **32.99** | **27.96** | **21.27** | **17.54** | **31.85** |

- `synthetic`: no image changes. Its parts are filled primitives with no holes, so there are no gaps to cut.
- `icons`: 9 images change, 6 up, and none loses more than 0.1 dB:
  - the compact disc, 7.4 dB up;
  - the badminton racket's strings, 3.5;
  - the trophy's handles, 3.3;
  - with no corruption, the sun's ring goes from 15.2 to 41.9.
- Held-out, scored once: `synthetic` 32.230 to 32.230; `icons` 22.63 to 23.32 (clean 28.09 to 30.83; light,
  medium and heavy unchanged).

**The light-severity sun still comes out as a crescent** (train seed 63, 15.2 dB). There, `join` joins the ring with
two of its rays first. `join` compares shapes fitted without holes, and to it a ring is a poor disc, so absorbing two
dots looks cheap. Holes are cut only once a part's shape is final. Giving `join` holes would mean fitting them on
every comparison it makes.

**Next, on `icons`:** the shapes' precision at edges (the largest error left on the clean render), and holes in
`join`'s comparisons.

### Rounded rectangles

The error maps of the clean render put most of what is left at shapes' edges, in two patterns:

- **Rounded rectangles** (keys, tiles, buttons) are everywhere in Twemoji. The subset's `<rect>` has no `rx`, so the
  fitter had a rect, a circle or a Bézier path to choose from. The keyboard's keys came back as circles and plain
  rects, and its body with polygon corners.
- **Small tips** of crescent-shaped paths (a mouth, an eyebrow) are lost (see "Tight splines").

This section is about the first. `rounded_rect([x, y, width, height, r])` is a path: four lines and four quarter circles
as cubics, with `r` clamped to half the shorter side, where it becomes a stadium. It is a candidate from every box
candidate, at radii of 0.15, 0.3 and 0.5 of the shorter side. Like any primitive it is refined, all five numbers, so the
radius finds its own size. It is described as a shape type plus five coordinates, not as its ten segments. Test:
`a_rounded_key_is_described_as_a_rounded_rectangle`.

**It has to fit, not just be short.** Five numbers cover most blobs roughly, so the first version could beat a much
closer outline on length alone. So a rounded rectangle now replaces the best other fit only if it is shorter *and* no
more than 1 pixel of misfit further off (`rounded_wins`). A real rounded rectangle passes, because it fits at least
as well. Test: `a_rounded_rectangle_must_fit_not_just_be_short`.

Measured on `train` (fidelity, dB):

| | `synthetic` | `icons` | `icons` images up / down | worst `icons` loss |
|---|---|---|---|---|
| before | 32.454 | 24.94 | | |
| shorter is enough | 32.430 | 25.16 | 25 / 4 | 1.0 |
| plus a "rounded" flag (+3 bits, as a spline pays) | 32.429 | 25.13 | 25 / 4 | 1.0 |
| within 0 px of misfit | 32.450 | 25.16 | 23 / 2 | 0.18 |
| **within 1 px** | **32.459** | **25.16** | **23 / 2** | **0.28** |
| within 3 px | 32.448 | 25.15 | 23 / 4 | 1.0 |
| within 1 px, plus the noise margin a fitted spline pays | 32.454 | 25.15 | 21 / 2 | 0.28 |

- `icons`: the keyboard is up 5.4 dB, and four more are up 3 dB or more. Five images supplied 64% of the first version's
  gain.
- `synthetic`: of the first version's five losses, one is left: seed 53, 1.7 dB, where a small ellipse on a noisy image
  becomes a rounded rectangle whose misfit against that image is lower. Neither the flag nor the noise margin stops it.
  It is still open.

Held-out, each version scored once:

| | `synthetic` | `icons` | `clean` | `light` | `medium` | `heavy` |
|---|---|---|---|---|---|---|
| before | 32.230 | 23.32 | 30.83 | | | |
| shorter is enough | 32.229 | 23.24 | 30.58 | 26.47 | 19.37 | 16.55 |
| **within 1 px** | **32.229** | **23.34** | **30.98** | **26.46** | **19.37** | **16.56** |

- **The first version's dip was small and came from a real flaw.** `icons` held-out is 24 images, 6 per severity, so
  its 0.08 dB is about 1.9 dB in total: the size of one image's loss on `train`. The flaw was found on `train`, in the
  images that lost there. The held-out images weren't looked at one by one, so that they don't become a second train
  set, which leaves it unconfirmed that the same images were behind the dip.
- **That held-out score did prompt the search,** so the second score isn't as clean a test as the first. The tolerance
  was still chosen on `train`.
- **The gain on held-out is still far smaller than on `train`:** 0.02 dB against 0.22. With five images supplying most
  of `train`'s gain, a set of 24 may just hold few rounded rectangles. That isn't measured.

### Tight splines

A mouth's tips looked sharp but aren't: in the clean SVG they are small rounded ends, about a pixel across. They were
lost because of how a spline's handles are drawn. Each is a sixth of the chord between its knot's neighbours, whatever
the spacing, so knots crowded at a tip overshoot, and a long stretch beside them can't hold its shape.

`tight_spline` keeps the tangent's direction and makes each handle a third of its own segment's length. It costs no
extra numbers. On evenly spaced knots it draws about the same curve. It starts from knots where the outline bends, as
well as from the even and fitted ones.

**It only redraws a part's final shape** (`Target::tightened`, next to the holes). As an ordinary candidate in `fit`,
it changed the structure: flexible enough to make a bad `join` or cut look good, it merged an ellipse and a circle into
one blob (`synthetic` seed 31) and rearranged a part (`icons` seed 77, 1.8 dB). Like the rounded rectangle, it must also
be shorter *and* no further off. Tests: `a_mouths_tips_are_kept_by_a_tight_spline`,
`a_tight_spline_on_even_knots_is_about_the_same_curve`.

Measured on `train` (fidelity, dB), each row on top of the rounded rectangle within 1 px:

| | `synthetic` | `icons` | `icons` images up / down | worst `icons` loss |
|---|---|---|---|---|
| rounded rectangle only | 32.459 | 25.157 | | |
| in `fit`, shorter is enough | 32.380 | | | |
| in `fit`, no further off | 32.429 | 25.309 | 48 / 14 | 1.77 |
| in `fit`, within 1 px | 32.434 | 25.294 | 48 / 15 | 1.77 |
| **final shape, no further off** | **32.463** | **25.328** | **40 / 8** | **0.25** |
| final shape, within 1 px | 32.442 | 25.313 | 40 / 11 | 0.88 |
| final shape, with the rounded rectangle moved there too | 32.456 | 25.294 | | |

The rounded rectangle stays in `fit`. Moved to the final shape, it lost 2.7 dB on `icons` seed 33.

Both changes together, against the start of this work:

| | `synthetic` train | `icons` train | `icons` no corruption | `synthetic` held-out | `icons` held-out |
|---|---|---|---|---|---|
| before | 32.454 | 24.94 | 31.85 | 32.230 | 23.32 |
| **rounded rectangles and tight splines** | **32.463** | **25.33** | **32.96** | **32.469** | **23.51** |

- `icons` train: 59 images up and 7 down, the worst by 0.21 dB. By severity on held-out: clean 30.83 to 31.33, light
  about 26.5 to 26.72, medium 19.37 to 19.42, heavy 16.55 to 16.58.
- `synthetic` train: 7 up and 2 down. Seed 53 still loses 1.7 dB, and seed 77 loses 0.34.
- `synthetic` gains far more on held-out (0.24 dB) than on `train` (0.01). Held-out's clean and light images rose by
  0.5 dB each; with 6 images per severity, that may be one or two images.
- Tracing time on `icons` held-out: 2988 ms an image, against 2762 ms with the rounded rectangle alone.

**Next, on `icons`:** the worst images, since the mean in dB is dominated by them.

### The worst icons

With no corruption, the five worst `train` icons were at 12 to 15 dB, and for two reasons that have nothing to do with
edges.

**The subject vanished.** A chicken's pale body, a speaker's cone and a boot were all missing. `restore` found each of
them and each stood out, but each had under `RESTORE_SHARE` of its boundary with other parts (0.19 to 0.25). That test
is there for clutter, which lies alone in the background, and a big pale body with an eye and a comb on it looks the
same to it. No share separates them: at 0, the pale subjects come back, and clutter costs `synthetic` seed 62 12 dB.
What does separate them is how far a region stands out, in deviations of the image's own background noise: clutter is
at 1.4 to 2.6, and two real parts checked (`icons` seed 116) at 6.3 and 7.6. So a region `RESTORE_FAR` (5) deviations
out needs no contact. 3.5 and 5 give the same result, and 8 starts to lose. Test:
`a_light_body_far_from_the_background_needs_little_contact`.

**The background was painted.** On a swimmer in the sea and a train on its rails, the top corners are the background
and the bottom ones are the subject. All four fit a gradient exactly, which then matches almost nothing else, so the
subject had to draw the difference, including a tinted shape over the empty background. `Background::fit` now:

1. **Checks a gradient against its middle** (`gradient_holds`): at least a quarter of the border pixels in the middle
   half along it must be within `cut`.
2. **If that fails, looks for one colour:** two corners that agree, whose flat colour explains at least a quarter of the
   border. The top pair goes first, because a subject more often rests on the bottom edge than hangs from the top.

A tile on a gradient background fails the first test too, because the tile covers its border. That is why the second
matters: the first version replaced those gradients and lost 30.6 and 7 dB. On a noisy image the corners' spread
inflated the flat colour's tolerance until it accepted the tile, so a pixel now belongs to the two corners only if it
is nearer them than the other two. Test: `a_subject_on_the_bottom_edge_is_not_a_gradient_of_the_background`, and
`a_tile_on_a_gradient_background_keeps_the_gradient` as a guard for what the first version broke.

Measured on `train` (fidelity, dB):

| | `synthetic` | `icons` | `icons` images up / down | `icons`, no corruption |
|---|---|---|---|---|
| before | 32.463 | 25.33 | | 32.96 |
| `RESTORE_SHARE` 0 | 32.281 | 25.51 | 6 / 1 | |
| `RESTORE_SHARE` 0.15 | 32.463 | 25.40 | 3 / 0 | |
| `RESTORE_FAR` 5 alone | 32.463 | 25.51 | 6 / 1 | |
| gradient check, without the guards against tiles | 32.463 | 25.24 | 6 / 4 | |
| **`RESTORE_FAR` 5 and the gradient check** | **32.463** | **25.68** | **7 / 0** | **34.34** |

- `icons`: the swimmer is up 18.0 dB, the speaker 8.4, the train 7.4, and four more by 2.3 to 4.2. With no corruption,
  the worst image goes from 12.4 to 15.9 dB.
- `synthetic`: no image changes.
- Held-out, scored once: **nothing changes, to every digit** (`synthetic` 32.469, `icons` 23.51), and neither does the
  golden evaluation's small field. Seven of `train`'s 128 images are affected, so 24 images holding none is unremarkable
  (about one chance in four), but **held-out says nothing for or against this change.** Only `train` measures it.
- Still vanishing: on a noisier image a pale subject can stand out no further than clutter does (the chicken at
  `medium`, 2.3 deviations).
- The top-first order is a prior, not a measurement: a subject hanging from the top edge would lose.

**Next, on `icons`:** the images now worst with no corruption (seeds 109, 21, 27, 126 and 30, at 16 to 21 dB), seed
53's ellipse, and holes in `join`'s comparisons.

### Measuring as we go

Until now, every number above came from throwaway scripts, and `copycat.clean.v1` had never been run into the real
store: the public page had no leaderboard, and a change left no record except its prose here. Copy Cat has no training
runs (it is a fixed pipeline with tuned constants), so its record is a report, not a run:

    uv run python -m evaluator jobs/evaluator/specs/copycat.report.yaml --set params.out=NAME --set params.baseline=EARLIER

`copycat.report` scores the structured pipeline per image, notes the commit, writes `<data dir>/experiments/NAME.json`,
and with `baseline` says what moved, image by image. `params.heldout=true` adds the held-out sets, once per change.
`icons/train-control` is `icons/train` at the `clean` severity, so it says what shapes and layers cost without
corruption. (It is the `clean` severity's flat background, not the bare render "The worst icons" measured, hence 34.49
here against 34.34 there.) After a merge that changes the pipeline, re-rank and release:

    uv run redqueen evaluate copycat --wait
    uv run redqueen publish site --wait
    uv run redqueen dev deploy --no-site

The baseline, `copycat-ae2f606` (clean fidelity, dB):

| | mean | worst image | clean | light | medium | heavy | layer recovery |
|---|---|---|---|---|---|---|---|
| `synthetic/train` | 32.46 | 19.97 | 44.05 | 35.32 | 27.54 | 22.94 | 0.886 |
| `icons/train` | 25.68 | 11.32 | 33.92 | 29.18 | 22.09 | 17.54 | 0.626 |
| `icons/train-control` | 34.49 | 15.80 | 34.49 | | | | 0.709 |
| `synthetic/heldout` | 32.47 | 20.78 | 45.30 | 34.68 | 26.87 | 23.02 | 0.888 |
| `icons/heldout` | 23.51 | 10.24 | 31.33 | 26.72 | 19.42 | 16.58 | 0.544 |

### Lost pieces

Keeping two close greys apart on the tram (seed 89) cost 9.2 dB, and its colours were all right: the rail's far side was
simply missing. `join` joins two pieces of one colour with every other part free to lie above them, so the rail joined
across the pole. Its final shape can only pass under what `order` actually put above it. When `order` put the pole below,
the rail's shape covered the near side, and nothing noticed that the far side had gone.

So `leftovers` checks: each connected piece of a part that its shapes left uncovered gets a shape of its own, if that is
shorter than leaving its pixels wrong. A piece needs a core (`MIN_HALF` pixels with all their neighbours in the piece),
because a sliver along a shape's edge is that shape's misfit. Without that, `synthetic/train` lost 0.1 dB and 19 images.

Measured with `copycat.report`, against `copycat-ae2f606` (clean fidelity, dB):

| | before | after | images up / down |
|---|---|---|---|
| `synthetic/train` | 32.464 | 32.464 | 0 / 0 |
| `icons/train` | 25.683 | **25.758** | 3 / 2 |
| `icons/train-control` | 34.491 | **34.524** | 1 / 0 |
| `synthetic/heldout` | 32.469 | 32.469 | 0 / 0 |
| `icons/heldout` | 23.512 | **23.604** | 1 / 0 |

- `icons/train`: seed 63 is up 5.8 and seed 27 up 4.2. Seed 64 is down 0.38 and seed 18 down 0.12.
- `icons/heldout`: seed 10073 is up 2.2, at `medium`, and nothing else moves. It is one image, so it agrees with `train`
  more than it measures anything.
- The leaderboard doesn't change, since `copycat.clean.v1` is `synthetic`'s held-out images.
- This treats the symptom. `order` is still wrong on these images, and the extra shape is a second shape where the clean
  SVG has one, so layer recovery doesn't move.

### Two flat colours

A forced merge joins colours the artist meant to differ. Few pixels in the middle third between two centres tells two flat
fills from one colour's shading, since a ramp fills that valley. On its own that rule cost `synthetic/train` 0.08 dB,
and two things were wrong with it, neither about colours:

- **A rim is flat too.** A blurred edge's in-between colour is a cluster of its own, and its pixels are as far from the
  middle as a fill's. On seed 53 it became a muddy outline. So both clusters need a core: at least 20% of their pixels
  with all their neighbours in the cluster. The share hardly matters from 0.1 to 0.4. With that, `synthetic/train` didn't
  move on any image.
- **Noise.** That version passed `train` (`icons/train` 10 up, 0 down) and failed held-out: `synthetic/heldout` fell 32.469 to
  32.413, and `icons/heldout` was flat. Rather than debug on held-out images, we scored 400 fresh seeds, where the rule
  moved 11 images: 6 up, 5 down, with the 5 all at `medium` or `heavy`. On the worst (seed 20194, down 4.7) the two blues
  really were two colours, but keeping them apart shifted k-means, and a clutter cluster that had been 0.091 from the
  background, inside `background_delta`, no longer merged into it. So the rule applies only below a noise gate,
  `FLAT_SIGMA` 0.0025, barely over the 0.002 floor. All five losers are above it, and so are three small gains at `heavy`.

Measured with `copycat.report`, against `copycat-leftovers` (clean fidelity, dB):

| | before | after | images up / down |
|---|---|---|---|
| `synthetic/train` | 32.464 | 32.464 | 0 / 0 |
| `icons/train` | 25.758 | **25.929** | 8 / 0 |
| `icons/train-control` | 34.524 | **34.780** | 16 / 1 |
| 400 fresh seeds | 31.596 | **31.638** | 3 / 0 |
| `synthetic/heldout` | 32.469 | 32.469 | 0 / 0 |
| `icons/heldout` | 23.604 | 23.601 | 1 / 1 |

- `icons/train`: seed 27 is up 8.1 and seed 19 up 6.3; six others gain 0.35 to 2.2.
- `icons/train-control`: seed 106 is the one down, by 1.8.
- The gain has not shown on `icons/heldout`: seed 10032 is up 0.15, seed 10108 down 0.21. Those 24 images may simply not
  hold a case the rule fires on (8 of 128 on train), which is stage 5's point, but it is not confirmation.
- Held-out was looked at twice here, and the gate came after the first look. The gate's evidence is the fresh seeds, but
  `synthetic/heldout`'s two losers are also above it, so that set's "no change" is weaker than a first look would be.
- The leaderboard doesn't change.

Those fresh seeds are now `synthetic/validate` in the report: `train` is small enough to overfit, held-out mustn't be
debugged on, and this caught what `train` couldn't.

### An outline as seen

We set out to fix `order` with search: flip two touching parts whenever the whole order's description gets shorter. Per-part
bits said something else. On the pen (seed 76), the thin curved clip costs 1,605 bits with anything free to lie above it
and 381 with nothing free. More free pixels should never cost more, since the 381-bit shape is still allowed. The fitter
just never tries it: `completed` fills the free pixels inside the part's hull, and in a concave part that gives a worse
outline to start from. So `Target::fit_visible` also tries the visible outline as it is.

Measured with `copycat.report`, against `copycat-flat-colours-gated` (clean fidelity, dB; down is images down by more
than 0.05):

| Variant | `synthetic/train` | `synthetic/validate` | `icons/train` | `icons/train-control` |
|---|---|---|---|---|
| before | 32.464 | 31.638 | 25.929 | 34.779 |
| re-order search, no margin | 32.492 (17 down) | 31.672 (71 down) | 25.911 (26 down) | 34.727 (22 down) |
| re-order search, must save 200 bits | 32.463 (1 down) | 31.645 (3 down) | 25.948 (6 down) | 34.828 (3 down) |
| re-order search, must save 500 bits | 32.464 | 31.638 | 25.981 | 34.835 (2 down) |
| the outline as seen, in every fit | 32.379 (15 down) | 31.629 (52 down) | 25.979 (21 down) | 34.800 (21 down) |
| ... only in the shapes `order` and `join` compare | 32.470 (1 down) | 31.628 (15 down) | 25.915 (4 down) | 34.740 (7 down) |
| **kept: ... in a final shape, if 40 pixels' worth shorter** | **32.464** | **31.638** | **25.983** | **34.836** |

- **Kept:** the last row moves nothing on `synthetic` and no image down anywhere. Held-out, scored once: `synthetic/heldout`
  32.469 unchanged, `icons/heldout` 23.601 to 23.677 (seed 10050 up 1.82, nothing else moves).
- **Cost:** one more fit for each part something lies above. Icons go from about 3.8 to 4.1 s and `synthetic` from 0.33 to
  0.38 s per image, single runs each.
- **The re-order search is not kept.** With a margin it gets the same gain, and it doubled tracing time (icons 3.8 to 7.9 s,
  `synthetic` 0.33 to 0.60 s). A bound to skip hopeless flips lost the pen, because that bound assumed what the 1,605 bits
  disproved.
- **The tram was the wrong diagnosis.** Without a margin the search does put its pole on top, as in the clean SVG, for
  +0.02 dB. Its rail stays in pieces, which is `join`'s doing.
- **This treats a symptom too.** The pen's clip is still painted under its body, where the clean SVG has it on top, so
  layer recovery barely moves (`icons/train` 0.6309 to 0.6314). The gain is mostly one image: seed 76 up 6.5, seed 34 up
  0.4 to 0.6.
- Changing the shapes `order` compares flips as many pairs wrong as right: `order`'s fractions depend on hidden shapes
  sprawling, and that is what it reads as "lies beneath".

### The staged plan

Each stage measures every change with `copycat.report` against the last one, and scores held-out once, at the end of
the change. The order comes from what a perfect tracer could reach. A warp can't be undone, so the clean SVG, drifted
and warped as each image was, then scored against itself, is roughly the most a tracer can expect on that image (`icons/train`,
dB):

| | `clean` | `light` | `medium` | `heavy` |
|---|---|---|---|---|
| now | 33.92 | 29.18 | 22.09 | 17.54 |
| that limit | 60 (the cap) | 36.40 | 25.52 | 21.06 |

It isn't a strict limit: 8 images of 128 beat it, because a fitted shape can smooth a warp away. But most of the 8.8 dB
between `icons/train` and its control is not there to win. No single step explains it either: with one step removed,
warp is worth 1.9 to 2.6 dB, every other step under 0.8, and removing JPEG makes things *worse* (-0.4 and -1.1), since it
smooths noise. So the room is in `clean` and `light`, where shapes and colours are what's wrong.

1. **Close colours and the layers around them.** A forced merge joins colours the artist meant to differ: `merge_delta`
   takes the trophy's amber into its yellow. Lowering it (0.05, 0.03, 0.015) loses overall, on both sets. "Lost pieces",
   "Two flat colours" and "An outline as seen" took `icons/train-control` from 34.49 to 34.84. **Stopped there, short of
   the 35.5 dB target**, which stays open with `synthetic/train` no worse. The last idea tried was loosening `join`'s
   guard: a piece counts as a detail only if the part covering it also owns half of what's around it. Measured
   (`copycat-join-surround`) and not kept: control 34.836 -> 34.832, 8 images up and 4 down, layer recovery 0.7153 ->
   0.7195. It lost 3.6 dB on seed 96 (a red slash on a tail joined to the tail) and 2.7 on seed 26 (two soles joined into
   a sliver), and won about 1 dB on four images. Neither that share (0.41 to 0.46 on both sides) nor the bits saved (81 for
   the soles, 48 to 92 for good joins) tells them apart. The tram didn't move at all, and its most visible miss is elsewhere:
   its rounded corner comes out a polygon. Telling a real split from a real detail needs more than one fit's bits.
2. **The worst images without corruption.** Seeds 109 and 21 are at 16 and 17 dB in the control: a fine mesh averaged away,
   and a pale bowl lost to the background. The rest are above 20. Done when none is under 20 dB.
3. **Corruption on `icons`.** About 3.5 dB below the limit at both `medium` and `heavy`, with parts vanishing or noise
   becoming parts (held-out alpha IoU 0.84, against 0.99 at `clean`). Done when each is within 2 dB of the limit.
4. **Layer recovery on `icons`.** 0.63, against 0.89 on `synthetic`: the picture can match while the layers don't, and
   the layers are what a person editing the SVG wants. Done at 0.75 on `train`.
5. **A bigger validation set.** 24 icons can't see a change that affects 7 images in 128: "The worst icons" scored
   identically on them. Done as `broad`; see "The `broad` set".
6. **`icons` on the public leaderboard.** `copycat.clean.v1` is `synthetic` only, which flatters every entrant against
   real artwork. Done as `copycat.clean.v2`, which ranks on all three sets; see "`copycat.clean.v2`".

[0023](0023-process-traces-and-research-log.md)'s research log is built: each report is a run under `/runs`, with progress
on the game page, so the direction can be set from the results rather than from a relayed summary. Its process traces then
come before the ladder's search and gradient rungs, which would be hard to debug without them.

### The `broad` set

`icons` has been tuned on, and its 24 held-out images can't see a change that moves a few. `broad` is 616 more icons, none of
them in `icons`, from two families, so a result isn't one artist's style: 326 Twemoji (v17.0.3, CC-BY 4.0) and 290 Flat Color
Icons (Icons8, MIT). 120 are held out, 30 per severity. 53 of the 669 downloaded were refused rather than approximated, most
for a transform. `copycat.report` measures `broad/train` and `broad/heldout` from here on.

**Other people's artwork isn't in the repo.** `copycat_sources.json` pins each set: a tag or commit, the exact files, and digests of
the files, of the SVGs converted from them, and of the held-out images. `arena.copycat_artwork` downloads and builds a set on
first use (under `<data dir>/experiments/copycat-sets/`, about 40 s for both) and refuses one that doesn't match, since a score
means something only on the same images. `icons` moved out the same way, and a fresh download rebuilt all 768 SVGs and every
held-out image byte for byte. Reads never download. `synthetic` stays checked in: we generate it, and the leaderboard ranks on
it. No test needs the network; they serve a small set from local files.

Where the pipeline stood, on `broad/train` (`copycat-broad-base`; clean fidelity, dB):

| | all | clean | light | medium | heavy |
|---|---|---|---|---|---|
| Flat Color Icons | 28.43 | 40.22 | 29.98 | 23.56 | 19.73 |
| Twemoji | 25.00 | 33.79 | 27.10 | 21.42 | 17.85 |

Held-out (26.80) agrees with train (26.59). The worst images with almost no corruption were mostly a pale part that vanished
into a pale background.

**A gradient was judged by a flat colour's spread.** `Background::cut` comes from the border's spread about a flat fit, even
when a gradient is chosen: as wide as 0.21 on these images, where a flat background's floor is 0.012. So a pale part on a
gradient didn't "stand out". Judged by the gradient's own spread, in `enclose` and `restore`:

| | `synthetic/validate` | `broad/train` | up / down | |
|---|---|---|---|---|
| on every image | 31.638 -> 31.410 | 26.590 -> 26.724 | 23 / 13 | faint clutter stands out too: -15.9 dB on one |
| **only where σ is under `FLAT_SIGMA`** | **31.638** | **26.705** | **8 / 2** | clutter comes with noise in `corrupt.v1` |

That gate is a real limit, not a fix: a pale part on a noisy gradient is still lost, because nothing here tells it from clutter.

**A crisp edge divides what colour can't.** A white sheet and a white background can fall in one k-means cluster, and that
region's mean is the background's. So `restore` runs a second time on what's left, on a near-noiseless image, divided by
pixels more than 4σ from a neighbour. The first attempt replaced the first pass and lost 0.16 dB: the edge pixels sat between
a part and what it touches, so it no longer touched anything, and thin bars were all edge. As a second pass, whose edge pixels
join a region before anything is tested, it can only add. With the contact rule, 3 images rose and the mean moved +0.018 dB;
without it, since clutter is what that rule is for, 5 rose by up to 9.3 dB, none fell, and the mean moved +0.057 dB.

Together (`copycat-pale-parts`), with held-out scored once:

| | `synthetic` | `icons` | `icons` control | `broad` |
|---|---|---|---|---|
| `train` | 32.464 unchanged | 25.983 unchanged | 34.836 -> 34.949 | 26.590 -> 26.763 (12 up, 2 down) |
| held-out | 32.469 unchanged | 23.677 -> 23.949 | | 26.803 -> 26.845 (1 up, 1 down) |

The two that fell on `train`, seeds 376 (-6.1 dB, still 28) and 15 (-4.2 dB, still 26), weren't looked into further.

Not kept: running `enclose` again after `restore`, on the idea that a restored part's ragged holes would pass its test. +0.002
dB, with the `icons` control down (`copycat-enclose-again`).

**Next**, from the worst low-corruption images that remain:
- restored pale parts are ragged, with holes (seeds 314, 554, 120, 384);
- a background that isn't where the corners say (488, 473): the sky drawn as a part, the tram's rails taken for the background;
- pale parts under noise, the limit above, which needs something other than colour to tell them from clutter.

**Pale parts under noise, tried and dropped.** Averaging a pixel with the background pixels around it removes noise, so a
third `restore` pass could find pale parts on noisy images: anything off the background by more than the border is.

| Limit | `synthetic/validate` | `broad/train` | up / down |
|---|---|---|---|
| 3 x the border's median | 31.638 -> 30.367 | 26.763 -> 26.548 | 21 / 43 |
| 6 x the border's median | 31.638 -> 31.257 | 26.763 -> 26.769 | 13 / 13 |
| the border's farthest | 31.638 -> 31.365 | 26.763 -> 26.784 | 9 / 9 |
| 1.5 x the border's farthest | 31.638 -> 31.551 | 26.763 -> 26.754 | 4 / 5 |

It finds them (+16 dB on one image) and restores clutter just as readily (-17 dB on another). Over the candidates on flat and
gradient backgrounds, 88% of the area was a real part; on cluttered ones, 16%. Neither contact, solidity, size nor colour
distance separated the two on a cluttered background, and counting regions alone in the background didn't either: clutter
joins into one region that touches the subject. The dark specks `corrupt.v1` adds would give clutter away, but that would
be tuning to our own corruption, so a pale part on a noisy background stays lost.

**A flat background's cut was stale too.** `cut` is computed before the last refit, so a subject along the border leaves it
wide on a clean image: 0.0875 on a flat background whose floor is 0.012. `judged_cut` now applies to flat backgrounds as well,
which removes a condition (`copycat-flat-judged`). One image moves: seed 532, +2.77 dB, so `broad/train` 26.763 -> 26.768. No
other group moves, held-out included, so this is a correction more than a gain.

Two other low-noise changes lost and were dropped. Dropping the contact rule from `restore`'s first pass, so edges don't cut a
whole part into pieces, fixed one ragged image (+4.68 dB) and hurt two (-7.45, -2.33). Holding a gradient to 12σ in
`gradient_holds`, for tiles that fill the bottom of the border, cost -16.4 and -13.4 dB on two images and helped none. The
ragged restored parts remain.

**A subject across both bottom corners.** The background fit expects the background at the corners. When a subject covers
both bottom corners (a banknote, a bed, a tram on its rail), three things went wrong:

| Image | What happened | The fix |
|---|---|---|
| 397, a banknote | The border's own cut, widened by the subject on half of it, let a gradient from the pink background into the purple band hold, and the lavender note joined the background | `gradient_holds` judged by the corner fit's cut |
| 488, a bed | The two top corners differ by the background's own slight gradient, more than the corner cut, so they didn't count as one colour | Two corners are one colour if nearer each other than a quarter of the way to the other two |
| 473, a tram | The sky shares 21% of the border, under the 25% asked of it, and the rail at the bottom took its place | The colour's pieces at the two corners must cover 5% of the image, not 25% of the border |

The share rule existed to tell this from a tile on a gradient background, which shows the gradient only at its cut corners.
In its test they cover under 3% of the image; the tram's sky, even split in two, covers more than 5%. Connecting the two
corners through the image would have been the cleaner rule, but the tram's pantograph runs from the wire to the roof and
splits its sky in two. Without a guard, the looser rules took two noisy images' corners for two colours (-9.0, -4.2 dB), so
the two colours must also be further apart than twice the corner cut (`copycat-background-misfit`):

| Clean fidelity, dB | Before | After | Images |
|---|---|---|---|
| `broad/train` | 26.768 | 26.895 | 4 up (397 +27.0, 488 +18.2, 473 +17.5, 481 +0.3), 0 down |
| `broad/heldout` (scored once) | 26.845 | 26.928 | 2 up (+5.5, +4.4), 0 down |

No other group moves.

**A restored region keeps its colours.** `restore` gave each region it took one colour, the mean of its pixels. A pale
calendar and its grey squares, both near enough the background to join it, came back as one grey. On a near-noiseless
image a region taken is now divided by its crisp edges, and its pieces grouped by colour, each measured off its edges
(`colours_of`). A colour past the largest must stand out by itself, or a blurred region's fringe, the background's colour,
comes back as slivers (73, a light image: 11 shapes to 27, -2.2 dB, before that rule) (`copycat-restored-colours`):

| Clean fidelity, dB | Before | After | Images |
|---|---|---|---|
| `broad/train` | 26.895 | 27.026 | 15 up (147 +11.4, 401 +8.7, 441 +8.7, 1 +7.9), 2 down (-1.5, -1.0) |
| `icons/train` | 25.983 | 26.000 | 1 up (+3.8), 1 down (-1.6) |
| `icons/train-control` | 34.949 | 35.023 | 5 up, 0 down |
| `icons/heldout` (scored once) | 23.949 | 24.494 | 2 up (+10.7, +2.4), 0 down |
| `broad/heldout` (scored once) | 26.928 | 26.920 | 1 up (+1.1), 1 down (-2.0) |

Two other changes for pale parts were dropped. Holding the border's fit to the corner fit's cut, which stays tight under
a gradient, lost on every set (`broad/train` 26.895 -> 26.755, 19 up, 26 down), and even on near-noiseless images alone
(9 down by up to 17 dB, none up): the wide cut also keeps faint clutter and compression's ringing in the background.
Running the pass divided by edges first, instead of dividing what the first pass takes, fixed as many images as it broke.

### `copycat.clean.v2`: the leaderboard on real artwork

`copycat.clean.v1` scored only `synthetic`, so the public number was the pipeline's best one (32.5 dB) and no change
measured on `icons` or `broad` could move it. `copycat.clean.v2` scores the held-out images of all three sets (24
`synthetic`, 24 `icons`, 120 `broad`), downloaded and built on first use like `copycat.report`'s. It ranks by the mean
of the three sets' means, so each set counts the same: a mean over all 168 images would be mostly `broad`. The interval
is the same mean's (each set's variance over its count, summed, over nine). Each record keeps `by_set`, and the
leaderboard shows one column per set. v1 is unchanged and still runs; its golden case still matches.

| entrant | v2 | synthetic | icons | broad | v1 |
|---|---|---|---|---|---|
| Structured pipeline | **27.96** ± 1.75 | 32.47 | 24.49 | 26.92 | 32.47 |
| VTracer (defaults), background deleted | 15.01 ± 1.08 | 20.18 | 11.95 | 12.89 | 20.18 |
| VTracer (polygons) | 12.82 | 11.37 | 13.88 | 13.21 | 11.37 |
| Hill climb (64 shapes) | 12.61 | 11.43 | 13.44 | 12.95 | 11.43 |

The order holds, but the gaps change. Deleting VTracer's bottom layer was worth 9 dB on `synthetic`, whose backgrounds
are a separate layer, and costs 1.9 dB on `icons` and 0.3 on `broad` against VTracer's defaults. The pipeline still
leads the next entrant by 13 dB. It takes 0.4 s an image on `synthetic`, 4.0 on `icons` and 3.5 on `broad`.

Seeds repeat across sets (`synthetic` and `broad` both have 10002), so an image is named `set/seed` everywhere:
`GET /games/copycat/targets/{set}/{seed}` and `?image=icons/10057` on the page.

### Search over the structure, measured before it was built

The ladder's next rung was search over the whole structure. Before building one, we measured how much it could win:
a greedy hill climb over the pipeline's own SVGs, with clean fidelity itself as the judge (an oracle, on `train`
only). Its moves were deleting a shape, swapping two neighbours in paint order, and recolouring a shape to another of
the SVG's fills. Even judged by the answer, it gained only +0.15 dB on `synthetic`, +0.08 on `icons`, +0.04 on the
control and +0.16 on `broad`. A search judged by L could only do worse, so it wasn't built.

Splitting the error by where it falls says why. We set each class of pixel right, and measured the dB gained (`train`):

| | coverage wrong | edges | colour inside the silhouette |
|---|---|---|---|
| `synthetic` | +2.21 | **+4.25** | +0.99 |
| `icons` | +3.34 | +1.63 | **+3.59** |
| `icons` control | +1.07 | **+4.01** | +3.33 |
| `broad` | **+3.28** | +2.65 | +2.43 |

Edges are unbiased (the mean signed offset is +0.01 to 0.05 px), but they are off by 0.2 to 0.34 px. Each part was fitted
alone to its own soft mask, never against the image as drawn.

### Polish: the shapes against the whole image

`copycat_polish.rs` runs after every part is fitted. It judges every shape as it is drawn: under the shapes above, over the
ones below and the background estimate, against the input image. The error is the squared error of that composite, with
the rasterizer's exact coverage, so what is polished is what the SVG draws. Coordinate descent nudges each shape's
numbers in the form it was fitted in: a spline's knots, a rounded rectangle's five numbers, or a primitive's own. A plain
path's anchors move with their handles. The structure is fixed: the number of shapes, their kind, colour and order.

| Run (train, dB) | `synthetic` | `icons` | control | `synthetic/validate` | `broad` | smoothness, `synthetic` |
|---|---|---|---|---|---|---|
| before (`copycat-restored-colours`) | 32.46 | 26.00 | 35.02 | 31.64 | 27.03 | 1.068x |
| a: every geometry's own numbers | 33.88 | 26.64 | 36.66 | 33.15 | 28.05 | 1.214x |
| b: in the fitted form | 33.61 | 26.54 | 36.16 | 32.91 | 27.96 | 1.072x |
| **d: b, steps 1 px to 0.02 px, 6 rounds** | **33.90** | **26.72** | **36.60** | **33.31** | **28.35** | **1.074x** |

Run a polished a spline as its cubics, which broke its smoothness (a smooth knot turned into a corner). Coarse steps
alone (e) reached 33.68 and fine steps alone (f) 33.87, so d keeps both. It adds 0.3 s an icon (4.1 s -> 4.4 s), and
0.02 s on `synthetic`. Most images that lose are `light` ones, where polishing follows the warp. On held-out, scored once
(`copycat-polish`):

| Clean fidelity, dB | Before | After | Images |
|---|---|---|---|
| `synthetic/heldout` | 32.469 | **34.666** | 18 up, 4 down (at most -0.24) |
| `icons/heldout` | 24.494 | **25.093** | 15 up, 5 down (at most -0.48) |
| `broad/heldout` | 26.920 | **27.952** | 97 up, 11 down (at most -2.06) |

On `copycat.clean.v2` the pipeline goes from 27.96 ± 1.75 to **29.24** ± 2.10 dB. Layer recovery is level everywhere.
The control reaching 36.60 meets the staged plan's 35.5 dB target, which shape and
colour rules alone had stopped short of at 34.84.

Not kept: polishing each palette colour too (every shape of one fill, together) lost 0.13 dB on `synthetic/train` and
0.15 on `synthetic/validate`. Fitting colour to the image fits the corruption's colour drift, which the clean SVG
doesn't have.

After polish, edges are still worth +2.0 to 4.4 dB. An error map of the control shows why: the error now lies along
almost every outline, where a few knots can't follow a Twemoji curve. A few small details are still missing (a boot's
streak, a mosquito's eyes). The next gains are in the shapes' vocabulary, not their placement.

### Growth: a point where the outline needs one

After polish, the error lay along outlines a few knots couldn't follow. Polish could move a point but never add one: a
part's point count was settled when it was fitted alone, to its own soft mask. Split by the kind of shape that drew
them, a shape's outline pixels (`icons` and `broad`, `train`) are 42-47% polygons and 31-38% splines; primitives 13%,
plain paths 3-6%, rounded rectangles 2-3%. So growth works on the two that carry the outline.

After polish's descent, `copycat_polish.rs` looks along each spline and polygon for the stretch it misses most, puts a
point at its midpoint (the cubic's midpoint for a spline, the edge's for a polygon), settles it in steps of 0.5 to 0.1 px and
keeps it if the error it saves pays for it, up to 32 points a shape; then every shape is descended once more, in
steps of 0.25 to 0.02 px. A point's price is its two coordinates' bits in pixel-error bits, scaled by the shape's contrast (the mean
squared difference between its fill and what is drawn under its edge), so a pale shape's edge buys fewer points, and
by `POINT_PRICE`. Tuned on `train` (dB, against `copycat-polish`):

| `POINT_PRICE` | `synthetic` | `icons` | control | `synthetic/validate` | `broad` | s / icon | s / `broad` |
|---|---|---|---|---|---|---|---|
| none (`copycat-polish`) | 33.90 | 26.72 | 36.60 | 33.31 | 28.35 | 7.0 | 4.4 |
| 1 (the description length) | 33.90 | 26.77 | 36.65 | 33.31 | 28.37 | | |
| 0.25 | 33.90 | 26.85 | 36.77 | 33.31 | 28.41 | | |
| 0.1 | 33.90 | 27.00 | 37.19 | 33.33 | 28.51 | | |
| 0.05 | 33.93 | 27.17 | 37.68 | 33.35 | 28.67 | 8.4 | 5.8 |
| **0.02** | **33.99** | **27.41** | **38.33** | **33.37** | **28.83** | **8.4** | **6.9** |
| 0.005 | 33.99 | 27.48 | 38.65 | 33.39 | 28.98 | 8.7 | 7.2 |

The gain rose as the price fell, and `synthetic/validate` (debug images never tuned on) never fell, so the points were
outline the image asked for, not noise: the control, with nothing to over-fit, gained most. At the description length
itself almost no point paid, since an outline's error lies on thin edge pixels and is cheap next to two coordinates.
Times are back to back on every sixth `train` image (between batches they varied by a quarter). 0.02 takes `icons`
half as far again as 0.05 for the same time and costs `broad` one second more for +0.16 dB; 0.005 adds little.
Knots alone (no polygons) reached only `icons` 26.79 at 0.05: most of the outline was polygons.

**The outlines came out wiggly.** Every report records `smoothness_ratio`, an outline's total turning over the clean
SVG's, and growth at 0.02 had pushed it from 1.07 / 0.98 / 0.95 / 1.12 / 1.02 (the five `train` groups above, in order)
to 1.46 / 1.56 / 1.23 / 1.46 / 1.63. A point that follows a pixel of the corruption's noise saves error like one that
follows a real bend, and a coordinate's price can't tell them apart. So turning costs bits too: a grown point (and its
settling) must also pay `TURN_BITS` for every radian of extra turning it adds to its shape, at the same contrast scale. A
convex outline turns a full circle however many points it has, so only dents and zigzags pay. Tuned on `train` with
`POINT_PRICE` 0.02 (dB, then smoothness):

| `TURN_BITS` | `synthetic` | `icons` | control | `synthetic/validate` | `broad` |
|---|---|---|---|---|---|
| none (`copycat-polish`) | 33.90 / 1.07 | 26.72 / 0.98 | 36.60 / 0.95 | 33.31 / 1.12 | 28.35 / 1.02 |
| 0 | 33.99 / 1.46 | 27.41 / 1.56 | 38.33 / 1.23 | 33.37 / 1.46 | 28.83 / 1.63 |
| 2 | 33.98 / 1.23 | 27.39 / 1.19 | 38.21 / 1.01 | 33.36 / 1.24 | 28.81 / 1.25 |
| 8 | 33.97 / 1.14 | 27.36 / 1.09 | 38.09 / 0.98 | 33.35 / 1.17 | 28.78 / 1.13 |
| **32** | **33.94 / 1.10** | **27.30 / 1.02** | **37.95 / 0.97** | **33.35 / 1.13** | **28.75 / 1.07** |

At 32 bits outlines turn within 5% of what they did before growth, for about 0.1 dB on `icons` and almost nothing
elsewhere. Greedy growth isn't monotone in the price: a test's star kept its ten knots at a turning price of 2 but not
at 5, since an early point that would have unlocked the later ones no longer paid on its own.

Held-out, once each (`copycat-grow` without the turning cost, marked instructive; `copycat-grow-turn`, kept):

| | `synthetic` | `icons` | `broad` | smoothness |
|---|---|---|---|---|
| v22 (`copycat-polish`) | 34.666 | 25.093 | 27.952 | 1.03 / 0.71 / 0.96 |
| growth, turning free | 34.711 | 25.635 | 28.444 | 1.33 / 1.08 / 1.48 |
| **v23: growth, turning at 32 bits** | **34.719** | **25.537** | **28.366** | **1.05 / 0.75 / 1.00** |

Fidelity alone would have kept the wiggly version. The live film's progress weights were re-measured for it too (polish
is now 30-47% of a slow image's time, not a fifth).

### Versions: the MDL tracer, numbered

"The structured pipeline" named a family, not an algorithm: each kept change replaced the entrant under the same name,
so a 29.0 dB result and a 34.7 dB one both said "structured pipeline". The pipeline is now the **MDL tracer**, numbered
one version per kept change. The leaderboard entrant keeps its id (`baseline:structured`, so links, goldens and the
static site hold), takes the label "MDL tracer N" from the evaluator's `MDL_TRACER_VERSION`, and records `version` and
`commit` in `metrics.copycat`. Only the current version is on the leaderboard: each evaluation replaces the last.

The versions are listed in `apps/frontend/app/data/copycatVersions.ts`, reconstructed from the merged PRs, the
`copycat.report` records and the held-out results in this doc; a test holds the last one equal to `MDL_TRACER_VERSION`.
Each row is the change, its PR and merge commit, its report (from version 14, when reports began; a report's own commit
may be `+dirty`, the change measured before it was committed) and its held-out scores, taken once. Learn's case study
draws from it, and the game page's progress panel names each kept report's version. Two kept changes the case study had
left out are in it: a flat background judged by its own spread (v19, held-out unchanged) and a subject across both
bottom corners (v20, `broad` 26.845 -> 26.928).

| v | change | PR | synthetic | icons | broad |
|---|---|---|---|---|---|
| 1 | The MDL tracer | #70 | 29.0 | | |
| 2 | Refine every shape | #72 | 30.8 | | |
| 3 | Smooth splines | #73 | 31.38 | | |
| 4 | Cut a part in two | #75 | 31.43 | | |
| 5 | Knots by least squares | #76 | 31.99 | | |
| 6 | A point must earn its keep | #77 | 32.00 | 18.52 | |
| 7 | Join only what an occluder cut | #80 | 31.999 | 18.77 | |
| 8 | A background that reaches the corners | #81 | 32.003 | 19.41 | |
| 9 | Paint order by fraction | #82 | 32.23 | 20.88 | |
| 10 | A corner fit noise can't break | #83 | 32.23 | 21.37 | |
| 11 | Light parts given back | #84 | 32.23 | 22.63 | |
| 12 | Holes | #85 | 32.23 | 23.32 | |
| 13 | Rounded rects, tight splines | #86 | 32.469 | 23.51 | |
| 14 | A pale subject restored, two colours not a gradient | #87 | 32.469 | 23.512 | |
| 15 | Lost pieces | #89 | 32.469 | 23.604 | |
| 16 | Two flat colours | #90 | 32.469 | 23.601 | |
| 17 | An outline as seen | #91 | 32.469 | 23.677 | |
| 18 | Pale parts | #97 | 32.469 | 23.949 | 26.845 |
| 19 | A flat background judged by its own spread | #98 | 32.469 | 23.949 | 26.845 |
| 20 | A subject across both bottom corners | #99 | 32.469 | 23.949 | 26.928 |
| 21 | Restored colours | #100 | 32.469 | 24.494 | 26.92 |
| 22 | Polish | #102 | 34.666 | 25.093 | 27.952 |
| 23 | A point where the outline needs one | #104 | 34.719 | 25.537 | 28.366 |
| 24 | Pale fills kept apart | #109 | 34.719 | 25.538 | 28.366 |
| 25 | Thin features kept | #109 | 34.719 | 25.797 | 28.382 |
| 26 | Small enclosed parts kept | #109 | 34.719 | 25.938 | 28.519 |
| 27 | Joins undone when the pieces say it far more briefly | #109 | 34.719 | 25.950 | 28.519 |
| 28 | Thin features kept on near-noiseless images | #109 | 34.719 | 25.950 | 28.549 |
| 29 | Outlines whose knots carry their own tangents | | 35.599 | 27.517 | 29.294 |

`icons` was first scored on v6, unchanged, when the set arrived (#79); `broad` on v18, in the PR that added the set (#97).
`copycat.clean.v2` scored v21 at 27.96, v22 at 29.24, v23 and v24 at 29.54, v25 at 29.63, v26 and v27 at 29.73, v28 at 29.74, and v29 at 30.80. Experiments not kept are labelled against the version they
started from (e.g. "v22 + point growth, price 0.02"); their reports already name their baseline. Older versions were not
re-scored on `copycat.clean.v2`: that would mean running each commit again, and nothing here needs it.

### After v23: where the error is, and three leads that didn't pay

Nothing here was kept, so there is no report or run: these are measurements on `train` (held-out never looked at),
made with throwaway scripts against `copycat-grow-turn`'s SVGs, to decide what to build next. They rule out more than
they find.

**The error split, re-measured on v23** (each class of pixel set right, dB gained; as in "Search over the structure"):

| | coverage wrong | edges | colour inside the silhouette |
|---|---|---|---|
| `synthetic` | +2.04 | **+3.75** | +1.28 |
| `icons` | **+2.90** | +2.16 | **+3.66** |
| `icons` control | +0.83 | **+4.58** | +3.84 |
| `broad` | **+2.97** | +2.42 | +2.79 |

Polish and growth took edges down, so colour and coverage now lead on `icons` and `broad`. But the colour column is not
one lever, and most of it isn't reachable:

- **Fill colours are worth little.** A palette oracle (every fill in our SVG replaced, by text substitution, with the
  clean render's mean colour under it, so the structure is untouched) gains +0.59 `synthetic`, +0.56 `icons`, +0.53
  `broad`, +0.01 the control. (An earlier per-shape oracle, recolouring by topmost cover, was broken: it *lost* 0.69 on
  the control.)
- **Most of that is drift, which one image can't reveal.** `icons/train`'s light-to-heavy images re-corrupted with a
  step switched off (every other random draw kept), traced, and given the oracle:

  | `icons/train`, not clean | fidelity | the oracle adds |
  |---|---|---|
  | as corrupted | 23.69 | +0.75 |
  | no drift | 24.30 | +0.26 |
  | no shading | 24.01 | +0.68 |
  | neither | 24.46 | +0.13 |

  Drift moves each part's colour at random, so about 0.5 of the 0.75 is gone from the image. What is honestly left
  (shading under 0.1, estimation under blur, noise and JPEG about 0.1) is worth +0.1 to 0.15 dB on the leaderboard.
- **A palette prior would win more, for the wrong reason.** 88% (`icons`) and 87% (`broad`) of the clean colours in one
  half of `train` occur exactly in the other half's palettes: the icon families share colours. Snapping to a palette
  learned from `train` would undo much of the drift on those two sets and nothing on `synthetic` (random colours), and
  wouldn't help a logo from anywhere else. If ever built, it is a separately labelled entrant, not the MDL tracer.
- **Missing details are a small pot too.** Taking each region of the clean render (a 4-connected run of one colour) that
  our render colours wrongly over most of its area, setting it right gains, per image:

  | | every such region | thin ones (1-2 px) | ones our colour gave to a neighbour |
  |---|---|---|---|
  | `icons` | +1.55 | +0.21 | +0.37 |
  | `broad` | +1.10 | +0.13 | +0.29 |
  | `synthetic` | +0.60 | +0.05 | +0.19 |

  Most of the "every" column is large parts (thicker than 9 px) at `heavy`, coloured wrong by drift. So most of the colour
  column is error spread over whole parts, not details never drawn.

**Shading splits a flat part along a jagged seam.** The largest single losses (`icons` 79, 133, 151; `broad` 497, 295,
36) are one flat part traced as two colours, the seam between them ragged with noise: corrupt.v1's shading (up to ±25%
at a `heavy` corner) puts the halves further apart than `merge_delta`. Dividing the shading out before a second trace
(the subject's share of each pixel only, never darkening a channel clipped at white) should merge them. It barely does:

| `icons/train` | v23 | shading fitted to the first trace's parts | fitted to pairs of flat pixels 24 px apart | the true shading |
|---|---|---|---|---|
| all | 27.297 | 27.424 | 27.333 | 27.376 |
| `medium` | 22.642 | 22.556 | 22.352 | 22.377 |

(The two fitted columns ran before the white guard was added.) Even the true gradient (replayed from the corruption's
seed, and checked against it) gains +0.08, where switching shading
off in the corruption gains +0.32. The difference is the tracer's fragility: a small change to the input flips
decisions. On seed 21 a white cup, no longer brightened, falls within `background_delta` of a near-white background and
is lost (-2.8 dB). Fitting to the first trace's parts is also biased: the two halves it already split soak up the
gradient (its estimate is about half the true one at `heavy`).

**Picking the shortest of several traces.** The same fragility is variance, and variance is something to search over:
each image traced with k-means seeds 0-3 and two deshaded versions (seeds 0-1), one kept by a description length computed
without the clean answer (residual bits against the image, composited over a plane fitted to the border, plus 24 bits a
colour and 2 x 10 bits a node):

| `train` | v23 | mean of 4 seeds | best of 6 by the clean score | shortest of the 4 seeds | shortest of the 6 |
|---|---|---|---|---|---|
| `icons` | 27.297 | 27.339 | 27.782 | **27.480** | 27.450 |
| `synthetic` | 33.942 | 33.937 | 34.488 | 33.868 | 33.800 |
| `broad` | 28.751 | 28.743 | 29.192 | 28.749 | 28.805 |

A perfect pick would gain about +0.45 dB everywhere; this one wins `icons` (+0.53 at `light`) and loses `synthetic` (-0.34
at `medium`). Its description is a stand-in, not the tracer's: its background, its σ, its coordinate costs. **The one
lead left**: compute the description natively, from the pipeline's own background, noise estimate and `Costs`, and pick by
it. The judge's `error` already composites the whole render over the background. It costs a trace per candidate, 4-6 times
the time, in the browser too, so it must win on every set to be worth that.

### v24: pale fills kept apart

Found by [0025](0025-copycat-prior-art-and-direction.md)'s look at where the tracer loses on clean images (the worst
`icons/train` images at the `clean` severity). Two bugs, both pale flat colours lost near an off-white background.
- **`enclose` averaged what it took back.** A CD's pale and paler wedges both joined the background (within
  `background_delta` of it). `enclose` then gave every enclosed background component back as *one* label with one mean
  colour: two wedge colours became their average (#ccd6dd and #f5f8fa drawn as #e1e7eb). Now, on a near-noiseless image,
  each component is first divided by its crisp edges into its colours (`colours_of`, as `restore` divides what it takes),
  and a colour within one deviation of the background stays a hole. Test: `enclosed_pale_colours_are_divided_not_averaged`.
- **A pale fill nearer another colour than the background was lost.** A steak's pale rim (#ffe8b6) is 0.081 from its
  background and 0.058 from the darker cream beside it. `segment` forced it into the background; the next k-means pass
  then gave its pixels to the nearer cream, so `restore` (which takes pale parts back from the background only) never saw
  them. Now, on a near-noiseless image, a cluster that is one flat fill (`flat_fill`: a core nearly all within 4σ of its
  centre), more than 4σ from the background, *and* nearer another colour than the background, is not absorbed, as
  `two_flat_colours` already kept two close colours apart. Test: `a_flat_fill_near_the_background_keeps_its_colour`.
- **A first version kept every such flat fill apart** (`copycat-pale-fills`, not kept). That lost what `restore` had
  been giving back: a shuttlecock's white feathers, 0.02 from the background, became an ordinary cluster and weren't
  drawn (−6.5 dB on that image, −5.9 on a `broad` one). The "nearer another colour" condition confines the change to the
  case `restore` can't reach.

Measured with `copycat.report`, `copycat-pale-fills-2` against `copycat-grow-turn` (clean fidelity, dB):

| | before | after | up | down |
|---|---|---|---|---|
| `icons/train` | 27.297 | **27.499** | 2 | 0 |
| `icons/train-control` | 37.947 | **38.251** | 4 | 0 |
| `broad/train` | 28.751 | **28.798** | 3 | 1 (−0.15) |
| `synthetic` (train, validate, held-out) | | unchanged | 0 | 0 |
| `icons/heldout` | 25.537 | 25.538 | 0 | 0 |
| `broad/heldout` | 28.366 | 28.366 | 0 | 0 |

Single images move a lot (the CD +16.6, the steak +9.3, a `broad` image +5.2); the means little, because only flat
images at the `clean` severity can change, and pale colours this close to the background are rare. Held-out is flat.

### v25: thin features kept

The worst clean `icons` after v24 lost mostly **thin features**: a shoe's inner seam, a cat's whiskers, an extinguisher's
valve ring, a motorcycle's struts. `clean_labels` dissolves a component when under a quarter of its pixels survive a
1 px erosion. On a noisy image that removes speckle; on a noiseless one it also removes every 2 px line. It must still
remove the anti-aliased rim along every edge, which is just as thin.

**The rule:** on a noiseless image (σ at its floor), a thin component stays unless it is an edge's blend (`blended`):
at least half its pixels are a mix `t A + (1 - t) B` of two colours beside them, within 6 levels of sRGB or a fifth of
A-B's distance. A and B are each label's solid pixels (all of whose neighbours share their label) within 3 px. A seam has
a colour of its own, far off any such line; a rim is on it. Test: `a_thin_line_of_its_own_colour_stays_but_an_edges_blend_does_not`.

Three versions were measured, each a `copycat.report` against `copycat-pale-fills-2`:

| Report | The blend judged against | Share | `icons/train-control` | `synthetic/validate` | `icons/heldout` | Kept |
|---|---|---|---|---|---|---|
| `copycat-thin-features` | each label's mean colour | 80% | +0.10 | +0.01 (7 down; `synthetic/train` -0.05) | +0.27 | no |
| `copycat-thin-features-2` | the colours within 3 px | 80% | +0.17 | -0.01 (8 down) | +0.26 | no |
| `copycat-thin-features-3` | the colours within 3 px | 50% | **+0.17** | **+0.02** (1 down) | **+0.26** | yes |

- **Against a label's mean colour, shaded parts kept rims.** On `light` images a part's colour drifts across it (the
  shading), so its edge is a mix of its colour *there*, not of its mean. The extra shapes were faint rims on one side.
- **At 80%, blurred and compressed edges kept rims.** Blur and JPEG move many of a rim's pixels off the line; at half,
  none of the train-side rims stayed. A feature's pixels are almost none of them on the line (the shoe's seam is ~60
  levels off the green-white mix), so half leaves a wide margin.
- An earlier probe, keeping every thin component and skipping the majority filter on a near-noiseless image (σ under
  `FLAT_SIGMA`, so `light` too), lost 2.3 dB on `broad`'s clean images: rims everywhere.

Measured with `copycat-thin-features-3` (clean fidelity, dB):

| | before | after | up | down |
|---|---|---|---|---|
| `icons/train` | 27.499 | **27.571** | 1 | 0 |
| `icons/train-control` | 38.251 | **38.422** | 3 | 1 (-0.58) |
| `synthetic/validate` | 33.345 | **33.363** | 2 | 1 (-0.17) |
| `broad/train` | 28.798 | **28.818** | 7 | 1 (-0.15) |
| `synthetic/train`, `synthetic/heldout` | | unchanged | 0 | 0 |
| `icons/heldout` | 25.538 | **25.797** | 2 | 0 |
| `broad/heldout` | 28.366 | **28.382** | 2 | 0 |

The shoe gains 16.8 dB clean and 9.2 at `light`; a held-out icon 5.7. `copycat.clean.v2`: 29.54 to 29.63.

### v26: small enclosed parts kept

The worst clean `broad` image after v25 is a camera whose screen has a small pale sun (r 4) and a thin pale mountain.
Both are within `background_delta` of the white background, so `segment` gave them to it, and `enclose` left them holes:
it judges an enclosed region by its interior (pixels 2 px clear of its edge), and needs 16 of them. The sun has ~13, and
the mountain almost none.

**The rule:** on a noiseless image, a region with too little interior is judged by its *core* instead: its pixels off
the crisp edges (`core_stands_out`), which are its own colour, unblended. It needs 8 of them, and must be 3 robust
deviations from the background (`restore`'s margin for a colour that isn't its region's first). A gap between parts,
blurred by them, is all edge. Test: `a_small_pale_part_enclosed_is_a_fill_not_a_hole`.

Two versions were measured, each a `copycat.report` against `copycat-thin-features-3`:

| Report | Judged by the core on | `broad/train` | `synthetic/validate` | `icons/heldout` | `broad/heldout` | Kept |
|---|---|---|---|---|---|---|
| `copycat-small-enclosed` | any image with crisp edges | +0.06 (18 up, 5 down) | -0.02 (2 down: -5.0, -2.5) | +0.14 | +0.14 (1 down: -2.3) | no |
| `copycat-small-enclosed-2` | noiseless images only | **+0.03** (4 up, 0 down) | **0** | **+0.14** | **+0.14** (2 up, 0 down) | yes |

- **On `light` and `medium` images, gaps became fills.** There the crisp-edge map is coarser, so a gap between parts where
  the background shows through (between a synthetic image's shapes, between a DNA ladder's rungs) has a core, and blur and
  shading put it more than 3 deviations off the fitted background: a pale fill where the answer is nothing.
  Gated on σ at its floor, as v25 is, no image went down.

`icons/train`, `icons/train-control`, `synthetic/*` were unchanged. The camera gains 3.7 dB; others in `broad/train`
gain 5.9 and 3.9; one held-out `broad` image gains 14.6. `copycat.clean.v2`: 29.63 to 29.73.

The camera's other fault is still there: its handle and its lens cone share a grey, and `join` makes them one part
behind the dark connector, whose fitted shapes then fill the handle's hole. v27 is for that.

### v27: joins undone when the pieces say it far more briefly

`join` merges components of one colour when one shape describes them more briefly than one shape each: a part an
occluder cut in two. The camera's body-and-handle and its lens cone pass that test under the connector between them. But
the body and handle need two shapes, and `shapes_of` cuts a part only along its largest piece's outline, so the joined
part came out as a long curve from the handle round to the lens, filling the hole under the handle: 27.8 dB. With no
`join` at all the camera scores 39.0.

**The rule** (`joined_or_apart`): a part `join` made of pieces is fitted both ways, as one (`shapes_of` on the whole) and
apart (`shapes_of` on each piece, the other pieces free), each with what `leftovers` would add to it. Both are judged by
one description length over the whole part (`described`: outline bits, plus every part pixel left uncovered and every
covered pixel that is neither the part's nor free). It is drawn apart only if that costs under half the joined
description (`UNDO_JOIN`). Test: `a_join_whose_pieces_say_it_far_more_briefly_is_drawn_apart`.

Measured off the books first, every train-side image against `copycat-small-enclosed-2`:

| Variant | `icons/train` | `icons/train-control` | `broad/train` | Down |
|---|---|---|---|---|
| no `join` at all | -0.03 | -0.02 | -0.01 | 56 (36 up), to -11.5 dB |
| apart if each fit's own bits are fewer | +0.01 | -0.01 | +0.01 | 19, to -4.2 dB |
| **apart if under half the whole description** | **+0.05** | **+0.01** | **+0.02** | **0** |

- **`join` is right more often than not, but not by much.** With no join, 56 images went down and 36 up, both by up to 11 dB.
- **Without a margin, cheap loose fits won.** Fitted with the other pieces free, a piece can take a loose, cheap shape:
  the ATM's lower band became one large circle whose edge passes close to its rounded corners (1,011 bits apart, 1,092
  joined), -2.9 dB. The bad joins were far apart from the good ones: apart cost 0.27-0.43 of joined where it helped,
  0.60 and 0.93 where it hurt.

Measured with `copycat-joined-or-apart` (clean fidelity, dB): no image down anywhere. The camera gains 11.3 dB, a lit
`icons/train` image 5.3, four more 0.2-0.95; one held-out icon 0.28 (`icons/heldout` 25.938 to 25.950). `copycat.clean.v2`
stays at 29.73 (29.725 to 29.729).

### v28: thin features kept on near-noiseless images

After v27 the worst `light` images were far below the ~30 dB a `light` image usually gets. Three causes, one fixed:

- **A mosquito lost every leg (15.6 dB).** v25 keeps thin features only where σ is at its floor (0.002); this image's
  estimate was 0.00205. Most `light` images are at the floor, but about 15% are not. **The rule:** v25's test on a
  near-noiseless image, σ under `FLAT_SIGMA` (0.0025), the line other near-noiseless rules already use. `blended` still
  drops the edges' rims. Test: `a_thin_line_stays_on_a_lightly_noisy_image_too` (the shoe's seam, a level of noise on
  every pixel). v26's small-enclosed rule widened the same way changed nothing, so it stays at the floor.
- **An envelope lost its whole pale body (13.9 dB), not fixed.** Its σ, 0.00252, is just over `FLAT_SIGMA`, so the
  judged cut kept the border's wide spread about a gradient, and the body stood 2.6 deviations out where `restore` asks
  3. Moving the judged cut's gate to 0.003 got the envelope back (+17 dB), but medium images under 0.003 then restored
  their clutter as parts (`synthetic/validate` medium -0.55, one image -15.7); dropping the gate, worse. `light` and
  `medium` overlap in σ, and nothing else seen yet tells clutter from a pale part.
- **A printer's paper (16.7 dB) and a toilet roll's paper (21.1, clean), not fixed.** The paper is 1.8 deviations off a
  light background. The roll's covers a corner and much of the border, so the background is fitted as a gradient bending
  toward the paper (2 of 256 clean images get a gradient; `clean` backgrounds are always flat).

Measured with `copycat-thin-near-noiseless` against `copycat-joined-or-apart`: no image down. The mosquito gains 7.9 dB,
three `broad/train` images 0.4-1.3, a held-out `broad` image 3.6 (`broad/heldout` 28.519 to 28.549).
`copycat.clean.v2`: 29.73 to 29.74.

### v29: outlines whose knots carry their own tangents

docs/design/0025's breakdown put most clean error on outlines, at sub-pixel edges. Drawing each true shape of a clean image
alone and tracing it (off the books, 48 images a set) puts it on curved paths. On `icons` they had 1,019 px of squared
misfit in all, and every other kind of shape almost none. The shapes behind it are long, flat arcs beside a tight bend:
a dome, a shield. Their true cubics' handles are very uneven. A spline's tangent comes from its neighbouring knots, so
7 knots still miss a dome by 12 px. Schneider's fit from the marching-squares contour does worse: 89 segments at a
0.1 px tolerance still miss by 0.9 px. They were not corners: corner knots were tried first (`drawn_spline`, kept,
because they pay a little), but the dome's and the shield's joints are smooth.

**The fit** (`Target::smoothed`, after `tightened`): the part's final outline is redrawn with knots that carry their
own tangent, `[x, y, dx, dy, back]` (`hermite_path`: the outgoing handle's offset, and the incoming handle's length
along the same line). Five numbers a knot, so a dome needs 4 or 5 knots. They are settled by Levenberg-Marquardt on
every pixel's coverage, with a Jacobian by finite differences (a rasterization a number). Coordinate descent can't
follow a handle's length against its direction against the next knot's; its 0.1 px steps are also too coarse. A uniform
0.1 px offset along a 300 px outline is already 3 px of misfit. Then a knot is dropped while that is shorter, as for
splines. A path that crosses itself is no fit: free to bend, it would loop over a ring's hole to carve it out under the
non-zero rule, which is `with_holes`'s job. Polish moves the same five numbers per knot, with the same guard
(`Form::Hermite`).

**The price.** At 12 bits a wrong pixel, the shortest description of the shield is 3 knots and 3 px of misfit. On a
noiseless image that is the wrong trade: a pixel's misfit there is all signal, so the honest price is far higher. The
final outline is judged with a wrong pixel `OUTLINE_PRICE` times as dear when `outline_gain` is 0 (σ at its floor), and
only there, not in segmentation, joins or cuts (pricing the whole pipeline higher was tried before and added turning).
Tests: `a_domes_flat_arcs_are_said_by_knots_with_their_own_tangents`, `a_path_that_crosses_itself_is_caught`.

Tuned off the books, every train-side image against `copycat-thin-near-noiseless` (clean / all severities, dB):

| `OUTLINE_PRICE` | `icons/train` | `icons/train-control` | `broad/train` | `synthetic/validate` |
|---|---|---|---|---|
| 1 | +0.16 / +0.02 | +0.33 | +0.19 / +0.07 | +0.38 / +0.10 |
| 4 | +1.38 / +0.35 | +1.20 | +0.83 / +0.23 | +1.16 / +0.35 |
| 16 | +2.77 / +0.71 | +2.59 | +1.93 / +0.51 | +3.48 / +0.98 |
| **64** | **+3.32 / +0.85** | **+3.05** | **+2.16 / +0.57** | **+4.54 / +1.25** |

Measured with `copycat-hermite-outlines` (clean fidelity, held-out):

| Set | clean | light | all |
|---|---|---|---|
| `synthetic/heldout` | 52.76 to 56.07 | 35.56 to 35.80 | 34.719 to 35.599 |
| `icons/heldout` | 36.76 to 42.51 | 30.09 to 30.61 | 25.950 to 27.517 |
| `broad/heldout` | 41.98 to 44.72 | 29.71 to 30.04 | 28.549 to 29.294 |

`copycat.clean.v2`: 29.74 to 30.80, the largest step since polish (v22). `light` gains too: most `light` images are at
the noise floor. Turning stays near the true SVGs' (median ratio, clean: `icons` 0.71 to 0.94, `broad` 0.98 to 1.12)
and the tracer draws fewer nodes (`icons/heldout` 72 to 66 a image). It is about 20% slower (32 `icons/train` images:
5.4 to 6.5 s each). Not yet looked at: the worst losers, two `broad/train` images down 5.7 dB and one held-out 2.7.
On `icons`, the outline error left is about a third what it was, and most of the rest is parts never drawn: dots under
`min_area`, and rings `shapes_of` cuts in pieces.

## The page

`/games/copycat` is the shared game page (docs/design/0007) with a `GameModule` (`app/games/copycat.ts`):

- **The leaderboard** ranks by clean fidelity. Its columns are alpha IoU, layer recovery, and the shape, colour and
  turning ratios.
- **Watch** shows one held-out image at a time, from any of the protocol's sets: the entrant's SVG beside the input, the
  clean SVG and its shapes as layers pulled apart, with that image's scores, in one panel with its timelapse (0023),
  which plays on the same stage and ends on the scored SVG. The layers follow the frame on stage. An artwork image
  credits where its clean SVG came from.
  - Each entrant's SVG per image is stored in its evaluation record, so what you see is exactly what was scored.
  - `GET /leaderboard` strips the SVGs. `GET /games/copycat/targets/{set}/{seed}` serves one image with every entrant's
    output, and the static site copies both routes.
  - The timelapse is the entrant run again in the browser, so its own stages play on the stage (for the pipeline: the
    k-means palette, the cleaned regions, each part, the polish), and a badge says whether it ends on the SVG that was
    scored. While the pipeline runs, the stage shows its SVG so far with a progress bar and the time left: the core
    reports how far it is (`Film::report`) after each stage, each part (weighted by its pixels) and each polish pass,
    at fractions measured in WebAssembly, and the time left is the time so far scaled by what remains.
- **Which image** is one row under the stage: a sample of held-out images across every set and corruption level, the
  image on stage always among them, and a "more" pop-up with every held-out image by set and level. The row also adds
  your own image, which then stays in it whoever is on stage: traced on this device by the entrant on stage (the MDL
  tracer or a hill climb), at up to 192 px a side, offered as a download, never uploaded. It has no clean SVG, so no
  score. There is no separate Play view.

Two items from "The game, as a player sees it" are not built yet:

- comparing against VTracer on your own image, since VTracer has no WASM build here;
- rebuilding a target by hand.

## Open questions

- **Licences:** which icon sets (Twemoji needs attribution), and which AI-generated images can go in `wild` and be
  shown on the site.
- **Commercial comparisons:** whether Vectorizer.AI's and Recraft's terms allow publishing a comparison, and at what
  cost. As read through a summarizing tool on 2026-11-01, Vectorizer.AI's terms bar using results to benchmark AI or ML
  systems or to validate competing ones, and Recraft's restrict competing services. A person should read the live clauses and
  seek written consent before step 1's hand-run comparisons. LayerPeeler's vectorization step calls Recraft too.
- **Strokes in v2:** many icons are drawn with strokes, and a stroke is the natural thing to animate as a line drawing.
- **Gradient fills in v2:** some logos genuinely have them, and v1 would have to fake them with bands.
- **Text:** recovering a font is out of reach. Text will come out as filled shapes. Detecting that a region is text, so
  it can be flagged, may be worth it.
- **The code lengths themselves:** the precision per coordinate and the primitive costs are a prior. They will need
  tuning on the training split, and the tuning must never see the held-out split.

## Explicitly out of scope

- Photographs and painterly illustrations.
- Large learned models that write SVG code: pretrained, or trained elsewhere. A small model trained from scratch here
  (at most about 30M parameters) is [0024](0024-copycat-learned-inverse.md)'s.
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

# 0022 — Copy Cat: recover the clean, layered SVG behind a generated logo or icon

Status: **Proposed** (2026-10-24). Nothing is built yet.
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
| Entrants | `jobs/trainer`: `copycat.pipeline`, then its search and gradient stages; `rl.*` on `copycat` later |
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

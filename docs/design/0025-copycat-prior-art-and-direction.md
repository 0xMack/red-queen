# 0025 — Copy Cat: what the prior art says, and learning around the tracer

Status: **proposed** (2026-11-10). D0 found 10-15 dB of headroom, but D1's first runs (2026-11-11, off the books) showed
most of it is colour drift and warp, which a pixel denoiser doesn't undo. The part a denoiser can reach is 1-2 dB, and
the first model already takes most of it (see "D1, first runs"). Fitting can't undo the warp either; the headroom left
is structure on clean and light images ("Can fitting undo the warp?"), where the tracer's v24 came from. A breakdown of what is left (2026-11-11) found it is
outline precision, not missing parts: the outline fitter, Q1b, comes before Q2 ("What kind of structure error?"). A survey of image-to-SVG work, read against what [0024](0024-copycat-learned-inverse.md)'s
experiments showed, and the direction it sets: a learned denoiser in front of the MDL tracer first, then a svgformer that
sees what it has drawn. It replaces 0024's next steps (an L-scale run, then RL); 0024's model, tokens, data and runs stay.
Relates to: [0022](0022-copy-cat.md) (the task, the sets, `copycat.clean.v2`, the MDL tracer),
[0024](0024-copycat-learned-inverse.md) (svgformer, E0-E2), [0023](0023-process-traces-and-research-log.md) (reports as
runs), [0010](0010-reinforcement-learning.md) (policy gradients).

## Context

0024 trained svgformer, a 12M-parameter encoder-decoder that reads the corrupted image and writes the SVG token by token.

- **On `synthetic`, it works.** It learns the generator's images (26 dB greedy, 32.7 with the tracer's polish). Paired
  with the tracer per image, it is the first trained entrant ahead of the tracer on any set: +0.40 dB on `synthetic`.
- **On real artwork it doesn't** (E2). Its scores on `icons` and `broad` are 12-15 dB behind the tracer.
  - **Trained from scratch with artwork in the mix,** the decoder learns the icon sets as a language model and ignores the
    image. Its loss barely moves when given another sample's image (2.26 → 2.44).
  - **Fine-tuned from a model that reads images,** it keeps reading, but greedy decoding loops on long paths.
  - **Neither decoding guards nor best-of-8 sampling** close the gap.
- **The tracer's loss is mostly corruption.** On `synthetic/validate` it scores 47.3 dB on clean images, 35.8 light, 27.4
  medium and 22.9 heavy.

0024 named scale (the L model, 30M parameters, trained far longer) as the next lever. Before spending GPU-days on it, we
read the prior art (2026-11-10). Primary sources, main text; appendices not always. Most numbers are self-reported, the
papers disagree on metrics, and the 2026 preprints aren't reviewed.

## The survey

### Large learned image-to-SVG models

| Model | Size | Training data | What it does |
|---|---|---|---|
| [StarVector](https://arxiv.org/abs/2312.11556) (CVPR 2025) | 1B / 8B, pretrained CLIP or SigLIP + StarCoder | 2.1M SVGs from The Stack | SVG code left to right. 8B: 10 days on 64 H100s. Best of 5 by DINO at test time |
| [OmniSVG](https://arxiv.org/abs/2504.06263) | 3B / 7B, Qwen2.5-VL | 2M (icons, illustrations, characters) | One token per (x, y) point and per colour: illustrations 29.5k → 9.7k tokens |
| [VectorArk](https://arxiv.org/abs/2605.24398) (2026) | 1B, InternVL2 | 5M SVGs, 250k steps at batch 256 | Built for degraded inputs, around a classical tracer (below) |

- **On pixel metrics, optimisation still wins.** [LIVE](https://arxiv.org/abs/2206.04655) beats StarVector on SVG-Stack
  (MSE 0.003 against 0.026) and OmniSVG on icons (0.004 against 0.008); both papers say so. Learned models win on DINO,
  compactness (3-5k tokens against LIVE's 18k) and time (15-60 s against 170 s).
- **At that scale, supervised training alone still loops.** RLRF reports that SFT-only models "often hallucinate or loop" on
  long SVGs, and Render-in-the-Loop sees "degenerate repetition loops" without its verification step. So 0024's E2 loops
  are a known failure of supervised training, not a bug of ours.
- **Our scale is far smaller:** about 1.3M training images and 12M parameters, against 50-100 times the data and 100 times
  the parameters, on pretrained vision.

### Hybrids around a classical tracer

**VectorArk** is the closest to Copy Cat: real-world inputs, not clean renders.
- **Input:** a classical tracer (Illustrator's Image Trace, or VTracer) first turns the image into outlines, without
  colour; the model reads that.
- **Model:** it predicts geometry only, as rounded polygons (28-47% fewer tokens than OmniSVG).
- **Colour and paint order** are set afterwards by render-and-compare: each path's median colour over the pixels only it
  covers, then orderings of overlapping paths scored by MSE against the input (200-300 ms per image on one CPU core).
- **Training inputs** come from a round trip: clean SVG → render (random size, blur) → classical tracer → render. So the
  model learns to correct a tracer's mistakes.
- **Why:** the authors found that a vectoriser trained on clean renders "faithfully reproduces" a degraded input's flaws.
- **Results:** on hard icons, SSIM 0.857 against StarVector's 0.626 and OmniSVG's 0.518.

[SuperSVG](https://arxiv.org/abs/2406.09794) is a coarse learned stage (paths per superpixel) refined by optimisation, the
shape of 0024's model + polish.

### RL from render feedback

- **[RLRF](https://arxiv.org/abs/2505.20793)** (0024's source for GRPO).
  - **Rewards:** pixel L2, L2 on edges, DreamSim, and a length penalty.
  - **Gain:** on a 7B model, icon MSE from 8.60 to 1.03, ahead of LIVE's 2.22, with the loops corrected.
  - **SFT first is necessary:** RL alone reaches 14.2.
  - **Rollouts per image:** going from 8 to 64 kept helping.
  - **Cost:** about 3 days on 32 H100s.
- **[IR4RL](https://arxiv.org/abs/2609.34587)** (2026) renders after every drawing command, and rewards each command by how
  much it moved the render towards the target.
  - **Result:** much better than a reward on the final SVG alone (DINO 97.5 against 92.9), with SVGs 2-4 times shorter.
  - **Granularity:** finer is better; rendering after every command was best.
  - **Data:** 700 training images; 10k gave no further gain.
  - **Model:** OmniSVG-4B, LoRA, 2 days on one B200.

So RL does fix looping and length. But every source starts it from a large model that is already good.

### Seeing what has been drawn

Every small, from-scratch method found that works shows the model the canvas it has drawn so far:

- **[Ellis et al. 2018](https://arxiv.org/abs/1707.09627)** (graphics programs from hand drawings).
  - **Inputs:** a CNN reads the target and the render so far as two stacked channels, and proposes the next primitive.
  - **Search:** Sequential Monte Carlo, weighted by the render's distance to the target.
  - **Results:** neither the network nor the search alone was enough. It handled scenes twice the size of its training
    scenes, and matched 63% of 100 real hand drawings exactly.
  - **Cost:** about a day on one GPU, on 10^5 synthetic scenes.
- **[Tree diffusion](https://arxiv.org/abs/2405.20519)** (Kapur, Jenner and Russell, ICLR 2025).
  - **Inputs:** the target, the current render and their difference.
  - **Output:** a grammar-constrained *edit* to the program, not a program written left to right.
  - **Training:** on random mutations of random programs.
  - **Search:** beam search with a value network that predicts edit distance.
  - **Ablation:** removing the current render "drastically hindered performance".
- **[Paint Transformer](https://arxiv.org/abs/2108.03798)** (ICCV 2021).
  - **Output:** from (canvas, target) patches it predicts a *set* of strokes, DETR-style, coarse to fine. No sequence.
  - **Training:** only on its own synthetic strokes, in under 4 hours on one RTX 2080 Ti.
  - **Speed:** 0.3 s per 512 px image.
- **[Render-in-the-Loop](https://arxiv.org/abs/2604.20730)** (2026).
  - **Feedback:** after each `<path>`, the canvas is rendered and fed back in.
  - **Inference:** a step that doesn't change the canvas, or repeats the code before it, is rejected and resampled.
  - **Result:** icon MSE 0.094 → 0.027 against the same model trained without feedback.
- **[Write, Execute, Assess](https://arxiv.org/abs/1906.04604)** (NeurIPS 2019): a policy and a value function over partial
  programs, run in a REPL and searched by SMC.

### Shorter sequences

- **Fewer tokens per point:** OmniSVG gives each (x, y) one token (3.6 times shorter); VectorArk uses rounded polygons;
  [StrokeNUWA](https://arxiv.org/abs/2401.17093) compresses strokes into VQ tokens.
- **Simpler targets:** VectorArk and OmniSVG both normalise SVGs before training: a few path commands, no transforms.

### Optimisation

- **Recent renderers:** [Bézier Splatting](https://arxiv.org/abs/2503.16424) (NeurIPS 2025: 30-150 times DiffVG's speed)
  and [Vector Scaffolding](https://arxiv.org/abs/2605.11913) (2026: +0.5-1.4 dB PSNR over it) target natural photos with
  hundreds to thousands of curves, which isn't our regime.
- **The pattern they share is the polish's:** add shapes where the error is largest, then optimise. LIVE starts each new
  path at the largest connected region of error.
- **[Image Vectorization with Depth](https://arxiv.org/abs/2409.06648):** paint order from an occlusion energy, with
  occluded parts completed by inpainting. Relevant to layered targets.

## What it means for Copy Cat

1. **Left-to-right decoding at our scale has no record of working on real artwork.** The models that do work are 100
   times larger and still loop.
2. **The small methods that work see their canvas.** Seeing the residual (what is still undrawn) addresses both E2
   failures:
   - **ignoring the image:** the next shape is decided by the residual, so there is no shortcut through a language model of
     icon code;
   - **looping:** each step is checked against the render, and one that does nothing is rejected.

   It also turns a 400-token icon into about 20 short shape decisions.
3. **Where pixels are scored, an optimiser or a tracer is the incumbent,** and the best learned systems work with one:
   VectorArk around a classical tracer, SuperSVG and 0024's polish as refinement after a learned stage.
4. **The tracer's loss is mostly corruption** (47.3 dB clean against 22.9 heavy). That is a dense, per-pixel learning
   problem: no sequence, no looping, and no way to ignore the input. Every artwork icon is usable through its clean
   render, with no token limit. VectorArk's lesson, read the other way round, is to clean the input rather than teach the
   vectoriser its flaws. We found no paper that does exactly this for clip art, so it is our hypothesis.
5. **RL fixes looping and length, but from a good supervised model.** A per-shape reward (IR4RL) fits a render-in-the-loop
   model naturally: the reward is how much the shape reduced the residual.

## Decision 1: denoise, then trace

**The idea.** A learned image-to-image model maps the corrupted input to an estimate of the clean render, and the MDL
tracer traces that. Training pairs are free: any clean SVG, rendered, and corrupted by `corrupt.v1`.

- **Data:** `synthetic`'s generator and the 36k-icon artwork pool (`svgformer.data`), never `icons` or `broad`.
- **Target:**
  - the clean pipeline's image (`corrupt.v1` at severity `clean`: the subject on a flat off-white background; D0 showed a
    pure white background costs the tracer the subject's white parts), with an L1 loss and an edge-weighted term, since 60% of svgformer's error was on edge pixels
    (0024);
  - an arm instead predicts a flat-colour image, each pixel snapped to its region's colour, which is closer to what the
    tracer looks for.
- **Model:** a small convolutional U-Net, or svgformer's ViT encoder with a pixel head. A few million parameters, in
  PyTorch, in `libs/svgformer`, which already owns the stream.
- **What the tracer and the polish see** (arms):
  - the tracer traces the denoised image;
  - the polish then fits against either the denoised image or the original input.
- **The entrant:** `denoise+structured`, and denoise + the model-or-tracer pick. The model runs on the GPU in the entrant's
  spawned process, as svgformer's does.
- **The risk:** the denoiser may learn `corrupt.v1` itself, not the designs. So `corrupt.v2` (0024 Decision 3, not yet
  built) is built before anything is promoted, and the entrant must hold there.

## Decision 2: svgformer sees its canvas

**The idea.** Decode shape by shape, as in Ellis et al. and Render-in-the-Loop:
- **Inputs:** before each shape, the encoder reads the target, the canvas rendered so far and their difference (stacked
  channels).
- **Output:** the decoder writes one shape's tokens.
- **Rendering:** each shape is rendered by the games core (`games.copycat`) before the next is decided.

- **Training:** teacher-forced on shape prefixes.
  - **Example:** a random prefix of a clean SVG's shapes, rendered, with the next shape as the target. When the prefix
    already draws everything, the target is `END`.
  - **Cost:** rendering a prefix is cheap in Rust, and the stream stays a pure function of `(seed, i)`.
- **Decoding, render-and-verify:** a shape that doesn't reduce the residual is rejected and resampled. Decoding stops at
  `END`, or when no sample improves the canvas. That replaces `no_repeat`'s token rules with the render.
- **Encoding cost:** the target's patches are encoded once. Only the canvas changes, so each shape re-encodes the canvas
  (or the residual) alone, cross-attended with the target's patches.
- **Starting canvas:** the canvas needn't start empty. Starting from the tracer's render, the model adds what the tracer
  missed; this is the hybrid. Edits to existing shapes (tree diffusion) are a later step.
- **Shorter tokens are arms inside this experiment, not experiments of their own:**
  - one token per point on a coarser grid (the polish restores precision);
  - targets normalised to the tracer's trace of the clean render.
- **RL comes after this,** as 0024's E4 reshaped. The reward for a shape is its reduction of the residual (IR4RL), plus
  the final render's score, by GRPO.

## Decision 3: deferred

- **The L model (30M parameters) and the illustrations track** (0024's E6). The literature's answer to E2's failures is
  feedback and structure, not 2.5 times the parameters. Revisit once a model is within 3 dB of the tracer on `icons/train`.
- **RL on left-to-right svgformer** (0024's E4 as written). Done instead after Decision 2, on the canvas-conditioned model.

## Experiments

Same rules as 0024:
- decisions are made on the train-side groups (`synthetic/validate`, `icons/train` and `broad/train`), weighted as the
  leaderboard weights its sets;
- small checks run off the books first; big comparisons are registered runs;
- held-out images are scored only when an entrant is promoted.

| Exp | Question | Arms | Gate |
|---|---|---|---|
| **D0** headroom (off the books) | How much would a perfect denoiser give the tracer? | the tracer on the clean render vs on the corrupted input, per group and severity | the clean-input tracer is at least 2 dB ahead on `icons/train` and `broad/train`. Otherwise Decision 1 isn't worth building |
| **D1** denoiser | Does a learned denoiser lift the tracer? | target clean RGB vs flat colour; data synthetic only vs + artwork; the polish against the denoised image vs the input | ahead of tracer v23 on each train-side group, by at least the seed spread; holds on `corrupt.v2` |
| **R1** canvas-conditioned svgformer | Does seeing the canvas fix E2? | canvas + residual vs left-to-right (0024's M), both with artwork; render-and-verify on/off | within 3 dB of the tracer on `icons/train` and `broad/train`: E2's gate |
| **R2** starting from the tracer | Does the model add what the tracer missed? | empty canvas vs the tracer's render (or D1's) as the start | the pairing ahead of D1's entrant |
| **R3** RL, per shape | Does optimising the render pay? | per-shape residual reward + the final score vs the final score alone; group size | +0.5 dB over its own SFT start, with 0024's structure ratios within 10% |

### D0 results (off the books, 2026-11-10)

**The check:** the MDL tracer (v23) on 96 images per train-side group (24 per severity), twice:
- on the corrupted input;
- on what a perfect denoiser would give: the same clean SVG through `corrupt.v1` at severity `clean`, with the image's
  own seed. That is the subject on a flat off-white background, as clean-severity inputs are drawn.

Both are scored by `copycat.clean.v2`'s fidelity against the clean SVG.

| Group | All | Clean | Light | Medium | Heavy |
|---|---|---|---|---|---|
| `synthetic/validate` | 32.90 → **47.00** (+14.1) | 44.83 → 44.83 | 37.43 → 47.24 | 27.21 → 48.37 | 22.12 → 47.55 |
| `icons/train` | 27.26 → **37.37** (+10.1) | 38.16 → 38.16 | 31.02 → 38.47 | 22.33 → 36.83 | 17.53 → 36.02 |
| `broad/train` | 31.16 → **46.39** (+15.2) | 49.14 → 49.14 | 32.68 → 46.73 | 23.59 → 45.28 | 19.22 → 44.40 |

- **The gate passes by far:** +10 to +15 dB per group, +13.1 dB in the leaderboard's weighting (30.44 → 43.59). The gate
  asked for 2.
- **Corruption is most of the tracer's loss, and even light corruption costs 7-14 dB.**
  - **`synthetic` and `broad`:** with a clean input the tracer reaches 47.0 and 46.4, at or above 0024 E0's token round
    trip (47.3, 44.7; a different measure, so only a guide).
  - **`icons`:** it stops at 37.4, about 8 dB short of its 45.2. So on `icons` the tracer's own structure costs a lot
    beyond corruption. That is where Decision 2 (a model that adds what the tracer missed) has room after D1.
- **The clean rows are +0.00 by construction,** which checks that the oracle input is drawn exactly as the set's own clean
  images are.
- **This is a ceiling, not a forecast.** A perfect denoiser also undoes each part's colour drift, which 0022 measured as
  partly unrecoverable from one image (about 0.5 dB). A real denoiser will also blur edges and invent detail. But D1 has
  over ten times the room it needs to pay.
- **A first attempt composited the clean render over pure white,** and lost 3-5 dB on clean images: an icon's white parts
  merged with the background, and the tracer dropped them. A denoiser's output must keep a background the tracer can tell
  apart from the subject. So D1's target is the clean pipeline's own image, background included.
- **Read with care (see D1 below):** most of this ceiling is drift and warp, not noise.

### D1, first runs (off the books, 2026-11-11)

**What was built:** `svgformer.denoise`, a convolutional U-Net (48 channels at full resolution, 4 levels, 7.3M
parameters).
- **Training pairs** come from svgformer's own stream, with the artwork share at 0.5.
- **The target** is the clean subject as premultiplied RGBA, with no background. A clean image's off-white is drawn at
  random, so a model can't predict it.
- **For the tracer,** the output is drawn over the off-white farthest from the subject's colours (`background_for`).
  Through that compositing, the true RGBA scores 46.8 / 36.9 / 46.1, within 0.5 dB of D0, so the choice costs nothing.
- **Training:** 20k steps at batch 32, L1 loss with edge pixels weighted 5 times, about 20 minutes.

The tracer on the denoiser's output, per train-side group (96 images each):

| Group | All | Clean | Light | Medium | Heavy |
|---|---|---|---|---|---|
| `synthetic/validate` | 32.90 → 33.63 | 44.83 → 43.72 | 37.43 → 39.32 | 27.21 → 27.71 | 22.12 → 23.78 |
| `icons/train` | 27.26 → 26.29 | 38.16 → 32.80 | 31.02 → 31.13 | 22.33 → 22.57 | 17.53 → 18.67 |
| `broad/train` | 31.16 → 29.51 | 49.14 → 39.81 | 32.68 → 32.93 | 23.59 → 24.54 | 19.22 → 20.78 |

Weighted, 30.44 → 29.81 (−0.63). After 5k steps it was −1.95.

**Why the gain is so much smaller than D0's.** `corrupt.v1` applies its stages in order:
1. per-part colour drift;
2. shading;
3. a smooth warp, a sinusoidal displacement of up to 1.8 px;
4. blur;
5. the background, noise, grain and JPEG.

The tracer was run on each image's own corruption cut off after each stage, over the same background as above. These
are means over the 72 corrupted images per group:

| Group | Severity | Drift | + shading | + warp | + blur | Corrupted input | Denoised |
|---|---|---|---|---|---|---|---|
| `synthetic/validate` | light | 44.4 | 44.1 | 40.2 | 39.1 | 37.4 | 39.3 |
| | medium | 38.0 | 37.0 | 27.9 | 28.0 | 27.2 | 27.7 |
| | heavy | 31.4 | 31.1 | 23.3 | 23.4 | 22.1 | 23.8 |
| `icons/train` | light | 36.7 | 36.2 | 33.3 | 32.9 | 31.0 | 31.1 |
| | medium | 31.8 | 31.0 | 24.0 | 24.0 | 22.3 | 22.6 |
| | heavy | 27.0 | 25.7 | 20.1 | 19.9 | 17.5 | 18.7 |
| `broad/train` | light | 42.2 | 41.8 | 35.1 | 34.6 | 32.7 | 32.9 |
| | medium | 34.6 | 34.1 | 26.0 | 26.0 | 23.6 | 24.5 |
| | heavy | 28.6 | 27.6 | 21.1 | 20.8 | 19.2 | 20.8 |

- **Drift costs 6-17 dB of D0's headroom, and warp another 3-9.** Background, noise, grain and JPEG together cost 1-2.5
  dB, the gap between the "+ blur" and "corrupted input" columns.
- **The denoiser recovers that last part,** reaching or beating the "+ blur" column on medium and heavy. So it is already
  near the ceiling of what removing noise can give.
- **D0's headroom was mostly drift and warp.** Both are hard to undo from one image:
  - a drifted fill is a plausible fill;
  - a warped edge is a plausible edge, unless the model knows edges should be straight, circles round and colours from a
    palette.
  The 5k-to-20k gain was on clean and light images. On heavy it was 0.4 dB, so more steps aren't recovering warp or
  drift.
- **The denoiser loses on clean inputs** (−1 to −9 dB). It doesn't reproduce a clean image exactly, and the plain tracer
  is already near its ceiling there.
  - With a perfect gate (clean inputs go to the plain tracer, the rest through the denoiser), the weighted score would be
    31.13, +0.69 over tracer v23.
  - That's an upper bound: a real gate has to estimate severity from the input.

**What this means for the plan:**
- **Decision 1, as a pixel denoiser, is worth at most about +0.7 dB.** That is too little to justify the trainer
  algorithm, the entrant and `corrupt.v2` on its own. Its code stays in `svgformer.denoise`, unregistered.
- **The remaining headroom is geometric and chromatic priors:**
  - **warp:** straight edges, true circles and arcs. This is the tracer's job (0022), where fitted primitives would snap
    a wobbled edge back; or a model's, if it emits primitives rather than pixels;
  - **drift:** colours. A shared palette across an image's parts is a fair prior, but snapping to a given set's palette
    would fit `corrupt.v1`, not artwork.
- **Decision 2 now matters more:** a model that writes shapes, not pixels, is the learned way to bring those priors in.
- **Before either, measure the headroom first** (done below, "Can fitting undo the warp?").

### Can fitting undo the warp? (off the books, 2026-11-11)

0022 had already tried simpler shapes on noisy images (a cheaper wrong pixel; outlines made to beat primitives by 3-30x).
Neither paid, because "a simpler shape fitted to the warped input is still the wrong shape". So instead of a sweep, an
oracle: each image's own warp, alone (no drift, shading, blur or noise), and on it the true SVG, its structure and colours
exact, with only its numbers fitted (the tracer's polish). That is the best any fit could do.

| Group | Light: tracer / true + polish | Medium | Heavy | Clean (no warp) |
|---|---|---|---|---|
| `synthetic/validate` | 41.6 / 44.7 | 28.4 / 29.5 | 24.3 / 25.5 | 44.9 / 60.0 |
| `icons/train` | 34.8 / 40.6 | 25.2 / 27.1 | 21.4 / 22.7 | 36.6 / 59.9 |
| `broad/train` | 36.4 / 38.8 | 26.9 / 28.9 | 22.3 / 24.0 | 47.2 / 60.0 |

(60 dB is the score's ceiling. The clean column's inputs are drawn over `background_for`'s background, not the set's,
hence 36.6 on `icons` against 38.2 elsewhere.)

- **The warp can't be fitted away.** With the exact structure, fitting gains only 1-2 dB over the tracer on medium and
  heavy. Over a part, a 40-90 px sinusoid is nearly a shift, and a fit puts the part where it was shifted to. Only priors
  the pixels don't hold (alignment, symmetry, a grid) could undo it, and they would fit `corrupt.v1` as much as artwork.
- **The headroom left is structure on clean and light images:** 13-23 dB on clean, 2.4-5.8 on light. As shares of each
  group (a quarter of images are clean), about +5 to +8 dB a group if the structure were exact, most of it on clean images. That is the room for
  both the tracer and Decision 2.
- **Where the tracer loses on clean `icons`,** looking at the eight worst images:
  1. pale colours near the background lost or averaged (a CD's two wedge colours, a steak's rim);
  2. sub-pixel edge error along every outline;
  3. small details dropped (a pen's tip).
  The first was two bugs, now fixed as the tracer's v24 (0022, "v24: pale fills kept apart"): +0.30 dB on
  `icons/train-control`, held-out flat. The third, as v25 (0022, "v25: thin features kept"): thin components kept on
  noiseless images unless an edge's blend, +0.17 on `icons/train-control` and +0.26 on `icons/heldout`.

**What this means:** the learned direction (Decision 2) is still where most structure headroom is, but D1 and this test
change what it should aim at. Its gains will come on clean and light images, not by undoing corruption. So its
experiments should report clean and light separately, and R1's gate (within 3 dB of the tracer) should be read on them.
(Revised below: the headroom is mostly outline precision, not structure.)

### What kind of structure error? (off the books, 2026-11-11)

After the tracer's v24-v28 (0022), the leaderboard moved 29.54 to 29.74, and the clean losses left looked like one-image
cases. Before building Q2, this asks what the clean and light error *is*, since Q2's hybrid (R2: start from the tracer's
render and add what it missed) only helps if it is missing parts.

**The breakdown.** Every pixel is owned by the topmost shape covering it, in the true SVG and in the tracer's (v28); each
traced shape is matched to the true shape it mostly shows, and a true shape is *found* if its matches own half its
visible pixels. Each pixel's error then goes to one kind, and the score is recomputed with that kind removed:

| Group | Sev. | Fidelity | Sub-pixel edge | Edge ≥ ½ px off | Missing part | Colour | Wrong part |
|---|---|---|---|---|---|---|---|
| `icons/train-control` | clean | 38.44 | 57% (+4.5 dB) | 34% (+2.0) | 7% (+0.5) | 1% | 0 |
| `icons/train` | light | 31.33 | 40% (+2.4) | 36% (+2.0) | 9% (+0.7) | 14% (+0.9) | 0 |
| `broad/train` | clean | 42.79 | 58% (+4.2) | 28% (+1.7) | 8% (+0.7) | 1% | 0 |
| `broad/train` | light | 30.80 | 46% (+3.1) | 26% (+1.5) | 15% (+1.1) | 13% (+0.8) | 0 |
| `synthetic/validate` | clean | 47.38 | 63% (+4.4) | 27% (+1.6) | 2% (+0.1) | 0 | 0 |
| `synthetic/validate` | light | 35.83 | 53% (+3.5) | 36% (+2.1) | 1% | 11% (+0.6) | 0 |

"Sub-pixel" is within ~1.5 px of a true boundary where both maps agree which shape is there (only the anti-aliased
coverage differs); "≥ ½ px off" is the same band where they disagree. Colour on `light` is mostly the corruption's drift.

- **About 90% of clean error is along outlines; missing parts are 2-8%.** Typical clean icons are structurally right
  (same shapes, same fills) with a thin line of error along every outline.
- **The polish isn't the limit.** Polishing the tracer's output a second time gains 0.6-0.9 dB (48 clean images per
  group). Turning every straight segment into a cubic (same nodes, free to bend), then polishing, gains 0.05 more.
- **The tracer draws lines where the designs have curves**: curve share of path segments 0.49 against the true 0.69 on
  `icons`, 0.43 against 0.63 on `broad` (0.67 against 0.74 on `synthetic`, where it scores best), and twice as many
  circles and ellipses as the designs (262 against 113 on clean `icons`). Free to bend, the same nodes still don't fit: a
  node in the wrong place is a local optimum the polish can't leave.
- **The fitter itself is the limit.** An oracle with perfect segmentation (the tracer on each true shape alone, its whole
  outline visible on a contrasting background, assembled in the true order with the true fills, then polished) reaches
  only 40.9 on clean `icons` (the tracer 38.2) and 49.9 on `broad` (48.2). The true outlines, polished, are 60.

**What this means:**
- **The clean headroom is the outline fitter, 10-19 dB of it,** not segmentation, occlusion or missing parts (about 2 dB
  between them). Turning a region into an outline is a classical problem (corner detection, then curve fitting, as
  Potrace and Schneider's algorithm do it, and here with the anti-aliased edge's sub-pixel information too). The
  isolated-shape oracle is a fast benchmark for it: no segmentation, and the answer is known.
- **R2's premise is weak.** Adding what the tracer missed reaches the 2-8% that is missing parts: under 1 dB on clean
  images. A learned model's outlines go through the same polish, and its tokens alone round-trip clean `icons` at 45.2
  (0024 E0), so Q2 wouldn't beat a good fitter at outline precision either.
- **Estimated gain:** clean images are a quarter of each set. If a better fitter took clean images halfway from the
  tracer to the true outlines' 60, about +2 dB on the leaderboard, ten times what v24-v28 gained together; light images'
  outline error (70-90% of theirs) would add some.

## Phases

| Phase | What | Checked by |
|---|---|---|
| Q0 | this doc; D0 | review; D0 recorded here |
| Q1 | the denoiser: model, stream, trainer algorithm (`copycat.denoise`), entrant; `corrupt.v2` | off the books first; then D1 as registered runs; the leaderboard. **Stopped after the off-the-books runs:** the model and stream are built, and the rest isn't worth +0.7 dB (see "D1, first runs") |
| Q1b | the tracer's outline fitter: corners, then curves, from the anti-aliased edge ("What kind of structure error?") | the isolated-shape oracle off the books; then `copycat.report` runs as tracer versions (0022) |
| Q2 | canvas-conditioned svgformer: prefix stream, model, render-and-verify decoding. **After Q1b:** its R2 premise (adding missing parts) is worth under 1 dB on clean images | off the books first; then R1 and R2 |
| Q3 | GRPO with a per-shape reward (0024's `copycat.grpo`) | R3 |

0024's P6 (a model package and the page) applies to whichever entrant is promoted.

## Sources

- StarVector: <https://arxiv.org/abs/2312.11556>
- OmniSVG: <https://arxiv.org/abs/2504.06263>
- VectorArk: <https://arxiv.org/abs/2605.24398>
- SuperSVG: <https://arxiv.org/abs/2406.09794>
- RLRF: <https://arxiv.org/abs/2505.20793>
- IR4RL, RL from intermediate renders: <https://arxiv.org/abs/2609.34587>
- Render-in-the-Loop: <https://arxiv.org/abs/2604.20730>
- RefineSVG: <https://arxiv.org/abs/2607.27699>
- Ellis et al., graphics programs from hand-drawn images: <https://arxiv.org/abs/1707.09627>
- Write, Execute, Assess: <https://arxiv.org/abs/1906.04604>
- Tree diffusion: <https://arxiv.org/abs/2405.20519>
- Paint Transformer: <https://arxiv.org/abs/2108.03798>
- LIVE: <https://arxiv.org/abs/2206.04655>
- Bézier Splatting: <https://arxiv.org/abs/2503.16424>
- Vector Scaffolding: <https://arxiv.org/abs/2605.11913>
- Image Vectorization with Depth: <https://arxiv.org/abs/2409.06648>
- StrokeNUWA: <https://arxiv.org/abs/2401.17093>

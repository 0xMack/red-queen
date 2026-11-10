# 0024 — Copy Cat, learned: a small transformer that reads pixels and writes SVG, then RL against the rendered result

Status: **proposed** (2026-11-08); P1 built: PyTorch on the RTX 5080, the tokenizer and grammar, and E0 (see "E0
results"). Phase B in part: the artwork is downloaded, pinned, converted and measured (see "Phase B results");
`synthetic.v2` and `corrupt.v2` are not built yet. P2 built: the model, the stream, `copycat.svgformer` with a CPU golden
case (see "P2 results"); E1 run (see "E1 results": S passes the gate, 23.7 dB at 20k steps, 25.7 at 60k).
Relates to: [0022](0022-copy-cat.md) (the task, the sets, `copycat.clean.v2`, the MDL tracer this must beat),
[0023](0023-process-traces-and-research-log.md) (reports and experiments as runs), [0004](0004-small-transformer-from-scratch.md)
(the transformer built from scratch), [0009](0009-client-side-inference-at-scale.md) (model packages, ONNX Runtime Web),
[0010](0010-reinforcement-learning.md) (policy gradients, PPO), [0018](0018-workloads-api-and-cli.md) (the trainer, runs,
experiments).

## Context

### Where the MDL tracer stands

The MDL tracer (0022) is at v23: **29.54 dB** on `copycat.clean.v2` (synthetic 34.72, icons 25.54, broad 28.37), 13 dB ahead
of the best tracer. Its gains have come one hand-built rule at a time: 23 kept changes, each a few tenths of a dB on one set.
0022's own measurements after v23 say where the rest is, and why more rules won't reach it:

- **The tracer is fragile.** A small change to the input flips its decisions (a white cup, no longer brightened, lost to
  `background_delta`; a shaded part split into two colours past `merge_delta`). The same image traced with four k-means
  seeds varies enough that a perfect pick among six traces would gain about +0.45 dB everywhere.
- **Structure search is nearly exhausted.** Even judged by the clean answer, a hill climb over the tracer's SVGs gains at
  most +0.16 dB.
- **What one image can't reveal, a prior over many images can.** Colour drift moves each part's colour at random; about
  0.5 dB of it is unrecoverable from the image alone. 87-88% of `icons`' and `broad`'s clean colours occur in other icons'
  palettes. Likewise, what shapes designers draw, how parts occlude and what the corruption does are regularities across
  images, which a per-image tracer has no way to learn.

### The idea

**Learn the inverse directly.** Train a small encoder-decoder transformer on pairs of (corrupted image, clean SVG), generated
without limit and drawn from tens of thousands of openly licensed graphics, so it writes the SVG as a sequence of tokens.
Then fine-tune it by reinforcement learning, with a reward computed by rendering what it wrote, as in RLRF. Supervised
training teaches the grammar and the priors. RL optimizes the rendered result itself, which a per-token loss can't: a
coordinate one bin off costs the same loss as one fifty bins off.

The MDL tracer stays, as the baseline and as a partner: a learned model's SVG can be polished by the tracer's own
whole-image polish (0022 "Polish"), and several samples can be ranked by description length.

### What changes in 0022's scope

- **Learned models:** 0022 ruled out "large learned models that write SVG code". This doc keeps the spirit (no pretrained
  models, nothing trained elsewhere) and narrows it to size and provenance: a model of at most about 30M parameters,
  trained here, on this repo's generated data and on openly licensed artwork.
- **Graphics beyond icons:** 0022's sets are emoji and icons at 128 px. This doc adds **illustrations**: larger, proper
  images with many parts (Decision 3), at 256 px.

## Decision 1: the model is a small encoder-decoder transformer, in PyTorch to start

- **Encoder:** a vision transformer over the input image's patches (256 patch tokens: 8 x 8 patches at 128 px, 16 x 16 at
  256 px), each a linear projection plus a learned position, plus Fourier features of the patch's place in the token
  grid's frame (P2: see "P2 results").
- **Number tokens carry their value.** A coordinate's embedding adds Fourier features of its value (frequencies 1 to 2^9
  over the grid), and its output logit adds the decoder state's dot product with the same features. So "x = 412" and the
  patch it falls in share a code, and neighbouring values share most of theirs.
- **Decoder:** a causal transformer writing SVG tokens (Decision 2), with cross-attention to the encoder's patch tokens.
- **Sizes to start:**
  - **S:** d = 192, 4 + 4 layers, about 4M parameters;
  - **M:** d = 256, 6 + 6 layers, about 11M parameters;
  - **L** (illustrations): d = 384, 8 + 8 layers, about 30M parameters, the ceiling this doc allows.
- **PyTorch, to start.** It gets to measured results fastest, and on the RTX 5080 (16 GB) it has fused attention and bf16.
  The repo's own `libs/autodiff` (0004) would need a GPU backend and new ops first. Whether to port a winning model to our
  own autodiff later, for the from-scratch story, is a later decision.
- **A new package, `libs/svgformer`:**
  - the tokenizer, the grammar and the data stream, in plain Python with no torch dependency (so they test anywhere);
  - the model, training, decoding and GRPO, which need the `torch` extra.

## Decision 2: SVG as tokens, with a grammar every sample obeys

- **The vocabulary** (about 1,400 tokens):
  - shape types (`rect`, `circle`, `ellipse`, `polygon`, `path`);
  - path commands (`M`, `L`, `Q`, `C`, `Z`; a second `M` in a path starts a subpath, which is how holes are drawn);
  - **coordinates on a grid of 1/8 px at 128 px** (1/4 px at 256 px), over the canvas plus a margin of 1/8 of it on every
    side: 1,280 values at either size, one token per number, x and y separately (E0 chose it: 1/4 px alone cost
    `synthetic` its 45 dB ceiling);
  - **colours at 6 bits a channel** (64 values, three tokens per fill);
  - `END`.
- **The sequence is the SVG in paint order,** bottom first, one shape after another. The clean SVG's own order is the
  target, since order is part of the design.
- **Within a shape, geometry comes before paint:** type, numbers (and the path's commands), then the colour and opacity.
  Colour first made the model guess a colour before it had said where it was looking (P2: see "P2 results").
- **Grammar-constrained decoding.** A small state machine knows what may come next (after `circle`, three numbers and a
  colour; inside a path, a command or a number), and every other token's probability is set to zero. So every sample
  parses into the subset, and RL never has to learn the syntax.
- **A cap per track:** 1,024 tokens for icons, 4,096 for illustrations, to start. Longer targets are left out of training,
  and counted.
- **The tokenizer is checked by round trip:** SVG → tokens → SVG, rendered, against the original (experiment E0).

## Decision 3: the training data

**Generated, without limit:**
- **`synthetic`'s generator on seeds from 1,000,000 up**, disjoint from every split (train 0-95, held-out 10,000+,
  validate 20,000-20,399).
- **`synthetic.v2`:** a richer grammar meant to look like icons: holes, mirror symmetry, rotational repetition, rounded
  rectangles, nested parts, 2-40 parts. Like `synthetic`, it is generated, so it can live in the repo.
- **Corruption on the fly:** `corrupt.v1` at a random severity with a fresh seed for every sample, so the model never sees
  the same corrupted image twice.

**Artwork, as much as the licences allow.** Licences were checked at the source (2026-11-08). The main source is
**Iconify's icon sets** (`iconify/icon-sets`): one JSON file per set, and an index giving each set's licence as an SPDX id
and whether it is multi-colour.

| Track | Source | Licence | About |
|---|---|---|---|
| icons | Noto Emoji (Iconify `noto`, `noto-v1`) | Apache-2.0 | 5,900 |
| icons | Fluent Emoji Flat (`fluent-emoji-flat`) | MIT | 3,100 |
| icons | Emoji One (`emojione`), Firefox OS Emoji (`fxemoji`), Streamline Emojis (`streamline-emojis`) | CC-BY-4.0, Apache-2.0, CC-BY-4.0 | 3,700 |
| icons | Streamline colour sets (`streamline-color`, `-flex-`, `-sharp-`, `-plump-`, `-ultimate-`, `-freehand-`, `-kameleon-`, `-cyber-`, `-stickies-color`) | CC-BY-4.0 | 8,300 |
| icons | Fluent UI System Color, IconPark, Glyphs Poly, Marketeq, Meteocons, Iconito, Flat UI | MIT / Apache-2.0 | 5,700 |
| icons | the `withicons-*` sets (28 styles) | MIT | 28,000 |
| icons | flags: `flag`, `circle-flags`, `tabler-flags`, `flagpack`, `cif` | MIT / CC0 | 1,700 |
| icons | logos and app icons: `logos`, `thesvg-color`, `selfhst`, `token-branded`, `devicon`, `skill-icons`, `vscode-icons`, `material-icon-theme`, `catppuccin`, `cryptocurrency-color`, `gcp`, `k8s`, `unjs` | CC0 / MIT / CC-BY-4.0 / Apache-2.0 | 23,000 |
| illustrations | **Openclipart** (CC0 since its start; the Debian `openclipart-svg` package or the site's archive) | CC0 / public domain | tens of thousands |

So **about 80,000 icons** and an illustrations track, before files the subset can't hold are refused (measured in "Phase B
results": 53,069 icons and 2,491 illustrations convert). Most illustrations
use strokes or gradients, outside the v1 subset, so their refusal rate is measured first (E0).

- **Excluded, and why:**
  - **unDraw:** its licence explicitly forbids training models on its illustrations and putting them in datasets.
  - **OpenMoji, Emoji One v1** (CC-BY-SA-4.0) and **MMSVG** (OmniSVG's 255k illustrations, CC-BY-NC-SA-4.0): share-alike
    and non-commercial terms on a model that is published on the site are not settled. Opt-in later only, if you decide so.
  - **StarVector's `svg-stack`:** 2.3M SVGs, but no licence is stated for the files.
  - **Twemoji and Flat Color Icons** (`twemoji`, `flat-color-icons`): the test sets `icons` and `broad` are drawn from
    them, so training on them would test the model on its own training data.
- **Attribution:** CC-BY sets are credited in a checked-in `ATTRIBUTION` file listing each set, its author and licence.
- **Logos are trademarks.** Their files are openly licensed, so they are used for training, but never shown on the site.
- **How it is stored:** as with 0022's sets, artwork is downloaded on first use, pinned by digest in a sources file,
  converted to the subset, and never checked in. Refused files are counted per set.
- **Deduplication:** every training SVG's canonical digest is checked against every held-out image's clean SVG.
- **`icons/train` and `broad/train` are used only by a final fine-tune arm** (E5), labelled as such. Held-out images are
  never trained on.

**New test sets:**
- **`illustrations`:** a held-out split of Openclipart, never trained on, corrupted like the others. It joins
  `copycat.report`'s groups first, and a leaderboard protocol (`copycat.clean.v3`) only once an entrant handles it.
- **`corrupt.v2`:** a second corruption family (a spline warp plus small rotation and scale, a different blur kernel), used
  only for testing. A model trained on `corrupt.v1` that has learned the corruption itself, rather than the design, shows
  it there.

## Decision 4: compute and determinism

- **The development machine** has an NVIDIA RTX 5080 (16 GB, Blackwell, compute capability 12.0) and 32 CPU cores.
  Blackwell needs a PyTorch build for CUDA 12.8 or later.
- **PyTorch is an optional extra** (`svgformer[torch]`), resolved from PyTorch's CUDA index on Windows and its CPU index on
  Linux, so `uv sync --all-packages` doesn't pull 3 GB of CUDA into every environment. CI runs the tokenizer and grammar
  tests without it, and the model tests on CPU torch.
- **Training runs bf16 autocast on the GPU** once fp32 has been checked against it (E1).
- **Determinism:**
  - On the CPU with `torch.use_deterministic_algorithms(True)`, a tiny model's training is bit-reproducible on one
    machine, and a golden case pins it (as `jobs/tests/test_golden.py` pins every training job).
  - GPU runs are not bit-reproducible. Each records its hardware fingerprint and seeds; its results are statements about
    runs, not bytes.

## Decision 5: RL against the rendered result

After supervised training (SFT), the model is fine-tuned by **GRPO-style policy gradients** (group-relative, as in
DeepSeekMath):

- **For each image, sample a group of G SVGs** (G = 8 to start; constrained decoding, temperature 1). Each SVG's reward
  is computed by rendering it. Its advantage is its reward minus the group's mean, over the group's spread.
- **The loss** is PPO's clipped ratio objective on each sampled token, weighted by its SVG's advantage, plus a KL penalty
  to the frozen SFT model, so the model doesn't drift into SVGs the reward can't tell apart. No value network.
- **Three rewards, as experiment arms:**
  - **−L, the MDL description length** of the SVG against the *input*, under the tracer's own background model, noise
    estimate and `Costs`. It needs no clean answer, so it works on any image, including real logos. It is built natively
    in the games core: this is the "one lead left" in 0022's "After v23", useful to the tracer too.
  - **Clean fidelity** where the clean answer is known (all training data has one). It is the leaderboard's own measure,
    so it is the most direct reward, and the most likely to be gamed.
  - **A mix:** clean fidelity, plus −L scaled down.
- **Watching for reward hacking:** shape, palette and smoothness ratios (0022 Decision 3) are logged every generation.
  A reward that rises while they drift past 10% is a hack, not progress.
- **Rendering stays in Rust** (`games.copycat`), called from the training loop. The GRPO loss is checked against a
  plain reference on a tiny model, as `libs/rl` checks its losses against `reference_pg.py`.

## Decision 6: decoding, entrants and the browser

**Three ways to decode, each its own leaderboard entrant** (`run:<id>@<variant>`, as published Snake variants already are):
- `greedy`: one SVG, the most likely token at each step;
- `best8`: 8 samples, the one with the shortest description L kept (label-free);
- `best8+polish`: that sample, then the MDL tracer's whole-image polish (0022).

**In the browser:**
- A model package (0009): the encoder and a decoder step with a KV cache, exported with `torch.onnx` by a new exporter in
  `libs/modelpack`.
- Run by ONNX Runtime Web, WebGPU first, on the Copy Cat page's Play stage, beside the MDL tracer.
- Its parity with the PyTorch model is measured: agreement of greedy tokens on held-out images.

## Decision 7: runs and experiments

- **Two trainer algorithms:**
  - **`copycat.svgformer`** (SFT, `paradigm: supervised`, like `tinylm`);
  - **`copycat.grpo`** (RL, `paradigm: reinforcement_learning`), which continues an SFT run through `init_from`, as
    `td_lambda` does.
- **A generation is a fixed number of optimizer steps.** It records:
  - loss and token accuracy;
  - a held-out curve every k generations on `synthetic/validate` and on a validate slice of the training artwork (clean
    fidelity, by `arena.copycat`'s scoring);
  - for `copycat.grpo`, the reward, the KL and the structure ratios.
- **Training cost** is measured (`arena.costs`: GPU time, peak memory) and recorded with the run. Checkpoints are
  artifacts, and the champion is the checkpoint best on validate.
- **The leaderboard:** `copycat.clean.v2` scores finished `copycat` runs that aren't experiment runs, each decoding
  variant as its own entrant. Experiment runs stay off the leaderboard; a winning arm is promoted as a plain run.
- **Experiments** are `experiments/copycat-learned-vN.yaml`: arms x seeds. An `experiment.copycat-report` compares arms
  image by image on the `train`-side groups `copycat.report` uses (a paired permutation test, as `.rl-report` does), and
  always includes the MDL tracer's current report as a reference arm.

## Off the books first

Unregistered scripts and unit tests, for finding bugs, never for deciding anything. Nothing is registered as a run until
these pass:

1. **PyTorch sees the GPU:** CUDA available, a bf16 matmul on the RTX 5080, a forward and backward pass that agree with
   the CPU's to fp32 tolerance.
2. **The tokenizer round-trips,** and the constrained decoder never emits an SVG that fails to parse (fuzzed with random
   logits).
3. **The model can learn at all:**
   - it drives one batch's loss to near zero;
   - a tiny model learns single-circle images;
   - the attention patterns look sane.
4. **The GRPO loss matches its reference** on a tiny model.
5. **Throughput:** steps per second on the GPU for S, M and L. This sets every later experiment's budget, and replaces the
   estimates below.

## Experiments

Decisions are made on `train`-side groups (`synthetic/validate`, a validate slice of the training artwork, and `icons/train`
and `broad/train` as transfer sets the model wasn't trained on until E5). Held-out images are scored only when an entrant
is promoted. Each experiment has a gate: what it must show for the next one to be worth running.

| Exp | Question | Arms | Gate |
|---|---|---|---|
| **E0** ceilings | What do the tokens cost before any learning? | clean SVGs tokenized and detokenized, per set and track; target lengths; refusal rates | round trip ≥ 45 dB on `synthetic` and ≥ 40 on the artwork; ≥ 95% of icons under 1,024 tokens. For illustrations it decides the track: their length and refusal rate say whether 4,096 tokens and the v1 subset can hold them |
| **E1** can it learn? | SFT on `synthetic` alone | S vs M; fp32 vs bf16; a data-budget curve | `synthetic/validate` beats VTracer (about 20 dB) clearly. Reaching the tracer (about 34) is the stretch |
| **E2** does it transfer? | Add `synthetic.v2` and the artwork | data mixes (generated only; plus emoji; plus every icon set) | on `icons/train` and `broad/train`, never trained on: within 3 dB of the tracer; on `corrupt.v2`, no collapse. Otherwise the priors learned don't transfer, and RL won't fix that |
| **E3** decoding | Is the model a better *starting point* than the tracer? | `greedy`, `best8`, `best8+polish` | `best8+polish` beats tracer v23 on at least one set: the first leaderboard promotion |
| **E4** RL | Does optimizing the render pay? | reward −L, clean fidelity, the mix; G 4 vs 8; KL weight | +0.5 dB or more over its own SFT start on validate, with structure ratios within 10% |
| **E5** in-family fine-tune | What do `icons/train` and `broad/train` add? | none vs a short fine-tune | reported separately and labelled: it learns those families' palettes, which (0022 notes) wouldn't help a logo from elsewhere |
| **E6** illustrations | Does it scale to proper images? | L at 256 px, 4,096 tokens, on the illustrations track | on `illustrations`: ahead of the MDL tracer there |

### E0 results (2026-11-08)

`python -m svgformer.ceilings`: every clean SVG tokenized, decoded and scored against itself as `copycat.clean.v2` scores
an entrant (`synthetic` is its 96 `train` and 400 `validate` seeds; `icons` and `broad` every clean SVG):

| Set | Images | Ceiling (dB) | Worst | Median tokens | 95th percentile | Longest | Under 1,024 |
|---|---|---|---|---|---|---|---|
| `synthetic` | 496 | **47.3** | 41.0 | 98 | 200 | 267 | 100% |
| `icons` | 152 | **45.2** | 41.5 | 322 | 889 | 1,510 | 96.7% |
| `broad` | 616 | **44.7** | 33.6 | 214 | 770 | 2,525 | 97.7% |

- **Every gate passes** at the 1/8 px grid. No number fell outside the grid.
- **The grid was chosen here.** At 1/4 px (640 values), the ceilings were 43.4 / 42.0 / 42.8, short of `synthetic`'s 45.
  Quantizing one thing at a time said why: coordinates at 1/4 px alone gave 44.7 / 44.4 / 45.9, at 1/8 px 50.8 / 50.7 /
  49.6; colours at 6 bits alone gave 52.8 / 48.7 / 50.1. So coordinates were the cost, and doubling the grid costs no
  sequence length.
- **Every one of the 768 artwork SVGs tokenizes** once empty subpaths (`M x y Z`, which draw nothing; one Twemoji has
  them) are dropped. The grammar accepts every tokenized SVG, and 3,000 random walks through the grammar all decode to
  SVGs the subset parses.
- **Illustrations are measured in phase B**, below.

### Phase B results (2026-11-08)

`svgformer.artwork` downloads 66 pinned files: the 65 Iconify sets at commit `3eb3338` (235 MB) and Openclipart 0.18
from Debian (86 MB, 8,124 SVGs). `svgformer.convert` turns each SVG into the subset exactly or refuses it with a reason.
It handles any viewBox, transforms, inherited styles, `<use>`, Illustrator's `<switch>`, rounded rects, and opacity on a
shape or on a one-shape group. It makes even-odd holes non-zero by orienting each subpath by its nesting depth, and drops
shapes drawn wholly off the canvas.

**Strokes become filled outlines** (`svgformer.stroke`), a second shape painted after the fill, as SVG paints them. Each
outline is built in user space, so a transform scales the stroke as SVG does, and in two forms:
- **the reference:** a quad per flattened segment, plus wedges, joins and caps, all wound alike so non-zero fills their
  union;
- **the compact outline,** what a person would write: each side offset as cubics (split until each piece turns at most
  22.5 degrees), the inner side of a turn meeting where the offsets cross or pivoting through the vertex, and joins and
  caps as arcs.

The compact outline is kept only if it renders within 3% of the reference's area; failing that, the file is refused.
- **The reference is rendered at 4x and averaged down.** The rasterizer accumulates signed area, so pieces that overlap
  at an anti-aliased edge count twice there. At 1x a union of round-join circles came out about 7% fat, and every compact
  outline of a dense polyline failed against it, wrongly.
- Of 52,506 strokes, 52,102 have a curve outline and 404 a flattened one. 2,612 files are refused because neither form
  is close enough, 1,816 of them from Openclipart. 300 more have a dashed stroke.

`svgformer.build` writes three files:
- `icons.jsonl` and `illustrations.jsonl` for training;
- `illustrations-heldout.jsonl`, held out by author: 11 authors' 432 images.

It drops exact duplicates (998) and anything whose 16 x 16 thumbnail matches a held-out image of `synthetic`, `icons` or
`broad` (11). Each record carries its token length, clamped count and stroke counts, so the training stream filters without
tokenizing.

| | Files | Kept | Under 1,024 tokens, nothing clamped | Under 4,096 | Ceiling (dB) | Median tokens |
|---|---|---|---|---|---|---|
| Iconify icons (128 px) | 78,566 | **53,069** (68%) | 35,676 | 51,455 | 45.9 | 591 |
| Openclipart, training (256 px) | 8,124 | 2,059 | 1,275 | 1,852 | 43.8 | 658 |
| Openclipart, `illustrations` held out | (same) | 432 | 179 | 342 | 43.5 | 1,600 |

Before strokes were outlined, the same sources gave 37,819 icons and 936 illustrations. 29,271 icons fitted 1,024 tokens.
- **Strokes added 15,236 icons and 1,555 illustrations,** at longer sequences: the icons' median length went from 398 to
  591 tokens.
- **Gradients are now most of what is refused:** 18,896 Iconify files and 2,290 Openclipart files. Next come clip paths
  (3,627) and masks (781). 12 sets still give nothing: `fluent-color`, `circle-flags` (all masks), and ten `withicons-*`
  styles that are all gradients.
- **Gradients stay refused.** The subset has none, and a flat stand-in would teach the model the wrong image.
- **So the training artwork is about 36,000 icons and 1,300 illustrations that fit the S/M context,** and about 51,500
  and 1,850 at 4,096 tokens. That is roughly 60 times `broad`'s 616.
- **Illustrations are long.** The held-out ones have a median of 1,600 tokens, and only 41% fit 1,024, so the
  illustrations track (P5) needs the L model's 4,096 context, as planned.
- **The held-out illustrations aren't pinned yet.** Membership follows what converts, since whole authors are taken in a
  seeded order until there are 240 images. It changed when strokes were added, from 6 authors to 11. It is pinned by
  digest, like `icons` and `broad`, before anything is scored on it (P5).

### P2 results, off the books (2026-11-08)

Unregistered checks, for bugs and settings, not decisions:
- **The checks pass:** one batch memorised (token accuracy above 0.95, greedy decoding reproducing it), the cached decoder
  step equal to teacher forcing, random weights still decoding to SVGs that parse, and bf16 within 0.01 of fp32's logits.
- **Throughput is data-bound.** On the GPU alone S trains 55 steps/s and M 31 (batch 64). With images made and corrupted
  on the fly, 24 workers deliver about 25 steps/s (1,600 images/s), after `corrupt_pixels` dropped the PNG round trip
  and each worker got one thread. So S and M cost about the same per step here: 20,000 steps is about 15 minutes.
- **The first model ignored the image.** Its loss on number tokens was the same with the batch's images shuffled (4.29
  real vs 4.32 shuffled): it learned the generator's prior, not the picture. An empty SVG scores 11.78 dB on
  `synthetic/validate`, and it reached 11.85. Three changes made it look:
  - **Fourier features for coordinates and patch places** (Decision 1): 11.85 → 13.12 dB at 3,000 steps;
  - **geometry before paint** (Decision 2);
  - **learning rate 1e-3, not 3e-4:** 13.17 → 18.50 dB at 6,000 steps. Now the number loss is 3.23 with the real images
    and 9.14 with them shuffled.
- **Soft coordinate labels didn't help:** a Gaussian over neighbouring grid values, 2 steps wide, scored 17.55 dB against
  18.50 for exact labels (`number_sigma`, off by default).

The registered comparison is E1, `experiments/copycat-learned-v1.yaml`: S, M, S in fp32, and S at 5,000 and 60,000 steps,
2 seeds each.

### E1 results (2026-11-08)

`copycat-learned-v1`, 10 runs (seeds 1 and 2), each scored by its best checkpoint on `synthetic/validate` (400 images,
greedy decoding). The empty SVG scores 11.78 dB.

| Arm | Seed 1 | Seed 2 | Mean | vs `s` | Layer recovery | Minutes |
|---|---|---|---|---|---|---|
| `s` (S, bf16, 20k steps) | 23.76 | 23.62 | **23.69** | | 0.84 | 14-17 |
| `s-fp32` | 24.64 | 24.35 | 24.50 | +0.81 | 0.86 | 19 |
| `s-60k` | 25.90 | 25.49 | **25.69** | +2.00 | 0.89 | 42 |
| `s-5k` | 14.32 | 16.53 | 15.43 | -8.26 | 0.47 | 4 |
| `m` (M, bf16, 20k steps) | 19.23 | 16.79 | 18.01 | -5.68 | 0.64 | 19-21 |

- **The gate is passed:** S beats VTracer's about 20 dB by 3.7 dB at 20,000 steps, and by 5.7 dB at 60,000. It scores 24.5
  dB on clean images and 22.0 on heavily corrupted ones. The stretch (the tracer's about 34 dB) is 8 dB away.
- **S's seeds agree** (0.14 dB apart at 20k, curves within 0.3 dB all the way), so the gaps below are well above seed noise.
  Two seeds give no confidence interval worth the name; these are measured gaps, not tested ones.
- **More steps pay, with diminishing returns:** +2.0 dB from 20k to 60k steps, but most of it came by 40k (25.5), and the last
  20k added about 0.3 dB. Steps alone won't close the 8 dB.
- **fp32 is 0.8 dB ahead of bf16** on both seeds, at 1.2-1.3 times the time. bf16 stays the default for now (Decision 4):
  the gap is smaller than more steps buy for the same time.
- **M lost to S, and not because it is bigger.** On both seeds it sat at the empty SVG's score for 6,000-8,000 steps (seed 2:
  12.6 dB at 8k), then climbed at about S's rate, so it is behind because it started late. Its scores were the same at every
  corruption level, so in those steps it was drawing the generator's average picture, not reading the image. Its
  training loss fell smoothly all the while, with no spike, so instability didn't look like the cause. It was, though:
  see the next section.

### E1, continued: tuning M (`copycat-learned-v2`, 2026-11-09)

Each arm differs from v1's `m` (18.01 dB) in one setting, at 20,000 steps:

| Arm | Seed 1 | Seed 2 | Mean | vs v1 `m` | vs `s` (20k) | Layer recovery |
|---|---|---|---|---|---|---|
| `m-warm4k` (warmup 4,000) | 25.75 | 25.93 | **25.84** | +7.8 | +2.15 | 0.89 |
| `m-lr5e-4` | 24.31 | 24.46 | 24.39 | +6.4 | +0.70 | 0.86 |
| `m-lr2e-3` | 15.48 | 11.81 | 13.65 | -4.4 | -10.0 | 0.35 |

- **A gentler start fixes M.** With 4,000 warmup steps it doesn't stall, and at 20,000 steps it matches S at 60,000 (25.69
  dB) in a third of the steps. A lower learning rate throughout helps less, since it slows the rest of the run too.
- **A higher learning rate makes the stall worse:** seed 2 never left the empty SVG's score. So the stall was the learning
  rate after all, even without a loss spike. The loss was falling on the parts of the sequence the image doesn't decide.
- **So the size question has an answer:** M is better than S, at the right warmup. Warmup 4,000 is M's default from here.

### Where the error is, and the polish (off the books, 2026-11-09)

Measured on `m-warm4k`'s champions (about 25.9 dB), on all 400 `synthetic/validate` images.

- **Colours are nearly solved; coordinates are the gap.** Teacher-forced, with the clean SVG's structure: the model's colours
  with clean coordinates score 40.5 dB, the model's coordinates with clean colours 23.1 (the tokenizer's ceiling is 47.4).
- **But teacher forcing misleads here.** Its coordinate misses (more than 1 px off) are nearly all a shape's *first*
  number: 75% of a path's `M`s, 64% of a polygon's first point, a rect's or ellipse's x far more than its y. Inside a
  shape, coordinates are good (a curve's end points: 0.6% missed). Given the clean prefix, several shapes could validly come
  next, and the model starts a different one. So free decoding (25.9) scores *above* teacher forcing (23.0).
- **In free decoding the error is at the edges:** 60% of the squared error is on the clean render's edge pixels and 11%
  within 1 px of them (12% of the pixels together). Only 20% is in whole regions (a missing, extra or misordered shape, or
  a wrong colour). The worst 10% of images hold 40% of the error.
- **Best-of-N doesn't pay.** Of greedy plus 7 samples, the best by its true score (an oracle) is +1.4 dB at temperature
  0.5, +0.7 at 1.0; the most probable by the model's own likelihood is +0.0. The model's errors aren't sampling noise.
- **The tracer's polish does.** `games.copycat.polish` (new: `copycat_trace::polish_document`) nudges any SVG's numbers
  against the input over the background `trace` fits, structure and colours fixed, in about 85 ms an image:

  | Severity | Greedy | + polish | MDL tracer (same images) |
  |---|---|---|---|
  | all | 25.93 | **32.26** (+6.33) | 33.34 |
  | clean | 27.35 | 40.73 | 47.31 |
  | light | 27.23 | 35.12 | 35.83 |
  | medium | 25.62 | **28.67** | 27.35 |
  | heavy | 23.51 | **24.49** | 22.89 |

  So the model with the polish is 1.1 dB behind the tracer overall, and ahead of it on medium and heavy corruption, where
  its prior matters. Its loss is on clean images, where the tracer's fitting nearly reaches the ceiling. On heavy
  corruption the polish fits the corruption too: 23 of those 100 images got worse.

### The fp32 output head (`copycat-learned-v3`, 2026-11-09)

`m-warm4k` with only the output logits in fp32 under bf16 autocast (`fp32_head`), 20,000 steps:

| | Seed 1 | Seed 2 | Mean | + polish (mean) |
|---|---|---|---|---|
| `m-warm4k` | 25.75 | 25.93 | 25.84 | 32.57 (32.88, 32.26) |
| `m-warm4k-fp32head` | 26.05 | 26.23 | **26.14** (+0.30) | 32.70 (32.74, 32.67) |

- **+0.30 dB on both seeds, at the same speed:** about 40% of what training wholly in fp32 bought S (0.8 dB). The rest of
  that gap is elsewhere in the network.
- **After the polish it is +0.13,** inside the baseline's own seed spread (0.6 dB after polish). The polish fixes coordinate
  precision, which is what the head was buying. It costs nothing, so `fp32_head` is on from here, but it isn't a lever.

### The model or the tracer, per image (off the books, 2026-11-09)

The polished model wins on medium and heavy images, while the tracer wins on clean and light ones. So the question is
whether, per image, the better of the two can be told from the input alone. Both SVGs are composited over the background
the tracer fits (`games.copycat.background`, new), and their misfit is the mean squared error against the input.
`m-warm4k-fp32head`, greedy + polish, on the 400 `synthetic/validate` images:

| Per image | All | Clean | Light | Medium | Heavy |
|---|---|---|---|---|---|
| model + polish | 32.74 / 32.67 | 42.40 / 41.81 | 35.36 / 35.86 | 28.83 / 28.64 | 24.37 / 24.38 |
| MDL tracer | 33.34 | 47.31 | 35.83 | 27.35 | 22.89 |
| the better fit | 33.86 / 33.81 | 48.55 / 48.46 | 36.45 / 36.35 | 27.50 / 27.50 | 22.94 / 22.92 |
| **the tracer only if it fits 2x better** | **34.42 / 34.33** | 48.36 / 48.23 | 36.07 / 36.15 | 28.78 / 28.50 | 24.47 / 24.44 |
| the better, known afterwards (oracle) | 34.74 / 34.68 | 48.57 / 48.48 | 36.75 / 36.73 | 29.08 / 28.91 | 24.56 / 24.59 |

Each cell is seed 1 / seed 2.

- **The better fit is the wrong judge under corruption.** The tracer draws what corruption adds (colour drift, the shading
  gradient, clutter), so it fits a corrupted input more closely and is picked. That holds for 90 of 100 medium images and
  96 of 100 heavy ones, though the model is the closer to the clean answer on 72-77 of the medium and 82 of the heavy.
- **Two things that should have told the cases apart didn't:**
  - comparing blurred renders;
  - a noise estimate (Immerkær's). JPEG flattens the grain, and the estimate is about the same at every severity.
- **A margin does:** take the tracer only when its squared error is under half the model's. The factor was tuned on half
  the images and tested on the other half. It came out at 1.96-2.11 on both seeds and both halves, and the cross-checked
  score matches the full-set one.
  - This gives +1.0 dB over the tracer, and gets 0.3 dB short of the oracle. It is the best synthetic score so far.
  - The tuning is on validate, which is what validate is for. Held-out hasn't been looked at.
  - The cost is both pipelines: the tracer's time plus a decode and a polish.
- **What's left in the oracle gap is mostly clean and light images.** There the model is picked when it shouldn't be
  (light: 36.1 vs 36.7).
- **Next:** this pairing as an entrant (P3). It would also be measured on `icons` and `broad`, where the model hasn't
  been trained yet (E2).

### On the leaderboard (P3, `copycat.clean.v2`, 2026-11-09)

**The run:** one run outside an experiment, `32b4430e`, in the leading configuration: M, warmup 4,000, fp32 head,
20,000 steps, seed 1. It reached 26.32 dB on validate.

**The entrants:** every completed `copycat.svgformer` run outside an experiment is now two entrants
(`evaluator.protocols.copycat_clean`, `svgformer.entrant`):
- `run:<id>`: greedy decoding + polish;
- `run:<id>+structured`: the per-image pick above, with margin 2.

They are run on the GPU in the entrant's own spawned process. An image's time is its share of its batch's decoding, plus
its polish, plus (when paired) the tracer and the pick.

**Results** on the 168 held-out images:

| Entrant | All (mean of sets) | Synthetic | Icons | Broad | ms / image |
|---|---|---|---|---|---|
| **SvgFormer M or MDL tracer 23** | **29.65** ± 2.1 | 35.17 | 25.58 | 28.21 | 5,808 |
| MDL tracer 23 | 29.54 ± 2.2 | 34.72 | 25.54 | 28.37 | 5,263 |
| SvgFormer M + polish | 21.90 ± 1.3 | 32.81 | 16.07 | 16.81 | 267 |

The pairing against the tracer, image by image:

| Set | Difference (dB) | Model's SVG used | Better / worse |
|---|---|---|---|
| synthetic | **+0.45 ± 0.36** | 18 of 24 | 13 / 5 |
| icons | +0.04 ± 0.51 | 4 of 24 | 1 / 2 |
| broad | −0.16 ± 0.17 | 14 of 120 | 1 / 12 |

- **The pairing ranks first, by 0.11 dB.** That is inside either entrant's interval. The one difference that holds up
  image by image is synthetic's (+0.45). It is the first trained entrant ahead of the hand-written tracer on any set.
- **The model alone is a synthetic-only model.**
  - On `synthetic` it is behind the tracer by 1.9 dB here. On validate the gap was 0.6 dB, but held-out has only 24
    images.
  - On `icons` and `broad` it is about 16-17 dB, against the tracer's 25-28. It has never seen real artwork.
- **Off its domain, the margin rule misjudges.** On `broad` it used the model for 14 images, and 12 of them got worse.
  The rule was tuned on synthetic images, where a 2x worse fit to the input was a sign of corruption. An icon the model
  can't read misses the input by as much. The rule isn't retuned on held-out; the fix is the model, through E2 (artwork
  in the training mix).
- **Cost:** the pairing costs the tracer's time (5.3 s an image on CPU) plus 0.5 s. The model alone is 0.27 s an image
  on the GPU, 20 times the tracer's speed.

**Statistics:** 2-3 seeds per arm (training is expensive), compared image by image, mean ± 95% CI.

**Compute, to be replaced by the throughput measurement:**
- an S run of 100k steps at batch 64 is an estimated 1-3 GPU-hours;
- an M run is 2-3 times that;
- an L run at 4,096 tokens is about 10 times that;
- an RL run costs about G decodes per image per step, so an estimated 3-10 GPU-hours.

The scheduler runs one GPU job at a time.

**What success looks like:**
- An entrant ahead of MDL tracer v23 (29.54 dB) on `copycat.clean.v2`, and ahead on each of its three sets.
- It holds on `corrupt.v2`.
- Ahead on `illustrations` too.
- It runs in the browser in a few seconds.

## Phases

| Phase | What | Checked by |
|---|---|---|
| P0 | this doc | review |
| P1 | `libs/svgformer`: PyTorch on the RTX 5080 confirmed, the tokenizer, the grammar and constrained decoding; E0 on the sets already here | off-the-books 1-2; E0 recorded |
| B | the artwork: Iconify's sets, Openclipart, conversion, deduplication, `ATTRIBUTION`; `synthetic.v2`, `corrupt.v2`, `illustrations` | refusal counts; digests pinned |
| P2 | the model, the training stream (generated + artwork, corrupted on the fly), `copycat.svgformer`, held-out curves, a golden case on CPU; E1 | off-the-books 3 and 5; then registered runs |
| P3 | decoding entrants (`greedy`, `best8`, `best8+polish`) on the leaderboard; E2, E3 | registered runs |
| P4 | native L in the games core, `copycat.grpo`; E4, E5 | off-the-books 4; then registered runs |
| P5 | the illustrations track; E6 | registered runs |
| P6 | the model package and the page: the learned entrant beside the tracer, in the browser | parity; the in-app browser |

## Risks

- **The domain gap.** Most of the data is generated or is icon sets; if that doesn't transfer to `icons` and `broad`,
  E2 says so before RL is built.
- **Long sequences.** Real icons have many path nodes, illustrations far more. E0 measures it; the fixes are a coarser
  grid, simpler targets (the tracer's own trace of the clean render, which is shorter), or a larger cap.
- **Illustrations outside the subset.** Strokes and gradients are common in clipart. If most is refused, the
  illustrations track needs strokes in the subset first (0022's open question).
- **Contamination.** Training on a family the test sets contain would inflate every score. Family choice and
  deduplication guard it.
- **Reward hacking.** Structure ratios are logged and gated; the KL keeps the model near its SFT start.
- **Licences.** Each set's terms are recorded in the sources file when it is pinned; a set whose terms change is dropped.

## Alternatives considered

- **A posterior over scenes by reversible-jump MCMC** (an earlier draft of this doc). Set aside before it was built,
  in favour of a learned model, which can carry priors across images that a per-image sampler can't.
- **Our own `libs/autodiff` on a GPU backend (CuPy).** Keeps the from-scratch story, but needs new ops and kernels before
  the first result. Deferred: PyTorch first, a port later if it earns one.
- **A Rust core on the CPU** (extending `libs/rl`'s MLP with attention). Deterministic and WASM-ready, but far too slow
  to train at this size.

## Open questions

- **Share-alike and non-commercial sets** (OpenMoji, MMSVG): whether to opt in, and how such a model would be labelled.
- **Image size:** 128 px for icons and 256 px for illustrations. A logo from a generator is larger; every input is
  downsampled to the track's size.
- **Strokes and gradients:** still outside the subset (0022), and so outside the vocabulary. The illustrations track may
  force the question.

## Sources

- RLRF, rendering feedback for SVG generation: <https://arxiv.org/abs/2505.20793>
- StarVector: <https://arxiv.org/abs/2312.11277>
- OmniSVG: <https://arxiv.org/abs/2504.06263>
- DeepSVG: <https://arxiv.org/abs/2007.11301>
- Pix2Seq (coordinates as tokens): <https://arxiv.org/abs/2109.10852>
- ViT: <https://arxiv.org/abs/2010.11929>
- DeepSeekMath (GRPO): <https://arxiv.org/abs/2402.03300>
- Iconify icon sets and their licences: <https://github.com/iconify/icon-sets> (`collections.json`)
- Noto Emoji: <https://github.com/googlefonts/noto-emoji>; Fluent Emoji: <https://github.com/microsoft/fluentui-emoji>
- Openclipart's licence: <https://openclipart.org/share>
- unDraw's licence (excluded): <https://undraw.co/license>
- MMSVG-Illustration (excluded): <https://huggingface.co/datasets/OmniSVG/MMSVG-Illustration>

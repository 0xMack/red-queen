# svgformer

Copy Cat, learned ([docs/design/0024](../../docs/design/0024-copycat-learned-inverse.md)): a small encoder-decoder
transformer that reads an image and writes the clean, layered SVG behind it as a sequence of tokens, trained on generated
and openly licensed artwork, then fine-tuned by RL against the rendered result. The MDL tracer (docs/design/0022) is its
baseline and partner.

## Contents

- `tokens.py` -- the vocabulary and the tokenizer: an SVG in Copy Cat's subset (docs/design/0022 Decision 2) to tokens and
  back. Coordinates on a 1,280-step grid over the canvas plus a margin of 1/8 a side (1/8 px at 128 px), colours at 6 bits a
  channel, shapes in paint order. Plain Python: no torch.
- `grammar.py` -- the state machine that says which tokens may come next, so constrained decoding always produces an SVG
  that parses. Plain Python.
- `ceilings.py` -- experiment E0: the fidelity a model writing these tokens can't pass, and sequence lengths, per set.
- `artwork.py`, `artwork_sources.json`, `convert.py`, `stroke.py`, `build.py` -- the training artwork (phase B): 65 openly licensed
  Iconify icon sets and Openclipart's illustrations, pinned by digest, downloaded on demand, never in the repo:

      uv run python -I -m svgformer.artwork fetch data/experiments/svgformer-artwork
      uv run python -I -m svgformer.artwork build data/experiments/svgformer-artwork

  `convert.py` turns a general SVG into the subset exactly (transforms, styles, `<use>`, even-odd holes) or refuses it
  with a reason; `stroke.py` turns strokes into filled outlines, checked by rendering against an exact union of pieces; `build.py` drops duplicates and anything that looks like a held-out image, holds out Openclipart's
  `illustrations` by author, and counts refusals per source in `build.json`. Credits are in [ATTRIBUTION.md](ATTRIBUTION.md).

- `data.py` -- the training stream (plain Python): sample `i` of stream `seed` is a `synthetic` image, corrupted by
  `corrupt.v1` at a severity drawn from its seed, and its tokens -- a pure function of `(seed, i)`, made in data workers.
  With `artwork_share` > 0 that fraction of samples are icons from `build.py`'s `icons.jsonl`, corrupted afresh each draw.
  Trained from scratch on that mix, the decoder learns the icon sets as a language model and stops reading the image
  (0024 E2); fine-tuning a model that already reads images keeps it reading them.
- `model.py` -- the transformer: a ViT encoder over 8 px patches and a causal decoder with cross-attention, sizes XS (tests),
  S (~5M), M (~12M) and L (256 px, 4,096 tokens). A number token's value enters and leaves as Fourier features in the token
  grid's frame, as do the patches' places, so a coordinate and the pixels it names line up (without them, the model
  ignored the image: 0024 "P2 results").
- `decode.py` -- grammar-constrained generation with a KV cache (greedy or sampled); every sequence parses. `no_repeat`
  bans path segments going nowhere or repeating the last few and ends a path that stalls (it doesn't fix 0024 E2's loops).
- `train.py` -- the training loop (AdamW, warmup then cosine, bf16 autocast) and `evaluate()`, which scores greedy SVGs
  exactly as `copycat.clean.v2` does. The trainer's `copycat.svgformer` records it as a run:

      uv run redqueen train copycat.svgformer --game copycat --budget steps=20000 -p model=S

  `-p validate_groups=...` scores more groups than `synthetic/validate` (held is their mean; never a `heldout` split), and
  `-p artwork_share=0.3` mixes in the artwork.
- `entrant.py` -- a run as a `copycat.clean.v2` entrant: greedy + polish, and per image that or the MDL tracer's SVG
  (`PICK_MARGIN`, 1.5: the tracer only if it fits the input 1.5 times as closely).

These four need PyTorch: `uv sync --all-packages --extra torch` (CUDA 13.2 wheels on Windows, CPU wheels on Linux). Nothing
else in the workspace depends on it.

## Tests

`uv run pytest libs/svgformer` -- the tokenizer round-trips every `synthetic` image and the grammar accepts every
tokenized SVG and only what parses (fuzzed); the stream is a pure function of its seed, the cached decoder step
matches teacher forcing, random weights still decode to SVGs that parse, and a model memorises a batch. Tests that need
torch are skipped without it. `jobs/tests/test_golden.py`'s `svgformer` case pins a CPU run (to 1e-4: torch's CPU kernels
round differently by instruction set).

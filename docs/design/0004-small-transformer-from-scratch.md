# 0004 — A small transformer/language model, built from scratch

Status: **Draft**
Relates to: [0003](0003-algorithm-landscape-and-roadmap.md) (phase 6)

## Context

Doc 0003 revised its roadmap to build a small transformer/LM from scratch instead of wiring up an
external LLM API, for the same reason `RedQueenCbind` and `evolve` exist as from-scratch
implementations rather than calls to existing libraries: understanding the mechanism is the point.
This is a bigger jump than any previous phase, though — everything built so far (linear GP, tree
GP, neuroevolution) is trained by *evolution* (selection + variation, no gradients at all). A
transformer is normally trained by backpropagation. That's new territory for this repo, not an
incremental extension of `evolve`, which is exactly why this gets its own doc before any code.

## The scope question that matters most: how far down does "from scratch" go?

Three honest options, in order of how literally they take "from scratch":

1. **A full custom autodiff engine** (a small reverse-mode automatic differentiation system,
   Karpathy's `micrograd` is the canonical ~150-line example) — every gradient, including the
   engine that computes them, is something we built. Maximum learning value, truest to the ethos,
   but it's a genuinely separate, well-defined sub-project layered *underneath* the transformer
   work, not part of understanding transformers specifically.
2. **NumPy tensors + hand-derived backward passes.** No autodiff engine — every layer's forward
   *and* backward pass (the actual backprop equations for attention, layer norm, the MLP block) is
   derived and hand-coded. This is "from scratch" for the thing we're actually trying to
   understand (how a transformer computes and learns), while treating array arithmetic the way
   `evolve` treats `random.Random` — infrastructure we use, not the subject of study.
3. **An existing autodiff/tensor framework** (PyTorch) under a hand-built transformer architecture
   (our own attention/embedding/block code, not a pretrained model or a high-level `transformers`
   library call). Fastest to a working result, but the framework does the part doc 0003 explicitly
   named as the point ("understanding the mechanism... not just wiring up someone else's").

**Decided: option 1**, a full custom autodiff engine, built first as its own milestone. More total
scope than option 2, and truer to the ethos at every layer, not just the transformer-specific one.

**A practicality note this implies**: Karpathy's `micrograd` is *scalar*-valued — one graph node
per individual float. That's fine for the tiny MLPs it was built to demonstrate, but a transformer
even at toy scale has thousands of values flowing through matrix multiplications; a graph node per
scalar would make Python object overhead dominate and training impractically slow. So the engine
here is **array-valued**: each node wraps a NumPy array (not a single float), and ops are vectorized
(elementwise +, *, matmul, sum, etc.) with hand-coded backward rules per op, same reverse-mode
autodiff idea as `micrograd`, scaled the way it needs to be to actually train something. NumPy
remains the array-arithmetic substrate underneath, exactly the same "infrastructure, not the
subject of study" role it plays in option 2 above — just one level further down, under the
autodiff engine rather than under hand-coded transformer backward passes.

## Other scope decisions (lower-stakes, deciding here rather than asking)

- **Package split**: `libs/autodiff` (the engine — `Tensor`, ops, `.backward()`; general-purpose,
  nothing transformer-specific) and `libs/tinylm` (the transformer architecture, built on top of
  `autodiff`). This is the first `libs/` package in this repo that depends on another one — every
  other package (`evolve`, `games`, `telemetry`) is currently a leaf; `autodiff` is a genuine shared
  primitive, not algorithm- or domain-specific, the same way `telemetry` is infrastructure `jobs/`
  depends on rather than the other way around.

- **Tokenization: character-level.** Simplest, no BPE implementation needed as a prerequisite,
  standard starting point (nanoGPT/Karpathy), works fine at small scale.
- **Scale: genuinely tiny.** A handful of transformer blocks, small embedding dimension, short
  context window — enough to see real attention/training dynamics, small enough to train on CPU in
  reasonable time (no GPU acceleration in this picture — pure NumPy). Exact numbers TBD once the
  first version is running and its speed is known — same "decide specifics when reached" approach
  every other phase used.
- **Not wired into `evolve` or `telemetry` yet** — per doc 0003, phase 6 is standalone; phase 7
  (deferred, open-ended) decides how/whether it connects to the rest of the repo.

## Decided: training corpus

A small public-domain literary text — *Alice's Adventures in Wonderland* / *Through the
Looking-Glass* (the source of "Red Queen"). Free, thematically fitting, plenty of character-level
structure to learn from without needing anything domain-specific (which phase 7 deliberately
defers).

## Incremental plan

1. **`libs/autodiff`**: `Tensor` (wraps a NumPy array + gradient), core ops with hand-coded
   backward rules (add, multiply, matmul, sum, exp, log, and whatever else falls out of building
   the ops below), topological-sort-based `.backward()`. Validated by **gradient checking** — for
   every op, compare the engine's analytical gradient against a numerical one (finite differences)
   on random inputs. This is the standard way to test an autodiff engine and the most important
   correctness check in this whole phase: every later result depends on these gradients being
   right.
2. Character-level tokenizer (encode/decode, vocabulary from the training corpus) — `libs/tinylm`.
3. Embedding + positional encoding, built as `autodiff.Tensor` ops.
4. Self-attention (single head, then multi-head) — architecture code, no manual backward pass;
   `autodiff` computes it.
5. Transformer block (attention + MLP + layer norm + residuals), stacked to a small depth.
6. Training loop (forward, loss, `.backward()`, parameter update — plain SGD/Adam).
7. Generation (sampling from the trained model).
8. Validate: loss decreases over training, generated text visibly improves from random characters
   toward something structured — a notebook, same as every other phase's validation.

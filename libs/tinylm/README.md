# tinylm

A small transformer language model, built from scratch on
[`libs/autodiff`](../autodiff/README.md) ([docs/design/0004](../../docs/design/0004-small-transformer-from-scratch.md),
doc 0003 phase 6). Character-level, causal self-attention, trained by backpropagation — the first
thing in this repo trained by gradients rather than evolution.

## Contents

- `tokenizer.py` — `CharTokenizer`: character-level encode/decode, vocabulary built from the
  training corpus.
- `layers.py` — `Linear`, `Embedding`, `LayerNorm`, `CausalSelfAttention` (multi-head, causally
  masked), `MLP`, `TransformerBlock`. Every layer is built from `autodiff.Tensor` ops — no manual
  backward pass anywhere here; the autodiff engine computes every gradient.
- `model.py` — `TinyLM`: token + learned positional embeddings, a stack of `TransformerBlock`s, a
  final layer norm, and an output projection to vocabulary logits.
- `loss.py` — `cross_entropy`, the standard next-token-prediction loss.
- `optim.py` — `Adam`, hand-coded (plain SGD is notoriously hard to train transformers with).
- `generate.py` — sampling from a trained model (reads `.data` only, never calls `.backward()`).
- `data/alice.txt` — the training corpus: *Alice's Adventures in Wonderland* (Project Gutenberg,
  public domain) — the source of "Red Queen". Doc 0003 phase 7 (domain-specific corpora, e.g.
  programs already in `telemetry`'s `ArtifactStore`) is explicitly deferred until after this
  mechanism-validation phase works.

Not wired into `evolve` or `telemetry` yet — per doc 0003, phase 6 is standalone.

## Usage

```python
import numpy as np

from tinylm import Adam, CharTokenizer, TinyLM, cross_entropy, generate

text = open("data/alice.txt", encoding="utf-8").read()
tokenizer = CharTokenizer(text)
rng = np.random.default_rng(0)

model = TinyLM(
    vocab_size=tokenizer.vocab_size, max_seq_len=64,
    d_model=64, n_heads=4, n_layers=2, d_hidden=128, rng=rng,
)
optimizer = Adam(model.parameters(), lr=3e-3)

# one training step, given a (batch, seq_len) array of token ids and the same shifted by one
logits = model(input_ids)
loss = cross_entropy(logits, target_ids)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(generate(model, tokenizer, prompt="Alice ", max_new_tokens=100, rng=rng))
```

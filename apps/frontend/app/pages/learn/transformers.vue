<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
// Numbers and samples are notebooks/0006-tinylm-from-scratch.ipynb's real, executed output.
const modelCode = `class TinyLM:
    def __call__(self, token_ids):                     # (batch, seq_len) ints
        _batch, seq_len = token_ids.shape
        x = self.token_embedding(token_ids) + self.position_embedding(np.arange(seq_len))
        for block in self.blocks:                      # 2 transformer blocks
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)                            # (batch, seq_len, vocab) logits`

const attentionCode = `def __call__(self, x):
    batch, seq_len, _ = x.shape
    q = self._split_heads(self.query(x), batch, seq_len)   # (batch, heads, seq, head_dim)
    k = self._split_heads(self.key(x), batch, seq_len)
    v = self._split_heads(self.value(x), batch, seq_len)

    scores = q.matmul(k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
    scores = scores + Tensor(_causal_mask(seq_len))        # -1e9 above the diagonal
    weights = softmax(scores, axis=-1)
    out = weights.matmul(v)
    return self.out_proj(self._merge_heads(out, batch, seq_len))`

const blockCode = `class TransformerBlock:
    def __call__(self, x):
        x = x + self.attn(self.ln1(x))   # tokens exchange information
        x = x + self.mlp(self.ln2(x))    # each token processes what it gathered
        return x`

const trainCode = `model = TinyLM(vocab_size=75, max_seq_len=32, d_model=64, n_heads=4,
               n_layers=2, d_hidden=128, rng=rng)      # 78,795 parameters
optimizer = Adam(model.parameters(), lr=3e-3)

for step in range(3000):
    x, y = get_batch()                 # 32 random 32-character windows; y = x shifted by one
    loss = cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()                    # libs/autodiff computes every gradient
    optimizer.step()`

const samples = [
  { step: 0, loss: 4.41, text: "Alice yu‘n)er“aotIg,uJs) [lhr)‘B(nt nn.Bf)ah‘Xg,[;3au”on ? a ZyEgd" },
  { step: 300, loss: 1.94, text: "Alice a— inmen wor as tares.” Alice wathe whas heatny apare.  “Thi" },
  { step: 900, loss: 1.69, text: "Alice reatiousised,” said the juter questing.”  “It quake her way " },
  { step: 2100, loss: 1.55, text: "Alice ratter, and at the sil.”  The Rabbit—and don a campiturll sa" },
  { step: 2700, loss: 1.42, text: "Alice fingons much up as not yea, how rooked ance worra thing inte" },
]
</script>

<template>
  <article class="prose-chapter">
    <p>
      The previous chapter built an engine that computes gradients. This one uses it for the
      architecture behind modern language models: a <strong>transformer</strong>, written from scratch in
      <code>libs/tinylm</code> -- no PyTorch, no pretrained weights -- and trained on
      <em>Alice's Adventures in Wonderland</em>, the book this project's name comes from. It's the one
      thing in the repo trained by gradients rather than evolution.
    </p>

    <h2>The task: guess the next character</h2>
    <p>
      The whole book is 144,602 characters using 75 distinct symbols. Each character becomes an integer
      id, and the model reads a window of 32 of them and predicts, at every position, what comes next.
      That's the entire training signal: text predicting itself, one character later. Before any
      training, a model that knows nothing should be equally unsure about all 75 symbols -- a loss of
      ln(75) ≈ 4.317. The untrained model measured 4.431: close, which is how you know the forward pass
      and loss are wired up correctly before spending minutes training.
    </p>
    <CodeBlock lang="python" :code="modelCode" />
    <p>
      Two lookups start it off: a learned vector per character (<em>what</em> it is) plus a learned
      vector per position (<em>where</em> it is). Attention by itself has no sense of order, so without
      the position embedding "was Alice" and "Alice was" would look identical.
    </p>

    <h2>Attention: every position asks every earlier one</h2>
    <p>
      Each position produces a <strong>query</strong> ("what am I looking for?"), a <strong>key</strong>
      ("what do I contain?"), and a <strong>value</strong> ("what do I pass on?"). A query scores every key
      with a dot product, softmax turns the scores into weights, and the output is the weighted mix of
      values. Four <strong>heads</strong> do this in parallel on 16-dimensional slices, so different
      heads can track different relationships.
    </p>
    <CodeBlock lang="python" :code="attentionCode" />
    <p>
      One line keeps it honest: the <strong>causal mask</strong>. When predicting the character after
      position <em>i</em>, the model mustn't see position <em>i+1</em> -- that's the answer. Adding −10⁹ to
      every score above the diagonal makes those weights zero after the softmax:
    </p>
    <ClientOnly><CausalMaskDemo /></ClientOnly>
    <p>
      There's a test for exactly this (<code>test_causal_attention_does_not_leak_future_information</code>):
      change a later character and assert that earlier positions' outputs don't move. A leak wouldn't
      crash anything -- the model would just learn to cheat, and its loss would look suspiciously good.
    </p>

    <h2>Blocks, residuals, and layer norm</h2>
    <p>
      A block is attention followed by a small two-layer MLP, each wrapped in a <strong>residual</strong>
      connection (add the input back to the output) and preceded by <strong>layer norm</strong> (rescale
      each position's vector to zero mean and unit variance). Residuals give gradients a direct path
      through the network; layer norm keeps activations in a sane range. Both are what make stacking
      blocks trainable at all.
    </p>
    <CodeBlock lang="python" :code="blockCode" />
    <Callout variant="finding" title="A real bug, caught by the first smoke test">
      <code>matmul</code>'s backward pass originally didn't handle broadcasting between a 2-D weight and a
      3-D batch of inputs -- which is exactly what every linear layer does inside a transformer. The
      per-op gradient checks at the time didn't cover that shape combination; the model's very first
      forward/backward run exposed it. The fix (<code>_unbroadcast</code> on both matmul gradients) now has its own tests
      for batched and mismatched-rank matmul, and a full transformer block is gradient-checked end to end,
      because composing many correct ops can still produce an incorrect whole.
    </Callout>

    <h2>Training it</h2>
    <CodeBlock lang="python" :code="trainCode" />
    <p>
      Two blocks, four heads, 64-dimensional vectors: 78,795 parameters. Three thousand steps took
      <strong>157.8 seconds</strong> on plain NumPy -- no GPU. The loss fell from 4.41 to about 1.45
      (averaged over the last 50 steps). Here's the same prompt, <code>"Alice "</code>, sampled at
      different points in that run:
    </p>
    <div class="card my-6 divide-y divide-line/60 overflow-hidden">
      <div v-for="s in samples" :key="s.step" class="grid gap-1 px-4 py-3 sm:grid-cols-[120px_minmax(0,1fr)] sm:gap-4">
        <div class="font-mono text-xs text-fg-subtle">
          step <span class="text-fg">{{ s.step }}</span><br>loss <span class="text-queen-300">{{ s.loss.toFixed(2) }}</span>
        </div>
        <p class="font-mono text-[13px] break-words text-fg-muted">{{ s.text }}</p>
      </div>
    </div>
    <p>
      Noise, then spaces and short words, then dialogue punctuation (<code>,” said</code>), then names from
      the book (<em>the Rabbit</em>). It isn't coherent English -- 3,000 steps on a model this size is a
      rounding error by real language-model standards -- but the character-level structure is genuinely
      learned, not templated: every sample above is unedited output.
    </p>

    <h2>Running it in your browser</h2>
    <p>
      The trained checkpoint is exported to ONNX -- the same forward pass, plus a <strong>KV cache</strong> so
      each new character only computes attention for itself instead of re-reading the whole window -- and
      runs on your own device via ONNX Runtime (docs/design/0009). The int8 variant stores its weights as
      8-bit integers: less than half the download, at the cost of predicting a different next character
      than the trained model about 2% of the time on held-out text. Pick either and compare.
    </p>
    <ClientOnly><TinyLMPlayground /></ClientOnly>

    <Callout variant="note" title="Why this one isn't on the leaderboard (yet)">
      Every other model in this project is evolved and plugs into <code>evolve</code>,
      <code>telemetry</code>, and the games. The transformer deliberately doesn't yet (docs/design/0004):
      the point was to prove the whole gradient stack works from scratch first. Connecting the two --
      say, a model trained on the programs evolution has produced -- is a later step on the roadmap.
    </Callout>
  </article>
</template>

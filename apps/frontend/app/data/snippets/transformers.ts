import type { Snippet } from "~/types/code"

// Transformers' snippets. TinyLM is Python (libs/tinylm, on libs/autodiff) and runs in the browser as an exported
// ONNX package -- there is no Rust transformer to show, so these are pseudocode and Python.
const SOURCE = "libs/tinylm/src/tinylm"

export const model: Snippet = {
  pseudo: `function tiny_lm(tokens)                          // a sequence of character ids
    x ← token_embedding(tokens) + position_embedding(0, 1, …, length − 1)
    for each of the 2 transformer blocks
        x ← block(x)
    x ← layer_norm(x)
    return head(x)                                  // for every position, a score per possible next character`,
  python: {
    source: `${SOURCE}/model.py`,
    code: `class TinyLM:
    def __call__(self, token_ids):                     # (batch, seq_len) ints
        _batch, seq_len = token_ids.shape
        x = self.token_embedding(token_ids) + self.position_embedding(np.arange(seq_len))
        for block in self.blocks:                      # 2 transformer blocks
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)                            # (batch, seq_len, vocab) logits`,
  },
}

export const attention: Snippet = {
  pseudo: `function attention(x)
    Q ← x · W_q,  K ← x · W_k,  V ← x · W_v          // split into 4 heads
    scores ← Q · Kᵀ / √head_dim                      // how much each position wants each other one
    scores[i, j] ← −∞ wherever j > i                 // the causal mask: no looking ahead
    weights ← softmax(scores) along each row
    return (weights · V), heads merged, · W_out`,
  python: {
    source: `${SOURCE}/layers.py`,
    code: `def __call__(self, x):
    batch, seq_len, _ = x.shape
    q = self._split_heads(self.query(x), batch, seq_len)   # (batch, heads, seq, head_dim)
    k = self._split_heads(self.key(x), batch, seq_len)
    v = self._split_heads(self.value(x), batch, seq_len)

    scores = q.matmul(k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
    scores = scores + Tensor(_causal_mask(seq_len))        # -1e9 above the diagonal
    weights = softmax(scores, axis=-1)
    out = weights.matmul(v)
    return self.out_proj(self._merge_heads(out, batch, seq_len))`,
  },
}

export const block: Snippet = {
  pseudo: `function block(x)
    x ← x + attention(layer_norm(x))     // tokens exchange information
    x ← x + mlp(layer_norm(x))           // each token processes what it gathered
    return x`,
  python: {
    source: `${SOURCE}/layers.py`,
    code: `class TransformerBlock:
    def __call__(self, x):
        x = x + self.attn(self.ln1(x))   # tokens exchange information
        x = x + self.mlp(self.ln2(x))    # each token processes what it gathered
        return x`,
  },
}

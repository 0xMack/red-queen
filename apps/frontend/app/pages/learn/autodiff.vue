<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
import * as code from "~/data/snippets/autodiff"
import * as math from "~/data/math/autodiff"

const unbroadcastCode = `def _unbroadcast(grad, shape):
    """Forward broadcasting silently reused a value at many positions; the gradient
    has to sum back over exactly those positions."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad`

const getitemCode = `def __getitem__(self, idx) -> Tensor:
    out = Tensor(self.data[idx], (self,), "getitem")

    def _backward():
        # np.add.at, not self.grad[idx] += ...: plain fancy-index assignment keeps only
        # ONE update when an index repeats (the same token twice in a sequence).
        np.add.at(self.grad, idx, out.grad)

    out._backward = _backward
    return out`

</script>

<template>
  <article class="prose-chapter">
    <p>
      Every Snake policy in this project was <em>evolved</em>: 243 weights, nudged at random, kept when
      the snake did better. That works at 243 weights. It doesn't at 78,795 -- the size of the
      <NuxtLink to="/learn/transformers">transformer</NuxtLink> this engine trains -- because random mutation can't tell which of those numbers to
      move, or which way. A <strong>gradient</strong> can: for every weight at once, it says exactly how
      the loss would change if you nudged it. This chapter builds the machine that computes gradients,
      <code>libs/autodiff</code>, from scratch.
    </p>

    <h2>The chain rule, as a graph</h2>
    <p>
      Any computation is a graph of small operations. If you know, for each operation, how its output
      changes with its inputs (the <em>local</em> derivative), the chain rule tells you how the final
      result changes with anything upstream: multiply the local derivatives along the path. Doing that
      from the output backwards, reusing each intermediate result once, is
      <strong>reverse-mode automatic differentiation</strong> -- backpropagation, in general form. Try
      it on a single neuron:
    </p>
    <ClientOnly><AutodiffPlayground /></ClientOnly>
    <p>
      Notice the gradient reaching <code>w</code> is the gradient at the <code>×</code> node times
      <code>x</code>, and vice versa -- each input's gradient is the <em>other</em> input's value. And
      the "gradient step" button is all training is: move every parameter a little against its
      gradient, repeat.
    </p>

    <h2>A Tensor remembers how it was made</h2>
    <p>
      In <code>libs/autodiff</code>, a <code>Tensor</code> wraps a NumPy array plus three things: its
      accumulated <code>grad</code>, the tensors it was computed from, and a <code>_backward</code>
      closure that pushes its gradient to those parents. Every operation builds that closure as it runs.
      Multiplication, in full:
    </p>
    <CodeBlock :snippet="code.mul" />
    <p>
      Gradients use <code>+=</code>, never <code>=</code>: a tensor used in two places gets gradient
      from both, and they add. Then <code>backward()</code> just orders the graph so every node runs
      after everything that depends on it:
    </p>
    <CodeBlock :snippet="code.backward" />

    <Callout variant="note" title="Why arrays, not single numbers">
      The classic teaching engine (Karpathy's micrograd) makes one graph node per scalar. A transformer
      pushes thousands of values through each matrix multiply, and a Python object per float would make
      interpreter overhead dominate. So a <code>Tensor</code> here holds a whole array, and NumPy does
      the arithmetic -- the same role <code>random.Random</code> plays for <code>evolve</code>. The
      graph, the backward rules, and <code>backward()</code> are what's built from scratch.
    </Callout>

    <h2>The two places it's easy to get wrong</h2>
    <p>
      <strong>Broadcasting.</strong> Adding a bias of shape <code>(features,)</code> to activations of
      shape <code>(batch, features)</code> silently reuses the bias for every row. Its gradient must
      therefore <em>sum</em> over those rows -- copying the upstream gradient would be wrong by a factor
      of the batch size. Every op that broadcasts reduces its gradients back to the original shape:
    </p>
    <CodeBlock lang="python" :code="unbroadcastCode" />
    <p>
      <strong>Repeated indices.</strong> An embedding lookup is fancy indexing: <code>table[token_ids]</code>.
      If the same token appears twice in a sequence, its row must receive <em>both</em> gradients. NumPy's
      plain <code>grad[idx] += g</code> quietly keeps only one of them; <code>np.add.at</code> accumulates
      correctly:
    </p>
    <CodeBlock lang="python" :code="getitemCode" />

    <h2>Trust, but verify: gradient checking</h2>
    <p>
      A wrong backward rule doesn't crash -- training just quietly gets worse. So every op in the engine
      is tested against a <strong>numerical gradient</strong>: nudge each input by ±ε, see how the output
      moves, and compare. The playground above does the same check live for <code>w</code>.
    </p>
    <MathFormula :formula="math.finiteDifference" caption="The definition of a derivative, with a small but finite ε: slow, simple, and hard to get wrong." />
    <CodeBlock :snippet="code.gradCheck" />
    <Callout variant="finding" title="The most important tests in the language-model work">
      Everything in the transformers chapter -- attention, layer norm, a 78,795-parameter model learning English
      spelling -- rests on these gradients being right. <code>libs/autodiff/tests</code> checks add,
      multiply, broadcasting, power and divide, matmul (including batched and mismatched batch ranks),
      sums and means over axes, exp and log, relu and tanh, transpose and reshape, indexing with repeated
      indices, softmax, and a whole two-layer network this way; <code>libs/tinylm/tests</code> then
      gradient-checks layer norm and a full transformer block.
    </Callout>

    <h2>From an engine to a network</h2>
    <p>
      With the engine in place, a layer is just a class that holds some tensors and combines them with
      tracked ops -- a linear layer is <code>x.matmul(weight) + bias</code>, and its backward pass comes
      for free. The one thing that isn't a graph op is the optimizer, which updates weights directly.
      This project uses Adam, hand-written: plain gradient descent struggles on transformers, and Adam's
      per-weight step sizes are a few lines on top of the gradients:
    </p>
    <MathScope>
      <MathFormula :formula="math.adamMomentum" />
      <MathFormula :formula="math.adamScale" />
      <MathFormula :formula="math.adamStep" caption="Each weight's step is its momentum over the size of its own gradients: a weight with small, steady gradients still moves." />
    </MathScope>
    <CodeBlock :snippet="code.adam" />
    <p>
      Evolution and gradients aren't rivals here, they're tools for different shapes of problem. Evolution
      needs only a score, handles non-differentiable things like "did the snake eat?", and is simple to
      run; gradients need a differentiable loss but scale to models evolution could never search. The
      <NuxtLink to="/learn/transformers">transformers chapter</NuxtLink> puts this engine to work on a language model, and
      <NuxtLink to="/learn/dqn">Deep Q-Networks</NuxtLink> on a network that learns from reward.
    </p>
  </article>
</template>

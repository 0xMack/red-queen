# autodiff

A small reverse-mode automatic differentiation engine, built from scratch
([docs/design/0004](../../docs/design/0004-small-transformer-from-scratch.md)) — the foundation
`libs/tinylm`'s transformer is built on. The first `libs/` package in this repo that another
package depends on; every other package so far is a leaf.

Unlike Karpathy's `micrograd` (the canonical teaching example — one graph node per individual
float), `Tensor` here wraps a NumPy array. A transformer even at toy scale pushes thousands of
values through matrix multiplications; a graph node per scalar would make Python object overhead
dominate and training impractically slow. NumPy is the array-arithmetic substrate underneath — the
same role `random.Random` plays for `evolve` — while the graph, each op's backward rule, and
`.backward()` itself are what's actually built here.

## Contents

- `tensor.py` — `Tensor`: wraps a NumPy array + its gradient, tracks the ops it was computed from,
  and `.backward()` (topological-sort-based reverse-mode differentiation). Core ops: `+`, `-`, `*`,
  `/`, `**`, `matmul`/`@`, `sum`, `mean`, `exp`, `log`, `relu`, `tanh`, `transpose`, `reshape`.
- `functional.py` — composite ops built from `Tensor` primitives (`softmax`) — their backward pass
  comes for free from the graph, no hand-derived rule needed.

## Validation: gradient checking

Every op is tested by comparing its analytical gradient (via `.backward()`) against a numerical one
(central finite differences) — the standard way to validate an autodiff engine, and the most
important correctness check in this whole effort, since every later result (training a transformer)
depends on these gradients being right. See `tests/test_tensor.py`, including a test for the
classic autodiff bug (a value reused in an expression must accumulate gradient from every path back
to it, not just overwrite) and an integration test through a small two-layer network.

## Usage

```python
from autodiff import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
w = Tensor([[0.5, -0.5], [0.5, 0.5]])

y = x.matmul(w).relu().sum()
y.backward()

print(x.grad)  # dy/dx
```

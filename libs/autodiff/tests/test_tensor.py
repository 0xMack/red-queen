"""Gradient checking: for every op, the engine's analytical gradient (via .backward()) is compared
against a numerical one (central finite differences). This is the standard way to validate an
autodiff engine, and the most important correctness check in this repo's whole transformer/LM
effort -- every later result depends on these gradients being right.
"""

import numpy as np

from autodiff import Tensor, softmax


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """f(x: np.ndarray) -> float. Perturbs each element of x by +-eps and returns the central
    difference gradient, the same shape as x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        f_plus = f(x)
        x[idx] = original - eps
        f_minus = f(x)
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_add_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    a_data, b_data = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))

    a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
    (a + b).sum().backward()

    assert np.allclose(a.grad, numerical_grad(lambda x: (x + b_data).sum(), a_data.copy()), atol=1e-4)
    assert np.allclose(b.grad, numerical_grad(lambda x: (a_data + x).sum(), b_data.copy()), atol=1e-4)


def test_broadcast_add_gradient_sums_over_the_broadcast_dimension():
    # (5, 3) + (3,) -- the bias's gradient must sum over the 5 broadcast rows, not just copy them
    x_data = np.random.default_rng(1).normal(size=(5, 3))
    b_data = np.random.default_rng(2).normal(size=(3,))

    x, b = Tensor(x_data.copy()), Tensor(b_data.copy())
    (x + b).sum().backward()

    assert np.allclose(x.grad, np.ones_like(x_data))
    assert np.allclose(b.grad, np.full_like(b_data, 5.0))


def test_mul_gradient_matches_numerical():
    rng = np.random.default_rng(3)
    a_data, b_data = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))

    a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
    (a * b).sum().backward()

    assert np.allclose(a.grad, numerical_grad(lambda x: (x * b_data).sum(), a_data.copy()), atol=1e-4)
    assert np.allclose(b.grad, numerical_grad(lambda x: (a_data * x).sum(), b_data.copy()), atol=1e-4)


def test_tensor_used_twice_accumulates_gradient_from_both_paths():
    # the classic autodiff bug: a value reused in an expression must sum gradient contributions
    # from every path back to it, not just the last one computed. d/dx sum(x * x) = 2x.
    x_data = np.array([2.0, 3.0, -1.0])
    x = Tensor(x_data.copy())

    (x * x).sum().backward()

    assert np.allclose(x.grad, 2 * x_data)


def test_pow_and_div_gradient_match_numerical():
    rng = np.random.default_rng(4)
    a_data = rng.uniform(0.5, 2.0, size=(3, 3))  # keep away from 0 to avoid a div blow-up
    b_data = rng.uniform(0.5, 2.0, size=(3, 3))

    a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
    (a / b).sum().backward()

    assert np.allclose(a.grad, numerical_grad(lambda x: (x / b_data).sum(), a_data.copy()), atol=1e-4)
    assert np.allclose(b.grad, numerical_grad(lambda x: (a_data / x).sum(), b_data.copy()), atol=1e-4)


def test_matmul_gradient_matches_numerical():
    rng = np.random.default_rng(5)
    a_data, b_data = rng.normal(size=(4, 3)), rng.normal(size=(3, 2))

    a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
    a.matmul(b).sum().backward()

    assert np.allclose(a.grad, numerical_grad(lambda x: (x @ b_data).sum(), a_data.copy()), atol=1e-4)
    assert np.allclose(b.grad, numerical_grad(lambda x: (a_data @ x).sum(), b_data.copy()), atol=1e-4)


def test_sum_and_mean_with_axis_gradient_match_numerical():
    rng = np.random.default_rng(6)
    x_data = rng.normal(size=(4, 3))

    x_sum = Tensor(x_data.copy())
    x_sum.sum(axis=1).sum().backward()
    assert np.allclose(x_sum.grad, np.ones_like(x_data))

    x_mean = Tensor(x_data.copy())
    x_mean.mean(axis=1).sum().backward()
    assert np.allclose(x_mean.grad, numerical_grad(lambda x: x.mean(axis=1).sum(), x_data.copy()), atol=1e-4)


def test_exp_and_log_gradient_match_numerical():
    rng = np.random.default_rng(7)
    x_data = rng.uniform(0.1, 2.0, size=(3, 3))

    x_exp = Tensor(x_data.copy())
    x_exp.exp().sum().backward()
    assert np.allclose(x_exp.grad, numerical_grad(lambda x: np.exp(x).sum(), x_data.copy()), atol=1e-4)

    x_log = Tensor(x_data.copy())
    x_log.log().sum().backward()
    assert np.allclose(x_log.grad, numerical_grad(lambda x: np.log(x).sum(), x_data.copy()), atol=1e-4)


def test_relu_and_tanh_gradient_match_numerical():
    rng = np.random.default_rng(8)
    x_data = rng.normal(size=(3, 4))

    x_relu = Tensor(x_data.copy())
    x_relu.relu().sum().backward()
    assert np.allclose(x_relu.grad, numerical_grad(lambda x: np.maximum(0.0, x).sum(), x_data.copy()), atol=1e-4)

    x_tanh = Tensor(x_data.copy())
    x_tanh.tanh().sum().backward()
    assert np.allclose(x_tanh.grad, numerical_grad(lambda x: np.tanh(x).sum(), x_data.copy()), atol=1e-4)


def test_transpose_and_reshape_gradient_match_numerical():
    rng = np.random.default_rng(9)
    x_data = rng.normal(size=(3, 4))

    x_t = Tensor(x_data.copy())
    (x_t.transpose() * 2.0).sum().backward()
    assert np.allclose(x_t.grad, np.full_like(x_data, 2.0))

    x_r = Tensor(x_data.copy())
    x_r.reshape(12).sum().backward()
    assert np.allclose(x_r.grad, np.ones_like(x_data))


def test_4d_transpose_gradient_matches_numerical():
    # the permutation multi-head attention uses: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
    rng = np.random.default_rng(12)
    x_data = rng.normal(size=(2, 5, 3, 4))
    weights = rng.normal(size=(2, 3, 5, 4))

    x = Tensor(x_data.copy())
    (x.transpose(0, 2, 1, 3) * weights).sum().backward()

    def loss(x_perturbed):
        return (x_perturbed.transpose(0, 2, 1, 3) * weights).sum()

    assert np.allclose(x.grad, numerical_grad(loss, x_data.copy()), atol=1e-4)


def test_batched_matmul_gradient_matches_numerical():
    # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -- attention score shape
    rng = np.random.default_rng(13)
    a_data = rng.normal(size=(2, 3, 4, 5))
    b_data = rng.normal(size=(2, 3, 5, 4))

    a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
    a.matmul(b).sum().backward()

    assert np.allclose(a.grad, numerical_grad(lambda x: (x @ b_data).sum(), a_data.copy()), atol=1e-4)
    assert np.allclose(b.grad, numerical_grad(lambda x: (a_data @ x).sum(), b_data.copy()), atol=1e-4)


def test_matmul_with_mismatched_batch_rank_broadcasts_correctly():
    # a Linear layer's shape: a 2D (in, out) weight applied to a 3D (batch, seq, in) input --
    # NumPy broadcasts this in the forward pass automatically; the weight's gradient must sum
    # contributions over the broadcast batch dimension, the same way +/* already handle broadcasting.
    rng = np.random.default_rng(17)
    x_data = rng.normal(size=(2, 5, 3))  # (batch, seq, in)
    w_data = rng.normal(size=(3, 4))  # (in, out), no batch dim

    x, w = Tensor(x_data.copy()), Tensor(w_data.copy())
    x.matmul(w).sum().backward()

    assert np.allclose(x.grad, numerical_grad(lambda v: (v @ w_data).sum(), x_data.copy()), atol=1e-4)
    assert np.allclose(w.grad, numerical_grad(lambda v: (x_data @ v).sum(), w_data.copy()), atol=1e-4)


def test_getitem_gradient_accumulates_for_repeated_indices():
    # the embedding-lookup case: if the same row is selected more than once, gradient must
    # accumulate from every occurrence, not just the last (plain assignment would silently drop
    # earlier contributions -- np.add.at is what makes this correct).
    table_data = np.random.default_rng(14).normal(size=(5, 3))
    idx = np.array([1, 3, 1, 1])  # row 1 selected three times

    table = Tensor(table_data.copy())
    weights = np.random.default_rng(15).normal(size=(4, 3))
    (table[idx] * weights).sum().backward()

    expected = np.zeros_like(table_data)
    for i, row in zip(idx, weights, strict=True):
        expected[i] += row
    assert np.allclose(table.grad, expected)


def test_getitem_row_col_gather_gradient_matches_numerical():
    # the cross-entropy case: probs[(rows, cols)] gathers one value per row at a per-row column.
    rng = np.random.default_rng(16)
    x_data = rng.normal(size=(4, 6))
    cols = np.array([2, 0, 5, 3])
    rows = np.arange(4)

    x = Tensor(x_data.copy())
    x[(rows, cols)].sum().backward()

    def loss(x_perturbed):
        return x_perturbed[(rows, cols)].sum()

    assert np.allclose(x.grad, numerical_grad(loss, x_data.copy()), atol=1e-4)


def test_softmax_gradient_matches_numerical():
    def numpy_softmax(x, axis=-1):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=axis, keepdims=True)

    rng = np.random.default_rng(10)
    x_data = rng.normal(size=(2, 5))
    weights = rng.normal(size=(2, 5))  # so the scalar loss depends on every output element

    x = Tensor(x_data.copy())
    (softmax(x) * weights).sum().backward()

    def loss(x_perturbed):
        return (numpy_softmax(x_perturbed) * weights).sum()

    assert np.allclose(x.grad, numerical_grad(loss, x_data.copy()), atol=1e-4)


def test_backward_through_a_small_two_layer_network_matches_numerical():
    # y = relu(x @ w1) @ w2, loss = sum(y) -- an integration test resembling a real MLP layer,
    # exercising matmul + relu + the graph mechanism together rather than one op in isolation.
    rng = np.random.default_rng(11)
    x_data = rng.normal(size=(4, 3))
    w1_data = rng.normal(size=(3, 5))
    w2_data = rng.normal(size=(5, 2))

    x, w1, w2 = Tensor(x_data.copy()), Tensor(w1_data.copy()), Tensor(w2_data.copy())
    loss = x.matmul(w1).relu().matmul(w2).sum()
    loss.backward()

    def forward(x_, w1_, w2_):
        return np.maximum(0.0, x_ @ w1_) @ w2_

    assert np.allclose(x.grad, numerical_grad(lambda v: forward(v, w1_data, w2_data).sum(), x_data.copy()), atol=1e-4)
    assert np.allclose(
        w1.grad,
        numerical_grad(lambda v: forward(x_data, v, w2_data).sum(), w1_data.copy()),
        atol=1e-4,
    )
    assert np.allclose(
        w2.grad,
        numerical_grad(lambda v: forward(x_data, w1_data, v).sum(), w2_data.copy()),
        atol=1e-4,
    )


def test_zero_grad_resets_to_zero():
    x = Tensor(np.array([1.0, 2.0]))
    (x * x).sum().backward()
    assert not np.allclose(x.grad, 0.0)

    x.zero_grad()

    assert np.allclose(x.grad, 0.0)

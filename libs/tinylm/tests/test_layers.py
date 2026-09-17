"""Layer-level tests. Every autodiff primitive is already gradient-checked in libs/autodiff --
these tests focus on shapes through composition, architecture-specific correctness properties
(causal masking, layer-norm statistics), and one full numerical gradient check through an entire
TransformerBlock as an integration-level proof that composing many ops together (layer norm's
mean/variance/sqrt, attention's masking/softmax/multi-head reshape-transpose) didn't introduce a
mistake no individual primitive's test could catch.
"""

import numpy as np

from autodiff import Tensor
from tinylm.layers import MLP, CausalSelfAttention, Embedding, LayerNorm, Linear, TransformerBlock


def _numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
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


def test_linear_output_shape():
    rng = np.random.default_rng(0)
    layer = Linear(in_features=3, out_features=5, rng=rng)

    out = layer(Tensor(rng.normal(size=(2, 4, 3))))

    assert out.shape == (2, 4, 5)


def test_embedding_lookup_shape_and_repeated_index_gradient():
    rng = np.random.default_rng(1)
    embedding = Embedding(num_embeddings=10, dim=4, rng=rng)
    idx = np.array([[1, 2, 1]])  # token 1 appears twice in this one sequence

    out = embedding(idx)
    assert out.shape == (1, 3, 4)

    out.sum().backward()
    # row 1's gradient must reflect both occurrences (2x a single occurrence's contribution)
    assert np.allclose(embedding.weight.grad[1], 2 * embedding.weight.grad[2])


def test_layer_norm_output_has_zero_mean_and_unit_variance():
    rng = np.random.default_rng(2)
    ln = LayerNorm(dim=8)
    x = Tensor(rng.normal(loc=5.0, scale=3.0, size=(2, 8)))

    out = ln(x)

    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out.data.std(axis=-1), 1.0, atol=1e-3)


def test_layer_norm_gradient_matches_numerical():
    rng = np.random.default_rng(3)
    ln = LayerNorm(dim=6)
    x_data = rng.normal(size=(2, 6))

    x = Tensor(x_data.copy())
    ln(x).sum().backward()

    def f(v):
        centered = v - v.mean(axis=-1, keepdims=True)
        var = (centered**2).mean(axis=-1, keepdims=True)
        return (centered / np.sqrt(var + ln.eps)).sum()

    assert np.allclose(x.grad, _numerical_grad(f, x_data.copy()), atol=1e-3)


def test_causal_attention_output_shape():
    rng = np.random.default_rng(4)
    attn = CausalSelfAttention(d_model=8, n_heads=2, rng=rng)

    out = attn(Tensor(rng.normal(size=(2, 5, 8))))

    assert out.shape == (2, 5, 8)


def test_causal_attention_does_not_leak_future_information():
    # the whole point of the causal mask: output at position i must be identical regardless of
    # what appears at positions > i. Change a later token's input and confirm earlier outputs
    # are unaffected.
    rng = np.random.default_rng(5)
    attn = CausalSelfAttention(d_model=8, n_heads=2, rng=rng)
    x_data = rng.normal(size=(1, 5, 8))

    out_before = attn(Tensor(x_data.copy())).data

    x_data_changed = x_data.copy()
    x_data_changed[0, -1] = rng.normal(size=8) * 100  # drastically change only the last position
    out_after = attn(Tensor(x_data_changed)).data

    assert np.allclose(out_before[0, :-1], out_after[0, :-1])  # every earlier position: unchanged
    assert not np.allclose(out_before[0, -1], out_after[0, -1])  # the changed position itself: does change


def test_mlp_output_shape():
    rng = np.random.default_rng(6)
    mlp = MLP(d_model=8, d_hidden=16, rng=rng)

    out = mlp(Tensor(rng.normal(size=(2, 5, 8))))

    assert out.shape == (2, 5, 8)


def test_transformer_block_output_shape():
    rng = np.random.default_rng(7)
    block = TransformerBlock(d_model=8, n_heads=2, d_hidden=16, rng=rng)

    out = block(Tensor(rng.normal(size=(2, 5, 8))))

    assert out.shape == (2, 5, 8)


def test_transformer_block_gradient_matches_numerical():
    rng = np.random.default_rng(8)
    block = TransformerBlock(d_model=6, n_heads=2, d_hidden=12, rng=rng)
    x_data = rng.normal(size=(1, 4, 6))

    x = Tensor(x_data.copy())
    block(x).sum().backward()

    def f(v):
        return block(Tensor(v)).sum().data

    numerical = _numerical_grad(f, x_data.copy(), eps=1e-4)
    assert np.allclose(x.grad, numerical, atol=1e-2)

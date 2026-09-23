"""The Rust network and optimizer against the `libs/autodiff` oracle (reference_nn.py)."""

import numpy as np
import pytest
from reference_nn import adam_steps, forward_backward

from rl import _native

SHAPES = [
    ([11, 16, 3], ["relu", "linear"]),
    ([11, 64, 64, 3], ["relu", "relu", "linear"]),
    ([27, 32, 16, 3], ["tanh", "relu", "linear"]),
    ([2, 8, 1], ["tanh", "tanh"]),
]


@pytest.mark.parametrize(("layer_sizes", "activations"), SHAPES)
def test_forward_and_gradients_match_autodiff(layer_sizes, activations):
    rng = np.random.default_rng(len(layer_sizes) * 100 + layer_sizes[0])
    params = np.array(_native.mlp_init(layer_sizes, activations, 7))
    params += rng.normal(0, 0.1, params.shape)  # nonzero biases too
    batch = 9
    inputs = rng.normal(size=(batch, layer_sizes[0]))
    upstream = rng.normal(size=(batch, layer_sizes[-1]))

    outputs, grads = _native.mlp_forward_backward(
        layer_sizes, activations, params.tolist(), inputs.ravel().tolist(), batch, upstream.ravel().tolist()
    )
    expected_outputs, expected_grads = forward_backward(layer_sizes, activations, params, inputs, upstream)

    np.testing.assert_allclose(np.array(outputs).reshape(batch, -1), expected_outputs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grads, expected_grads, rtol=1e-11, atol=1e-12)


def test_init_is_he_uniform_with_zero_biases_and_seeded():
    layer_sizes, activations = [11, 64, 3], ["relu", "linear"]
    params = np.array(_native.mlp_init(layer_sizes, activations, 3))
    assert params.size == 11 * 64 + 64 + 64 * 3 + 3
    first = params[: 11 * 64]
    assert np.abs(first).max() <= np.sqrt(6 / 11) and first.std() > 0.3
    assert np.all(params[11 * 64 : 11 * 64 + 64] == 0)
    assert params.tolist() == _native.mlp_init(layer_sizes, activations, 3)
    assert params.tolist() != _native.mlp_init(layer_sizes, activations, 4)


def test_adam_matches_the_paper():
    rng = np.random.default_rng(11)
    params = rng.normal(size=40)
    gradients = [rng.normal(size=40) * scale for scale in (1.0, 0.1, 5.0, 1e-4, 2.0)]
    result = _native.adam_steps(params.tolist(), [g.tolist() for g in gradients], 1e-2, 0.9, 0.99, 1e-7)
    np.testing.assert_allclose(result, adam_steps(params, gradients, 1e-2, 0.9, 0.99, 1e-7), rtol=1e-13, atol=1e-15)


def test_shape_mistakes_are_errors():
    with pytest.raises(ValueError, match="activations"):
        _native.mlp_init([3, 4, 2], ["relu"], 0)
    with pytest.raises(ValueError, match="unknown activation"):
        _native.mlp_init([3, 2], ["sigmoid"], 0)
    with pytest.raises(ValueError, match="parameters"):
        _native.mlp_forward_backward([3, 2], ["linear"], [0.0], [0.0] * 3, 1, [0.0] * 2)

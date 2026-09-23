import numpy as np
import pytest
from tinylm import CharTokenizer, TinyLM
from tinylm.checkpoint import named_parameters

from modelpack import LocalModelStore
from modelpack.lm import LMConfig, LMRunner, build_lm_package, export_tinylm

CONFIG = LMConfig(vocab_size=12, max_seq_len=10, d_model=16, n_heads=4, n_layers=2, d_hidden=24)


@pytest.fixture(scope="module")
def trained_like():
    model = TinyLM(rng=np.random.default_rng(3), **CONFIG.__dict__)
    # Scale weights up so logits spread out, like a trained model's (random init gives near-uniform ones).
    for array in named_parameters(model).values():
        if array.ndim == 2:
            array *= 3.0
    windows = np.random.default_rng(4).integers(0, CONFIG.vocab_size, size=(6, CONFIG.max_seq_len))
    return model, windows


def _reference(model):
    return lambda windows: model(windows).data


def test_graph_matches_the_checkpoint_and_the_cache_matches_a_full_pass(trained_like):
    model, windows = trained_like
    runner = LMRunner(export_tinylm(named_parameters(model), CONFIG), CONFIG)
    full = runner.full(windows)
    assert np.abs(full - model(windows).data).max() < 1e-4
    for prompt in (1, 3, CONFIG.max_seq_len - 1):
        assert np.abs(runner.incremental(windows[0], prompt) - full[0]).max() < 1e-4


def test_package_has_fp32_and_int8_with_measured_agreement(trained_like, tmp_path):
    model, windows = trained_like
    vocab = [chr(ord("a") + i) for i in range(CONFIG.vocab_size)]
    package = build_lm_package(
        named_parameters(model), CONFIG, vocab, _reference(model), windows, label="tiny", description="test"
    )
    manifest = package.manifest
    assert manifest.kind == "causal-lm" and set(manifest.assets) == {"tokenizer"}
    fp32, int8 = manifest.variants
    assert fp32.parity.action_agreement == 1.0
    assert 0.5 < int8.parity.action_agreement <= 1.0
    assert int8.requirements.backends == ["wasm"]
    assert int8.dtype == "int8"  # (smaller only at real sizes: quantization params dominate 16x16 matrices)
    store = LocalModelStore(tmp_path)
    store.put(package)
    assert store.manifest(manifest.package_id).config["n_layers"] == CONFIG.n_layers


def test_a_wrong_export_is_rejected(trained_like):
    model, windows = trained_like
    weights = dict(named_parameters(model))
    weights["head.bias"] = weights["head.bias"] + 1.0  # the reference below doesn't have this
    with pytest.raises(ValueError, match="differ"):
        build_lm_package(
            weights,
            CONFIG,
            list("abcdefghijkl"),
            _reference(model),
            windows,
            label="broken",
            description="",
            variants=("fp32",),
        )


def test_tokenizer_is_the_checkpoints():
    tokenizer = CharTokenizer("hello world")
    vocab = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    assert CharTokenizer("".join(vocab)).encode("hello") == tokenizer.encode("hello")


def test_shapes_and_small_tensors_stay_in_the_graph(trained_like):
    # ONNX Runtime Web resolves Reshape targets before attaching external data; a sharded int64 shape
    # makes session creation fail in the browser (never in Python, which inlines everything first).
    import onnx
    from onnx.external_data_helper import uses_external_data

    from modelpack.packaging import INLINE_BELOW_BYTES, shard_initializers

    model, _ = trained_like
    sharded, shards, _ = shard_initializers(export_tinylm(named_parameters(model), CONFIG))
    assert shards
    for tensor in sharded.graph.initializer:
        integer = tensor.data_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32)
        if uses_external_data(tensor):
            assert not integer
        else:
            assert integer or len(tensor.raw_data) < INLINE_BELOW_BYTES
    assert any(t.name == "const.heads_shape" and not uses_external_data(t) for t in sharded.graph.initializer)


def test_quantized_weights_are_sharded_not_left_in_the_graph(trained_like):
    from onnx.external_data_helper import uses_external_data

    from modelpack.lm import quantize_int8
    from modelpack.packaging import INLINE_BELOW_BYTES, shard_initializers

    model, _ = trained_like
    sharded, _, _ = shard_initializers(quantize_int8(export_tinylm(named_parameters(model), CONFIG)))
    big_int8 = [t for t in sharded.graph.initializer if t.name.endswith("_quantized")]
    assert big_int8
    for tensor in big_int8:
        assert uses_external_data(tensor) or len(tensor.raw_data) < INLINE_BELOW_BYTES

"""TinyLM (libs/tinylm, trained with libs/autodiff) → ONNX with a KV cache, and its package
(docs/design/0009 Decision 4, plan step 6).

Exported layer by layer rather than by tracing autodiff's graph: a trace records one fixed
(batch, seq_len) with the embedding lookups already resolved to concrete rows, which can neither take
a variable-length prompt nor carry a cache. Each tinylm layer maps to a handful of ONNX ops:

    Embedding -> Gather          LayerNorm -> LayerNormalization      Linear -> MatMul + Add
    attention -> Reshape/Transpose, Concat onto the cache, MatMul, causal mask, Softmax, MatMul
    MLP -> Linear, Relu, Linear

One graph serves both phases of generation:

    inputs   input_ids [batch, seq] int64, position_ids [batch, seq] int64,
             past_key_<i> / past_value_<i> [batch, heads, past, head_dim]   (past = 0 for a prompt)
    outputs  logits [batch, seq, vocab], present_key_<i> / present_value_<i> [batch, heads, past + seq, head_dim]

The causal mask is built in-graph from the cache length: query j (absolute position past + j) may see
keys 0..past + j. Positions are inputs, not derived, because TinyLM's positional embeddings are
learned up to max_seq_len -- a client that outgrows the window recomputes from scratch on the last
max_seq_len tokens, exactly like `tinylm.generate` does.

Parity for a language model: max |logit error| against the checkpoint's own float64 forward pass, a
check that decoding token by token through the cache gives the same logits as one full pass, and
top-1 agreement (does the variant predict the same next character?) at every position of held-out
text, decoded the way a client generates -- the LM counterpart of a policy's action agreement.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from modelpack.exporters import IR_VERSION, OPSET
from modelpack.manifest import Blob, ModelManifest, Parity, Provenance, TensorSpec, Variant
from modelpack.packaging import Package, _versions, requirements_for, sha256, shard_initializers, with_parity
from modelpack.runtime import inline_external_data, session_options

MASKED = -1e9  # what tinylm adds above the diagonal; exp() of it underflows to exactly 0
PROTOCOL = "tinylm.held-out-top1.v1"
FIXTURE_WINDOWS = 4
AGREEMENT_WINDOWS = 64  # held-out windows decoded token by token for the agreement number


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    n_layers: int
    d_hidden: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def export_tinylm(weights: dict[str, np.ndarray], config: LMConfig) -> onnx.ModelProto:
    """float32 graph from named weights (`tinylm.checkpoint.named_parameters`)."""
    c = config
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def init(name: str, array: np.ndarray, dtype=np.float32) -> str:
        inits.append(numpy_helper.from_array(np.asarray(array, dtype=dtype), name))
        return name

    def node(op: str, inputs: list[str], output: str, **attrs) -> str:
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        return output

    def linear(x: str, prefix: str) -> str:
        mm = node("MatMul", [x, init(f"{prefix}.weight", weights[f"{prefix}.weight"])], f"{prefix}.mm")
        return node("Add", [mm, init(f"{prefix}.bias", weights[f"{prefix}.bias"])], f"{prefix}.out")

    def layer_norm(x: str, prefix: str) -> str:
        return node(
            "LayerNormalization",
            [x, init(f"{prefix}.gamma", weights[f"{prefix}.gamma"]), init(f"{prefix}.beta", weights[f"{prefix}.beta"])],
            f"{prefix}.out",
            axis=-1,
            epsilon=1e-5,
        )

    # Constants shared by every layer.
    heads_shape = init("const.heads_shape", [0, 0, c.n_heads, c.head_dim], np.int64)
    merged_shape = init("const.merged_shape", [0, 0, c.d_model], np.int64)
    scale = init("const.scale", 1.0 / np.sqrt(c.head_dim))
    masked = init("const.masked", MASKED)
    zero_f = init("const.zero", 0.0)
    zero_i = init("const.zero_i", 0, np.int64)
    one_i = init("const.one_i", 1, np.int64)
    axis0 = init("const.axis0", [0], np.int64)
    axis1 = init("const.axis1", [1], np.int64)

    # Causal mask for this call: [seq, past + seq], MASKED where key position > query position.
    past_len = node("Gather", [node("Shape", ["past_key_0"], "mask.past_shape"), init("const.i2", 2, np.int64)], "mask.past_len", axis=0)
    seq_len = node("Gather", [node("Shape", ["input_ids"], "mask.ids_shape"), one_i], "mask.seq_len", axis=0)
    total = node("Add", [past_len, seq_len], "mask.total")
    query_pos = node("Unsqueeze", [node("Range", [past_len, total, one_i], "mask.query_range"), axis1], "mask.query_pos")
    key_pos = node("Unsqueeze", [node("Range", [zero_i, total, one_i], "mask.key_range"), axis0], "mask.key_pos")
    future = node("Greater", [key_pos, query_pos], "mask.future")
    mask = node("Where", [future, masked, zero_f], "mask")

    tokens = node("Gather", [init("token_embedding.weight", weights["token_embedding.weight"]), "input_ids"], "embed.tokens", axis=0)
    positions = node("Gather", [init("position_embedding.weight", weights["position_embedding.weight"]), "position_ids"], "embed.positions", axis=0)
    x = node("Add", [tokens, positions], "embed.out")

    for i in range(c.n_layers):
        p = f"blocks.{i}"
        h = layer_norm(x, f"{p}.ln1")

        def heads(name: str, _h: str = h, _p: str = p) -> str:
            projected = linear(_h, f"{_p}.attn.{name}")
            split = node("Reshape", [projected, heads_shape], f"{_p}.attn.{name}.split")
            return node("Transpose", [split], f"{_p}.attn.{name}.heads", perm=[0, 2, 1, 3])

        q, k, v = heads("query"), heads("key"), heads("value")
        present_k = node("Concat", [f"past_key_{i}", k], f"present_key_{i}", axis=2)
        present_v = node("Concat", [f"past_value_{i}", v], f"present_value_{i}", axis=2)
        k_t = node("Transpose", [present_k], f"{p}.attn.key_t", perm=[0, 1, 3, 2])
        scores = node("Mul", [node("MatMul", [q, k_t], f"{p}.attn.qk"), scale], f"{p}.attn.scaled")
        weights_ = node("Softmax", [node("Add", [scores, mask], f"{p}.attn.masked")], f"{p}.attn.weights", axis=-1)
        attended = node("MatMul", [weights_, present_v], f"{p}.attn.attended")
        merged = node("Reshape", [node("Transpose", [attended], f"{p}.attn.merge_t", perm=[0, 2, 1, 3]), merged_shape], f"{p}.attn.merged")
        x = node("Add", [x, linear(merged, f"{p}.attn.out_proj")], f"{p}.residual1")

        hidden = node("Relu", [linear(layer_norm(x, f"{p}.ln2"), f"{p}.mlp.fc1")], f"{p}.mlp.relu")
        x = node("Add", [x, linear(hidden, f"{p}.mlp.fc2")], f"{p}.residual2")

    logits = linear(layer_norm(x, "ln_final"), "head")
    nodes.append(helper.make_node("Identity", [logits], ["logits"], name="logits"))

    cache_dims = lambda n: ["batch", c.n_heads, n, c.head_dim]  # noqa: E731
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "seq"]),
    ]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "seq", c.vocab_size])]
    for i in range(c.n_layers):
        for kind in ("key", "value"):
            inputs.append(helper.make_tensor_value_info(f"past_{kind}_{i}", TensorProto.FLOAT, cache_dims("past")))
            outputs.append(helper.make_tensor_value_info(f"present_{kind}_{i}", TensorProto.FLOAT, cache_dims("total")))
    graph = helper.make_graph(nodes, "tinylm", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="modelpack")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def quantize_int8(model: onnx.ModelProto) -> onnx.ModelProto:
    """Dynamic int8 quantization of every MatMul's weights (activations quantized per call). Runs on the
    WASM/CPU backend; embeddings stay float."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)  # it logs a generic "consider pre-processing" warning on every call
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "f32.onnx", Path(tmp) / "int8.onnx"
            onnx.save_model(model, src)
            quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
            quantized = onnx.load_model(dst)
    finally:
        root.setLevel(level)
    quantized.ir_version = max(quantized.ir_version, IR_VERSION)
    return quantized


class LMRunner:
    """One variant in ONNX Runtime CPU, with the cache plumbing a generation loop needs."""

    def __init__(self, model: onnx.ModelProto, config: LMConfig):
        self.config = config
        self.session = ort.InferenceSession(model.SerializeToString(), sess_options=session_options(), providers=["CPUExecutionProvider"])
        self.float = np.float16 if model.graph.input[2].type.tensor_type.elem_type == TensorProto.FLOAT16 else np.float32

    def empty_cache(self, batch: int = 1) -> dict[str, np.ndarray]:
        shape = (batch, self.config.n_heads, 0, self.config.head_dim)
        return {f"past_{kind}_{i}": np.zeros(shape, self.float) for i in range(self.config.n_layers) for kind in ("key", "value")}

    def run(self, ids: np.ndarray, positions: np.ndarray, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        names = [o.name for o in self.session.get_outputs()]
        results = self.session.run(names, {"input_ids": ids.astype(np.int64), "position_ids": positions.astype(np.int64), **cache})
        by_name = dict(zip(names, results, strict=True))
        present = {n.replace("present_", "past_"): v for n, v in by_name.items() if n.startswith("present_")}
        return by_name["logits"].astype(np.float64), present

    def full(self, windows: np.ndarray) -> np.ndarray:
        positions = np.broadcast_to(np.arange(windows.shape[1]), windows.shape)
        return self.run(windows, positions, self.empty_cache(len(windows)))[0]

    def incremental(self, window: np.ndarray, prompt: int) -> np.ndarray:
        """Logits for `window` computed as a `prompt`-token prefill, then one token at a time through the cache."""
        logits, cache = self.run(window[None, :prompt], np.arange(prompt)[None], self.empty_cache())
        rows = [logits[0]]
        for t in range(prompt, len(window)):
            step, cache = self.run(window[None, t : t + 1], np.array([[t]]), cache)
            rows.append(step[0])
        return np.concatenate(rows)


VARIANT_TOLERANCE = {"fp32": 1e-3, "int8": 5.0}  # |logit| error gates: catch a broken export, not rounding


def build_lm_package(
    weights: dict[str, np.ndarray],
    config: LMConfig,
    vocab: Sequence[str],
    reference_logits,
    held_out_windows: np.ndarray,
    *,
    label: str,
    description: str,
    variants: Sequence[str] = ("fp32", "int8"),
    provenance: dict | None = None,
    shard_bytes: int | None = None,
) -> Package:
    """Package a TinyLM checkpoint. `reference_logits(windows) -> [n, seq, vocab]` is the checkpoint's
    own float64 forward pass; `held_out_windows` ([n, max_seq_len] token ids) is text it never trained
    on, used for every parity number."""
    f32 = export_tinylm(weights, config)
    graphs = {"fp32": f32}
    if "int8" in variants:
        graphs["int8"] = quantize_int8(f32)
    reference = reference_logits(held_out_windows)
    expected_top1 = reference.argmax(-1)

    blobs: dict[str, bytes] = {}
    built: list[Variant] = []
    for variant_id in variants:
        kwargs = {} if shard_bytes is None else {"shard_bytes": shard_bytes}
        sharded, shards, contents = shard_initializers(graphs[variant_id], **kwargs)
        blobs.update(contents)
        graph = sharded.SerializeToString()
        blobs[sha256(graph)] = graph
        weight_bytes = sum(len(v) for v in contents.values())
        max_tensor = max(len(t.raw_data) for t in graphs[variant_id].graph.initializer)
        requirements = requirements_for("float32", weight_bytes, max_tensor, len(graph))
        if variant_id == "int8":
            # MatMulInteger/DynamicQuantizeLinear: CPU kernels only (ORT's WebGPU EP has none).
            requirements = requirements.model_copy(update={"backends": ["wasm"]})
        cache_spec = lambda kind, i, n: TensorSpec(name=f"{kind}_{i}", dtype="float32", shape=["batch", config.n_heads, n, config.head_dim])  # noqa: E731
        built.append(
            Variant(
                id=variant_id,
                dtype="int8" if variant_id == "int8" else "float32",
                inputs=[
                    TensorSpec(name="input_ids", dtype="int64", shape=["batch", "seq"]),
                    TensorSpec(name="position_ids", dtype="int64", shape=["batch", "seq"]),
                    *[cache_spec(f"past_{kind}", i, "past") for i in range(config.n_layers) for kind in ("key", "value")],
                ],
                outputs=[
                    TensorSpec(name="logits", dtype="float32", shape=["batch", "seq", config.vocab_size]),
                    *[cache_spec(f"present_{kind}", i, "total") for i in range(config.n_layers) for kind in ("key", "value")],
                ],
                graph=Blob(sha256=sha256(graph), bytes=len(graph)),
                shards=shards,
                requirements=requirements,
                parity=Parity(reference="tinylm.TinyLM (float64, autodiff)", samples=0, max_abs_error=0.0, tolerance=VARIANT_TOLERANCE[variant_id]),
            )
        )

    tokenizer = json.dumps({"type": "char", "vocab": list(vocab)}).encode()
    blobs[sha256(tokenizer)] = tokenizer
    fixture_windows = held_out_windows[:FIXTURE_WINDOWS]
    fixture = json.dumps(
        {"input_ids": fixture_windows.tolist(), "logits": np.round(reference[:FIXTURE_WINDOWS], 6).tolist()}
    ).encode()
    blobs[sha256(fixture)] = fixture
    manifest = ModelManifest(
        kind="causal-lm",
        label=label,
        description=description,
        interface="text/char.v1",
        parameters=int(sum(w.size for w in weights.values())),
        provenance=Provenance(trainer="tinylm (autodiff)", source_format="tinylm.npz", exported_with=_versions(), **(provenance or {})),
        config={**config.__dict__, "head_dim": config.head_dim, "masked": MASKED},
        assets={"tokenizer": Blob(sha256=sha256(tokenizer), bytes=len(tokenizer))},
        variants=built,
        parity_fixture=Blob(sha256=sha256(fixture), bytes=len(fixture)),
    )
    package = Package(manifest=manifest, blobs=blobs)

    # Check what was packaged -- shards and all -- per variant.
    for variant in built:
        restored = inline_external_data(blobs[variant.graph.sha256], blobs.__getitem__, {s.path: s.sha256 for s in variant.shards})
        runner = LMRunner(restored, config)
        logits = runner.full(held_out_windows)
        error = float(np.abs(logits - reference).max())
        if not error <= variant.parity.tolerance:
            raise ValueError(f"{label} ({variant.id}): logits differ from the checkpoint by {error:.3g} (> {variant.parity.tolerance})")
        # Agreement is measured the way a client generates -- a prompt, then one token at a time through
        # the cache -- because that's what visitors see. For float variants that equals a full pass
        # (checked); int8 quantizes activations per call, so a one-token call legitimately rounds
        # differently from a full pass, and only the generation-mode numbers describe it.
        prompt = max(1, config.max_seq_len // 4)
        sampled = held_out_windows[:AGREEMENT_WINDOWS]
        generated = np.stack([runner.incremental(w, prompt) for w in sampled])
        if variant.dtype == "float32":
            cache_error = float(np.abs(generated - logits[: len(sampled)]).max())
            if not cache_error <= 1e-3:
                raise ValueError(f"{label} ({variant.id}): decoding through the cache differs from a full pass by {cache_error:.3g}")
        agree = float((generated.argmax(-1) == expected_top1[: len(sampled)]).mean())
        package = with_parity(
            package,
            variant.id,
            samples=int(held_out_windows.size),
            max_abs_error=error,
            protocol=PROTOCOL,
            decisions=int(expected_top1[:AGREEMENT_WINDOWS].size),
            action_agreement=round(agree, 6),
        )
    return package

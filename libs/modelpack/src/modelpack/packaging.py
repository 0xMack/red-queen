"""Exported graph → sealed package (docs/design/0009 Decision 4): per-variant graphs with weights moved
into content-addressed shards, a numeric parity check against the trainer's own forward pass, device
requirements, and a self-test fixture -- the same for every trainer."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import set_external_data

from modelpack.exporters import INPUT, OUTPUT, Exported
from modelpack.manifest import (
    Blob,
    ModelManifest,
    Parity,
    Provenance,
    Requirements,
    TensorSpec,
    Variant,
    WeightShard,
)
from modelpack.runtime import PackagedModel

DEFAULT_SHARD_BYTES = 32 * 1024 * 1024  # small enough to resume cheaply, big enough to keep request counts low
SHARD_ALIGNMENT = 64
INLINE_BELOW_BYTES = 1024  # the onnx library's own default for save_as_external_data
# Per-dtype gate on max |exported - reference|. It catches a *wrong* export (a transposed matrix, a
# missing bias: errors of order 0.1), not rounding; whether rounding changes decisions is what the
# publishing job's action-agreement check measures. fp32 against a float64 reference lands around
# 1e-7..1e-5 (it grows with fan-in and weight magnitude: a 100-input grid champion hit 1.1e-5).
TOLERANCE = {"float64": 1e-9, "float32": 1e-4}
VARIANT_IDS = {"float64": "fp64", "float32": "fp32"}
# Below this many weight bytes, a GPU's dispatch overhead outweighs its math -- prefer the WASM backend.
SMALL_MODEL_BYTES = 4 * 1024 * 1024
# The WASM backend's heap is 32-bit (4 GB) and also holds activations and the runtime itself.
WASM_MAX_WEIGHT_BYTES = 1536 * 1024 * 1024
FIXTURE_SAMPLES = 32


@dataclass(frozen=True)
class Package:
    manifest: ModelManifest
    blobs: dict[str, bytes]  # sha256 -> content, every blob the manifest refers to


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(data: bytes) -> Blob:
    return Blob(sha256=sha256(data), bytes=len(data))


def stays_inline(tensor: onnx.TensorProto) -> bool:
    """Small tensors and integer tensors stay in the graph. Integer initializers are shapes, indexes and
    axes that ONNX Runtime Web needs *while resolving the graph* (a Reshape's target shape), before
    external data is attached -- sharding one fails session creation with "Cannot parse data from
    external tensors" (found running TinyLM in the browser; the Python runtime inlines everything first,
    so it never saw this). Small float tensors aren't worth a lookup."""
    if tensor.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16):
        return True
    return len(tensor.raw_data) < INLINE_BELOW_BYTES


def shard_initializers(model: onnx.ModelProto, shard_bytes: int = DEFAULT_SHARD_BYTES) -> tuple[onnx.ModelProto, list[WeightShard], dict[str, bytes]]:
    """Move every weight initializer into external shard files named `weights/<sha256>.bin` (small and
    integer tensors stay inline, `stays_inline`). Tensors are never split across shards; one larger than
    `shard_bytes` gets a shard of its own."""
    model = onnx.ModelProto.FromString(model.SerializeToString())  # don't mutate the caller's
    groups: list[list[onnx.TensorProto]] = [[]]
    sizes = [0]
    for tensor in model.graph.initializer:
        if not tensor.HasField("raw_data"):
            # Some producers (e.g. ONNX Runtime's quantizer) write small tensors as typed fields;
            # normalize so every initializer is handled the same way.
            tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor), tensor.name))
        if stays_inline(tensor):
            continue
        size = len(tensor.raw_data)
        if groups[-1] and sizes[-1] + size > shard_bytes:
            groups.append([])
            sizes.append(0)
        groups[-1].append(tensor)
        sizes[-1] += size + (-size % SHARD_ALIGNMENT)
    shards: list[WeightShard] = []
    contents: dict[str, bytes] = {}
    for group in groups:
        if not group:
            continue
        chunks, placements, offset = [], [], 0
        for tensor in group:
            data = tensor.raw_data
            placements.append((tensor, offset, len(data)))
            padding = -len(data) % SHARD_ALIGNMENT
            chunks += [data, b"\0" * padding]
            offset += len(data) + padding
        content = b"".join(chunks)
        digest = sha256(content)
        path = f"weights/{digest}.bin"
        for tensor, start, length in placements:
            set_external_data(tensor, path, offset=start, length=length)
            tensor.ClearField("raw_data")
            tensor.data_location = onnx.TensorProto.EXTERNAL
        shards.append(WeightShard(sha256=digest, bytes=len(content), path=path))
        contents[digest] = content
    return model, shards, contents


def sample_inputs(num_inputs: int, count: int = 256, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(num_inputs)] for _ in range(count)]


def requirements_for(dtype: str, weight_bytes: int, max_tensor_bytes: int, graph_bytes: int) -> Requirements:
    if dtype == "float64":
        backends = ["wasm"]  # WGSL has no f64: a float64 graph is CPU/WASM-only
    elif weight_bytes > WASM_MAX_WEIGHT_BYTES:
        backends = ["webgpu"]
    elif weight_bytes < SMALL_MODEL_BYTES:
        backends = ["wasm", "webgpu"]
    else:
        backends = ["webgpu", "wasm"]
    return Requirements(
        backends=backends,
        download_bytes=graph_bytes + weight_bytes,
        max_tensor_bytes=max_tensor_bytes,
        # Rough: weights once in the runtime + once in transit/staging, plus graph and working space.
        peak_memory_bytes=2 * weight_bytes + graph_bytes + 16 * 1024 * 1024,
    )


def _versions() -> dict[str, str]:
    found = {}
    for dist in ("modelpack", "onnx", "onnxruntime", "numpy"):
        try:
            found[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def _variant(exported: Exported, shard_bytes: int, blobs: dict[str, bytes]) -> Variant:
    model, shards, contents = shard_initializers(exported.model, shard_bytes)
    blobs.update(contents)
    graph = model.SerializeToString()
    blobs[sha256(graph)] = graph
    weight_bytes = sum(len(c) for c in contents.values())
    max_tensor = max((len(t.raw_data) for t in exported.model.graph.initializer), default=0)
    return Variant(
        id=VARIANT_IDS[exported.dtype],
        dtype=exported.dtype,
        inputs=[TensorSpec(name=INPUT, dtype=exported.dtype, shape=["batch", exported.num_inputs])],
        outputs=[TensorSpec(name=OUTPUT, dtype=exported.dtype, shape=["batch", exported.num_outputs])],
        graph=_blob(graph),
        shards=shards,
        requirements=requirements_for(exported.dtype, weight_bytes, max_tensor, len(graph)),
        parity=Parity(reference=exported.reference_name, samples=0, max_abs_error=0.0, tolerance=TOLERANCE[exported.dtype]),
    )


def build_package(
    exports: Exported | Sequence[Exported],
    *,
    label: str,
    interface: str | None = None,
    run_id: str | None = None,
    champion_ref: str | None = None,
    samples: Sequence[Sequence[float]] | None = None,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> Package:
    """Package one or more exports of the same network -- one variant each, most exact first -- and
    check every *packaged* variant (shards, offsets and all) against the reference forward pass on
    `samples` (real observations if the caller has them, random ones otherwise). Raises if any
    variant is numerically wrong. Quantized variants (fp16/int8/int4) slot in the same way once a
    model is big enough to need them."""
    exports = [exports] if isinstance(exports, Exported) else list(exports)
    first = exports[0]
    if samples is None:
        samples = sample_inputs(first.num_inputs)
    samples = [list(s) for s in samples]
    reference = np.asarray([first.reference(s) for s in samples], dtype=np.float64)

    blobs: dict[str, bytes] = {}
    variants = [_variant(e, shard_bytes, blobs) for e in exports]
    fixture = json.dumps(
        {
            "input": INPUT,
            "output": OUTPUT,
            "inputs": samples[:FIXTURE_SAMPLES],
            "outputs": reference[:FIXTURE_SAMPLES].tolist(),
        }
    ).encode()
    blobs[sha256(fixture)] = fixture
    package = Package(
        manifest=ModelManifest(
            label=label,
            description=first.description,
            interface=interface,
            parameters=first.parameters,
            provenance=Provenance(
                trainer=first.trainer,
                source_format=first.source_format,
                run_id=run_id,
                champion_ref=champion_ref,
                exported_with=_versions(),
            ),
            variants=variants,
            parity_fixture=_blob(fixture),
        ),
        blobs=blobs,
    )
    for variant in variants:
        loaded = PackagedModel(package.manifest, variant.id, blobs.__getitem__)
        actual = loaded.run(np.asarray(samples)).astype(np.float64)
        error = float(np.max(np.abs(actual - reference))) if samples else 0.0
        if not error <= variant.parity.tolerance:
            raise ValueError(
                f"{label} ({variant.id}): exported graph differs from {variant.parity.reference} by {error:.3g} "
                f"(> {variant.parity.tolerance})"
            )
        package = with_parity(package, variant.id, samples=len(samples), max_abs_error=error)
    return package


def with_parity(package: Package, variant_id: str, **fields) -> Package:
    """A new (re-sealed, so re-addressed) package with extra parity facts for one variant -- how the
    publishing job adds the action-agreement numbers it measured on a protocol's games."""
    manifest = package.manifest.model_copy(deep=True)
    variant = manifest.variant(variant_id)
    variant.parity = variant.parity.model_copy(update=fields)
    return Package(manifest=manifest.sealed(), blobs=package.blobs)

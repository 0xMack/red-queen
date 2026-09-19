"""Running a package in Python with ONNX Runtime (CPU) -- what the evaluation job uses, so a leaderboard
score describes the exact artifact a visitor downloads (docs/design/0009 Decision 4), not the trainer's
in-memory object."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

from modelpack.manifest import ModelManifest

BlobReader = Callable[[str], bytes]  # sha256 -> bytes


def inline_external_data(graph: bytes, read_blob: BlobReader, shard_sha_by_path: dict[str, str]) -> onnx.ModelProto:
    """Parse a variant's graph and copy every external initializer's bytes back into it, so it can be
    handed to ONNX Runtime from memory. (Fine up to protobuf's 2 GB limit; a package bigger than that
    would need `SessionOptions.add_external_initializers` instead.)"""
    model = onnx.load_model_from_string(graph)
    cache: dict[str, bytes] = {}
    for tensor in model.graph.initializer:
        if not uses_external_data(tensor):
            continue
        info = ExternalDataInfo(tensor)
        sha = shard_sha_by_path[info.location]
        if sha not in cache:
            cache[sha] = read_blob(sha)
        start = info.offset or 0
        length = info.length if info.length is not None else len(cache[sha]) - start
        tensor.raw_data = cache[sha][start : start + length]
        tensor.data_location = onnx.TensorProto.DEFAULT
        del tensor.external_data[:]
    return model


def session_options(threads: int = 1) -> ort.SessionOptions:
    options = ort.SessionOptions()
    # One thread: evaluation plays games one decision at a time, where thread handoff costs more than
    # a small model's math; it also keeps timings comparable with the pure-Python forward pass.
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return options


class PackagedModel:
    """One variant of a package, loaded into an ONNX Runtime CPU session."""

    def __init__(self, manifest: ModelManifest, variant_id: str, read_blob: BlobReader, threads: int = 1):
        self.manifest = manifest
        self.variant = manifest.variant(variant_id)
        graph = read_blob(self.variant.graph.sha256)
        model = inline_external_data(graph, read_blob, {s.path: s.sha256 for s in self.variant.shards})
        self.session = ort.InferenceSession(
            model.SerializeToString(), sess_options=session_options(threads), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.variant.inputs[0].name
        self.output_name = self.variant.outputs[0].name
        self.input_dtype = np.dtype(self.variant.inputs[0].dtype)

    def run(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: np.asarray(batch, dtype=self.input_dtype)})[0]

    def forward(self, observation: Sequence[float]) -> list[float]:
        """Same shape as `WeightVector.forward`/`NeatGenome.forward`, so it drops into any existing
        policy wrapper."""
        return self.run(np.asarray([observation], dtype=self.input_dtype))[0].tolist()

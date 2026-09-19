"""The model package manifest (docs/design/0009 Decision 4): the small JSON a client fetches first, and
the only thing that decides what else it downloads.

Every file a package refers to is a *blob*, named by the sha256 of its bytes, so every URL is
immutable and a cache entry can never go stale. The manifest's own id (`package_id`) is the hash of
everything else in it, so a manifest is content-addressed too.

`apps/frontend/app/types/modelpack.ts` mirrors these models by hand -- keep them in sync.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

FORMAT_VERSION = 1

Backend = Literal["webgpu", "wasm"]


class Blob(BaseModel):
    """One content-addressed file."""

    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(..., ge=0)


class WeightShard(Blob):
    """An external-data file of a variant's graph. `path` is the location string the ONNX graph
    refers to it by (`weights/<sha256>.bin`) -- ONNX Runtime matches external data by that string, so
    a client hands each shard to the runtime under this exact name."""

    path: str


class TensorSpec(BaseModel):
    name: str
    dtype: Literal["float64", "float32", "float16", "int64", "int32", "uint8"]
    # A string is a symbolic dimension (e.g. "batch"), an int a fixed one.
    shape: list[int | str]


class Requirements(BaseModel):
    """What a device must offer to run a variant -- matched against a browser's probed capabilities
    (apps/frontend `matchVariant`) so an unsupported model says *why* instead of crashing."""

    backends: list[Backend] = Field(..., min_length=1, description="acceptable backends, most preferred first")
    webgpu_features: list[str] = Field(default_factory=list, description="e.g. 'shader-f16'")
    download_bytes: int = Field(..., ge=0, description="graph + every shard")
    max_tensor_bytes: int = Field(..., ge=0, description="largest single initializer (vs. a GPU's maxBufferSize)")
    peak_memory_bytes: int = Field(..., ge=0, description="rough estimate: weights + largest activations")


class Parity(BaseModel):
    """How far a variant is from its reference implementation (the trainer's own forward pass).
    Numeric error alone isn't what matters -- a flipped argmax is -- so the publishing job adds the
    action-agreement numbers from playing the evaluation protocol's games with both."""

    reference: str = Field(..., description="what the variant was compared against, e.g. 'evolve.neuro.WeightVector.forward (float64)'")
    samples: int
    max_abs_error: float
    tolerance: float
    protocol: str | None = None
    decisions: int | None = Field(None, description="decisions compared on the protocol's games")
    action_agreement: float | None = Field(None, description="fraction of identical decisions, 1.0 = every one")
    scores_match: bool | None = Field(None, description="every protocol game scored the same under both")


class Variant(BaseModel):
    """One runnable form of the model (fp64, fp32, fp16, int8, ...), with its own graph, weights and
    I/O dtypes. Tensor names and shapes are the same across a package's variants; dtypes aren't (an
    fp64 variant takes and returns float64)."""

    id: str
    dtype: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    graph: Blob
    shards: list[WeightShard]
    requirements: Requirements
    parity: Parity


class Provenance(BaseModel):
    trainer: str = Field(..., description="e.g. 'evolve.neuro', 'evolve.neat', 'autodiff', 'torch'")
    source_format: str = Field(..., description="e.g. 'weight_vector.json', 'neat.json'")
    run_id: str | None = None
    champion_ref: str | None = None
    exported_with: dict[str, str] = Field(default_factory=dict, description="tool -> version")


class ModelManifest(BaseModel):
    format_version: int = FORMAT_VERSION
    package_id: str = Field("", description="sha256 of this manifest with package_id empty")
    kind: Literal["policy", "causal-lm"] = Field("policy", description="how a client drives it: one decision per call, or token-by-token generation with a KV cache")
    label: str
    description: str
    interface: str | None = Field(None, description="docs/design/0007 interface id the model runs under")
    parameters: int
    provenance: Provenance
    variants: list[Variant] = Field(..., min_length=1)
    parity_fixture: Blob | None = Field(None, description="JSON a client can self-test against: {inputs, outputs} for a policy, {input_ids, logits} for a causal LM")
    config: dict[str, Any] = Field(default_factory=dict, description="architecture facts a client needs to drive the model (e.g. an LM's layers, heads, max_seq_len)")
    assets: dict[str, Blob] = Field(default_factory=dict, description="other files the model needs, by role (e.g. 'tokenizer')")

    def variant(self, variant_id: str) -> Variant:
        for v in self.variants:
            if v.id == variant_id:
                return v
        raise KeyError(f"no variant {variant_id!r} in package {self.package_id} (has {[v.id for v in self.variants]})")

    def blobs(self) -> list[Blob]:
        found: list[Blob] = [b for v in self.variants for b in (v.graph, *v.shards)]
        if self.parity_fixture:
            found.append(self.parity_fixture)
        found.extend(self.assets.values())
        return found

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def sealed(self) -> ModelManifest:
        """A copy whose package_id is the hash of its content. Any change -- a weight, a parity
        number, a label -- is a different package."""
        unsealed = self.model_copy(update={"package_id": ""})
        digest = hashlib.sha256(unsealed.canonical_json().encode()).hexdigest()
        return self.model_copy(update={"package_id": digest})

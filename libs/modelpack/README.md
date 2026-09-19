# modelpack

Portable, content-addressed model packages for client-side inference ([docs/design/0009](../../docs/design/0009-client-side-inference-at-scale.md)).

A package is a sealed `manifest.json` plus blobs named by their sha256: one ONNX graph per variant
(`fp32` today), its weights moved into external shard files, and a small parity fixture a browser can
self-test against. Every trainer exports to this one format; the browser (ONNX Runtime Web) and the
evaluation job (`PackagedModel`, ONNX Runtime CPU) run the same bytes.

- `exporters` — `WeightVector` → `Gemm/Tanh` chain; `NeatGenome` → one dense `Gemm/Tanh/Concat` per
  depth group (skip connections read from the running concatenation), outputs gathered at the end.
- `packaging` — sharding, hashing, device requirements, and a parity gate: the *packaged* artifact
  must match the trainer's own float64 forward pass within tolerance, or publishing fails.
- `store` — `ModelStore` protocol + `LocalModelStore` (`blobs/`, `manifests/`, `catalog/<game>.json`).
  Only the catalog is mutable.
- `runtime` — load a package variant into an ONNX Runtime CPU session.

Publishing (export → verify → action agreement on the protocol's games → store → catalog) is
`jobs/publish_models.py`, since it plays games. Depends on `evolve`, never the other way round.

Tests: `uv run pytest libs/modelpack/tests`.

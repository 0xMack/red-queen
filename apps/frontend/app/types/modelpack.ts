// Mirrors libs/modelpack/src/modelpack/manifest.py and store.py's CatalogEntry (docs/design/0009) --
// keep in sync by hand.

export type Backend = "webgpu" | "wasm"
export type TensorDtype = "float64" | "float32" | "float16" | "int64" | "int32" | "uint8"

export interface Blob {
  sha256: string
  bytes: number
}

export interface WeightShard extends Blob {
  // The location string the ONNX graph refers to this shard by -- ORT matches external data on it.
  path: string
}

export interface TensorSpec {
  name: string
  dtype: TensorDtype
  shape: (number | string)[]
}

export interface Requirements {
  backends: Backend[] // most preferred first
  webgpu_features: string[]
  download_bytes: number
  max_tensor_bytes: number
  peak_memory_bytes: number
}

export interface Parity {
  reference: string
  samples: number
  max_abs_error: number
  tolerance: number
  protocol: string | null
  decisions: number | null
  action_agreement: number | null
  scores_match: boolean | null
}

export interface Variant {
  id: string
  dtype: string
  inputs: TensorSpec[]
  outputs: TensorSpec[]
  graph: Blob
  shards: WeightShard[]
  requirements: Requirements
  parity: Parity
}

export interface ModelManifest {
  format_version: number
  package_id: string
  kind: "policy" | "causal-lm"
  label: string
  description: string
  interface: string | null
  parameters: number
  provenance: {
    trainer: string
    source_format: string
    run_id: string | null
    champion_ref: string | null
    exported_with: Record<string, string>
  }
  variants: Variant[]
  parity_fixture: Blob | null
  config: Record<string, unknown>
  assets: Record<string, Blob>
}

// manifest.config of a causal LM (modelpack/lm.py).
export interface LMConfig {
  vocab_size: number
  max_seq_len: number
  d_model: number
  n_heads: number
  n_layers: number
  d_hidden: number
  head_dim: number
}

export interface CatalogEntry {
  entrant_id: string
  package_id: string
  label: string
  interface: string | null
  run_id: string | null
  champion_ref: string | null
  variants: string[] // the variants that play as this entrant, most exact first
  download_bytes: Record<string, number>
}

export interface Catalog {
  game: string
  base_url: string // where blobs/<sha> and manifests/<id>.json live
  entries: CatalogEntry[]
}

// What the session worker needs to load a package: everything is resolved on the main thread (so the
// UI can explain a mismatch before anything downloads), the worker only fetches and runs.
export interface ModelSpec {
  baseUrl: string
  manifest: ModelManifest
  variantId: string
  backend: Backend
}

// Whether a leaderboard entrant can run on this device, with a one-line note (the variant/backend it
// would use, or why it can't).
export interface DeviceFit {
  ok: boolean
  note: string
}

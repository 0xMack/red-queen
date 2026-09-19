// The model runtime (docs/design/0009 Decision 3): load one variant of a package into ONNX Runtime
// Web and run it. Only the ORT build the chosen backend needs is downloaded (WASM-only ≈3.7 MB gzipped,
// WebGPU ≈6.6 MB), lazily, the first time a model needs it -- and like every other asset it's
// versioned, so it's cached after that.
//
// Runs inside the session worker (app/workers/snakeGame.worker.ts), next to the game, so a decision
// costs no postMessage. Pages never import this directly; they resolve *which* variant/backend on the
// main thread (app/inference/match.ts) and hand the worker a ModelSpec.

import type * as OrtNamespace from "onnxruntime-web"
import ortWasmMjs from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url"
import ortWasm from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url"
import ortWebGpuMjs from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url"
import ortWebGpuWasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url"
import { fetchBlob, fetchBlobs } from "~/inference/blobs"
import type { Backend, ModelSpec, TensorDtype } from "~/types/modelpack"

type Ort = typeof OrtNamespace

// Below this download size a model runs single-threaded: waking worker threads costs more than a
// tiny matmul (and threads need cross-origin isolation anyway).
const THREADS_ABOVE_BYTES = 4 * 1024 * 1024

const runtimes = new Map<Backend, Promise<Ort>>()

function loadOrt(backend: Backend, downloadBytes: number): Promise<Ort> {
  let runtime = runtimes.get(backend)
  if (!runtime) {
    runtime = (async () => {
      const ort = (backend === "webgpu" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm")) as Ort
      ort.env.wasm.wasmPaths = backend === "webgpu" ? { mjs: ortWebGpuMjs, wasm: ortWebGpuWasm } : { mjs: ortWasmMjs, wasm: ortWasm }
      ort.env.wasm.proxy = false // already off the main thread
      ort.env.logLevel = "error"
      return ort
    })()
    runtimes.set(backend, runtime)
  }
  return runtime.then((ort) => {
    // Settable only before the first session; later sessions keep what the first one got.
    const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true
    ort.env.wasm.numThreads = isolated && downloadBytes > THREADS_ABOVE_BYTES ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
    return ort
  })
}

const ARRAYS = {
  float64: Float64Array,
  float32: Float32Array,
  int32: Int32Array,
  uint8: Uint8Array,
} as const

export interface LoadedModel {
  backend: Backend
  variantId: string
  loadMs: number
  selfTest: { samples: number; maxAbsError: number; tolerance: number } | null
  // One batch of rows in, one batch of rows out (plain numbers: the game side speaks JSON-ish arrays).
  run(rows: number[][]): Promise<number[][]>
  release(): Promise<void>
}

export interface LoadProgress {
  stage: "runtime" | "download" | "compile" | "self-test"
  loaded?: number
  total?: number
}

export async function loadModel(spec: ModelSpec, onProgress?: (p: LoadProgress) => void): Promise<LoadedModel> {
  const started = performance.now()
  const variant = spec.manifest.variants.find((v) => v.id === spec.variantId)
  if (!variant) throw new Error(`package ${spec.manifest.package_id.slice(0, 12)} has no variant ${spec.variantId}`)

  onProgress?.({ stage: "runtime" })
  const ort = await loadOrt(spec.backend, variant.requirements.download_bytes)

  const blobs = [variant.graph, ...variant.shards]
  const [graph, ...shards] = await fetchBlobs(spec.baseUrl, blobs, (loaded, total) =>
    onProgress?.({ stage: "download", loaded, total }),
  )

  onProgress?.({ stage: "compile" })
  const session = await ort.InferenceSession.create(graph!, {
    executionProviders: [spec.backend],
    graphOptimizationLevel: "all",
    externalData: variant.shards.map((s, i) => ({ path: s.path, data: shards[i]! })),
  })

  const input = variant.inputs[0]!
  const output = variant.outputs[0]!
  const dtype = input.dtype as TensorDtype
  const ArrayType = ARRAYS[dtype as keyof typeof ARRAYS]
  if (!ArrayType) throw new Error(`unsupported input dtype ${dtype}`)
  const width = Number(input.shape[1])

  async function run(rows: number[][]): Promise<number[][]> {
    const flat = new ArrayType(rows.length * width)
    rows.forEach((row, r) => flat.set(row, r * width))
    const feeds = { [input.name]: new ort.Tensor(dtype as "float32", flat as Float32Array, [rows.length, width]) }
    const results = await session.run(feeds)
    const tensor = results[output.name]!
    const data = (await tensor.getData()) as ArrayLike<number>
    const cols = Number(tensor.dims[1])
    const out: number[][] = []
    for (let r = 0; r < rows.length; r++) out.push(Array.from({ length: cols }, (_, c) => Number(data[r * cols + c])))
    tensor.dispose()
    return out
  }

  // Self-test against the package's parity fixture: the trainer's own float64 outputs for real
  // observations. Catches a backend that loads but computes something else (driver bugs, precision
  // lost somewhere) before a visitor is shown a model playing differently from its leaderboard entry.
  let selfTest: LoadedModel["selfTest"] = null
  if (spec.manifest.parity_fixture) {
    onProgress?.({ stage: "self-test" })
    const fixture = JSON.parse(new TextDecoder().decode(await fetchBlob(spec.baseUrl, spec.manifest.parity_fixture))) as {
      inputs: number[][]
      outputs: number[][]
    }
    const actual = await run(fixture.inputs)
    let maxAbsError = 0
    actual.forEach((row, r) => row.forEach((v, c) => (maxAbsError = Math.max(maxAbsError, Math.abs(v - fixture.outputs[r]![c]!)))))
    selfTest = { samples: fixture.inputs.length, maxAbsError, tolerance: variant.parity.tolerance }
    if (!(maxAbsError <= variant.parity.tolerance)) {
      await session.release()
      throw new Error(
        `self-test failed on ${spec.backend}: outputs differ from the reference by ${maxAbsError.toExponential(2)} (allowed ${variant.parity.tolerance})`,
      )
    }
  }

  return {
    backend: spec.backend,
    variantId: variant.id,
    loadMs: performance.now() - started,
    selfTest,
    run,
    release: () => session.release(),
  }
}

/// <reference lib="webworker" />

// A language model, generating off the main thread (docs/design/0009 plan step 6): loads a causal-LM
// package into ONNX Runtime (WASM or WebGPU, whichever the page matched) and streams tokens back.

import { cacheOutputs, generate, loadTokenizer, type CharTokenizer } from "~/inference/lm"
import { loadModel, type LoadedModel, type LoadProgress } from "~/inference/runtime"
import type { LMConfig, ModelSpec } from "~/types/modelpack"

declare const self: DedicatedWorkerGlobalScope

type Inbound =
  | { type: "load"; spec: ModelSpec }
  | { type: "generate"; prompt: string; maxNewTokens: number; temperature: number; seed: number }
  | { type: "stop" }

type Outbound =
  | { type: "model_progress"; progress: LoadProgress }
  | { type: "model_loaded"; backend: string; variantId: string; loadMs: number; selfTest: LoadedModel["selfTest"] }
  | { type: "model_error"; message: string; packageId: string; variantId: string; backend: string }
  | { type: "token"; text: string; ms: number; recomputed: boolean }
  | { type: "done"; tokens: number; ms: number; stopped: boolean }
  | { type: "error"; message: string }

let model: LoadedModel | null = null
let modelKey: string | null = null
let config: LMConfig | null = null
let tokenizer: CharTokenizer | null = null
let stopRequested = false
let busy = false

function post(message: Outbound) {
  self.postMessage(message)
}

async function load(spec: ModelSpec) {
  const key = `${spec.manifest.package_id}/${spec.variantId}/${spec.backend}`
  if (key === modelKey && model) {
    post({ type: "model_loaded", backend: model.backend, variantId: model.variantId, loadMs: 0, selfTest: model.selfTest })
    return
  }
  const previous = model
  model = null
  modelKey = null
  await previous?.release()
  config = spec.manifest.config as unknown as LMConfig
  try {
    tokenizer = await loadTokenizer(spec)
    model = await loadModel(spec, (progress) => post({ type: "model_progress", progress }), { gpuOutputs: cacheOutputs(config) })
  } catch (e) {
    post({
      type: "model_error",
      message: e instanceof Error ? e.message : String(e),
      packageId: spec.manifest.package_id,
      variantId: spec.variantId,
      backend: spec.backend,
    })
    return
  }
  modelKey = key
  post({ type: "model_loaded", backend: model.backend, variantId: model.variantId, loadMs: model.loadMs, selfTest: model.selfTest })
}

async function run(message: Extract<Inbound, { type: "generate" }>) {
  if (!model || !config || !tokenizer) throw new Error("no model loaded")
  if (busy) return
  busy = true
  stopRequested = false
  const started = performance.now()
  let tokens = 0
  try {
    for await (const token of generate(model, config, tokenizer, message.prompt, {
      maxNewTokens: message.maxNewTokens,
      temperature: message.temperature,
      seed: message.seed,
      shouldStop: () => stopRequested,
    })) {
      tokens += 1
      post({ type: "token", text: token.text, ms: token.ms, recomputed: token.recomputed })
    }
  } finally {
    busy = false
    post({ type: "done", tokens, ms: performance.now() - started, stopped: stopRequested })
  }
}

self.onmessage = (event: MessageEvent<Inbound>) => {
  const message = event.data
  const fail = (e: unknown) => post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  if (message.type === "load") load(message.spec).catch(fail)
  else if (message.type === "generate") run(message).catch(fail)
  else if (message.type === "stop") stopRequested = true
}

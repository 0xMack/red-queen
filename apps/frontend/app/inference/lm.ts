// Generating text with a causal-LM package (modelpack/lm.py, docs/design/0009 plan step 6) in ONNX
// Runtime: the prompt is one call with an empty cache, then one call per new token, each feeding the
// previous call's `present_*` cache back in as `past_*` -- under WebGPU those tensors never leave the
// GPU. Mirrors tinylm.generate's semantics exactly: positions are absolute, and once the text outgrows
// the learned positional embeddings (max_seq_len) every step recomputes the last max_seq_len
// characters from scratch, as the Python does.

import { fetchBlob } from "~/inference/blobs"
import type { LoadedModel, Ort, Session } from "~/inference/runtime"
import type { LMConfig, ModelSpec } from "~/types/modelpack"

type Tensor = InstanceType<Ort["Tensor"]>
type Cache = Record<string, Tensor>

export class CharTokenizer {
  readonly vocab: string[]
  private readonly index: Map<string, number>

  constructor(vocab: string[]) {
    this.vocab = vocab
    this.index = new Map(vocab.map((ch, i) => [ch, i]))
  }

  // Characters the model never saw (not in the corpus) are dropped, not guessed at.
  encode(text: string): number[] {
    return [...text].flatMap((ch) => (this.index.has(ch) ? [this.index.get(ch)!] : []))
  }

  decode(ids: number[]): string {
    return ids.map((i) => this.vocab[i] ?? "").join("")
  }

  unknown(text: string): string[] {
    return [...new Set([...text].filter((ch) => !this.index.has(ch)))]
  }
}

export async function loadTokenizer(spec: ModelSpec): Promise<CharTokenizer> {
  const asset = spec.manifest.assets.tokenizer
  if (!asset) throw new Error("package has no tokenizer")
  const parsed = JSON.parse(new TextDecoder().decode(await fetchBlob(spec.baseUrl, asset))) as { type: string; vocab: string[] }
  if (parsed.type !== "char") throw new Error(`unsupported tokenizer: ${parsed.type}`)
  return new CharTokenizer(parsed.vocab)
}

export function cacheOutputs(config: LMConfig): string[] {
  return Array.from({ length: config.n_layers }, (_, i) => [`present_key_${i}`, `present_value_${i}`]).flat()
}

function emptyCache(ort: Ort, config: LMConfig, batch = 1): Cache {
  const cache: Cache = {}
  for (let i = 0; i < config.n_layers; i++) {
    for (const kind of ["key", "value"]) {
      cache[`past_${kind}_${i}`] = new ort.Tensor("float32", new Float32Array(0), [batch, config.n_heads, 0, config.head_dim])
    }
  }
  return cache
}

function disposeCache(cache: Cache) {
  for (const tensor of Object.values(cache)) tensor.dispose()
}

// One call: `rows` (batch of equal-length token-id rows) at `positions`, on top of `cache`. Returns the
// logits (all positions, CPU) and the grown cache (the caller disposes the old one).
async function forward(ort: Ort, session: Session, config: LMConfig, rows: number[][], positions: number[], cache: Cache) {
  const seq = rows[0]!.length
  const ids = new BigInt64Array(rows.length * seq)
  const pos = new BigInt64Array(rows.length * seq)
  rows.forEach((row, r) =>
    row.forEach((id, t) => {
      ids[r * seq + t] = BigInt(id)
      pos[r * seq + t] = BigInt(positions[t]!)
    }),
  )
  const results = await session.run({
    input_ids: new ort.Tensor("int64", ids, [rows.length, seq]),
    position_ids: new ort.Tensor("int64", pos, [rows.length, seq]),
    ...cache,
  })
  const logits = (await results.logits!.getData()) as Float32Array
  results.logits!.dispose()
  const next: Cache = {}
  for (const name of cacheOutputs(config)) next[name.replace("present_", "past_")] = results[name]!
  return { logits, next, vocab: config.vocab_size, seq }
}

// Self-test: the checkpoint's own float64 logits for a few windows of held-out text.
export async function selfTestLM(ort: Ort, session: Session, spec: ModelSpec, tolerance: number) {
  const config = spec.manifest.config as unknown as LMConfig
  const fixture = JSON.parse(new TextDecoder().decode(await fetchBlob(spec.baseUrl, spec.manifest.parity_fixture!))) as {
    input_ids: number[][]
    logits: number[][][]
  }
  const seq = fixture.input_ids[0]!.length
  const cache = emptyCache(ort, config, fixture.input_ids.length)
  const { logits, next } = await forward(ort, session, config, fixture.input_ids, Array.from({ length: seq }, (_, t) => t), cache)
  disposeCache(next)
  let maxAbsError = 0
  fixture.logits.forEach((window, b) =>
    window.forEach((row, t) =>
      row.forEach((expected, v) => {
        maxAbsError = Math.max(maxAbsError, Math.abs(logits[(b * seq + t) * config.vocab_size + v]! - expected))
      }),
    ),
  )
  // The fixture is rounded to 6 decimals; allow for that on top of the variant's own tolerance.
  return { samples: fixture.input_ids.length * seq, maxAbsError, tolerance: tolerance + 1e-6 }
}

// Deterministic sampling (mulberry32): the same seed and prompt give the same text on the same backend.
function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function sample(logits: Float32Array, offset: number, vocab: number, temperature: number, random: () => number): number {
  if (temperature <= 0) {
    let best = 0
    for (let v = 1; v < vocab; v++) if (logits[offset + v]! > logits[offset + best]!) best = v
    return best
  }
  let max = -Infinity
  for (let v = 0; v < vocab; v++) max = Math.max(max, logits[offset + v]! / temperature)
  const probs = new Float64Array(vocab)
  let sum = 0
  for (let v = 0; v < vocab; v++) sum += probs[v] = Math.exp(logits[offset + v]! / temperature - max)
  let r = random() * sum
  for (let v = 0; v < vocab; v++) if ((r -= probs[v]!) <= 0) return v
  return vocab - 1
}

export interface GenerateOptions {
  maxNewTokens: number
  temperature: number
  seed: number
  shouldStop?: () => boolean
}

export interface Generated {
  id: number
  text: string
  ms: number // time to produce this token
  recomputed: boolean // past the context window: this step re-ran the whole window
}

export async function* generate(
  model: LoadedModel,
  config: LMConfig,
  tokenizer: CharTokenizer,
  prompt: string,
  options: GenerateOptions,
): AsyncGenerator<Generated> {
  const { ort, session } = model
  const random = rng(options.seed)
  const ids = tokenizer.encode(prompt)
  if (!ids.length) throw new Error("the prompt has no characters this model knows")
  let cache: Cache = emptyCache(ort, config)
  let cachedLength = 0 // how many of `ids` the cache covers, at positions 0..cachedLength-1
  try {
    for (let n = 0; n < options.maxNewTokens; n++) {
      if (options.shouldStop?.()) return
      const started = performance.now()
      let rows: number[]
      let positions: number[]
      const recompute = ids.length > config.max_seq_len
      if (recompute || cachedLength === 0) {
        // Prompt (or past the window): the whole context from scratch, positions from 0.
        disposeCache(cache)
        cache = emptyCache(ort, config)
        rows = ids.slice(-config.max_seq_len)
        positions = rows.map((_, t) => t)
      } else {
        rows = ids.slice(cachedLength)
        positions = rows.map((_, t) => cachedLength + t)
      }
      const { logits, next, vocab, seq } = await forward(ort, session, config, [rows], positions, cache)
      disposeCache(cache)
      cache = next
      cachedLength = recompute ? 0 : positions.at(-1)! + 1 // past the window, the cache can't be reused
      const id = sample(logits, (seq - 1) * vocab, vocab, options.temperature, random)
      ids.push(id)
      yield { id, text: tokenizer.decode([id]), ms: performance.now() - started, recomputed: recompute }
    }
  } finally {
    disposeCache(cache)
  }
}

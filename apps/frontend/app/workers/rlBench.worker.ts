// Device check for reinforcement learning in the browser (docs/design/0010 Phase 0): does this browser's build of the
// RL core train bit-for-bit like the native one (the determinism digests), and how fast does it run? The cases match
// jobs/rl_benchmark.py, which measures the same things natively. Runs in a worker so timing never blocks the page.
import init, { benchForwards, benchUpdates, learningDigest, rolloutDigest, Trainer, trainingDigest } from "~/wasm/rl/rl.js"
import rlWasmUrl from "~/wasm/rl/rl_bg.wasm?url"
import fixture from "~/wasm/rl/determinism.json"

export interface DigestResult {
  name: string
  expected: string
  actual: string
}

export interface BenchResult {
  name: string
  perSecond: number
}

export type RlBenchMessage =
  | { type: "digests"; results: DigestResult[] }
  | { type: "bench"; result: BenchResult }
  | { type: "done" }
  | { type: "error"; message: string }

const NETWORKS: [string, number[]][] = [
  ["11-64-64-3", [11, 64, 64, 3]],
  ["27-128-128-3", [27, 128, 128, 3]],
  ["100-256-256-3", [100, 256, 256, 3]],
]
const ACTIVATIONS = "relu,relu,linear"
const BATCH = 32

/** Units per second: repeat `work(n)` with a growing n until one call takes at least `minMs`. */
function rate(work: (n: number) => unknown, unitCount: number, minMs = 400): number {
  let n = 1
  for (;;) {
    const started = performance.now()
    work(n)
    const elapsed = performance.now() - started
    if (elapsed >= minMs) return (n * unitCount) / (elapsed / 1000)
    n = Math.max(n * 2, Math.ceil((n * minMs) / Math.max(elapsed, 0.01)))
  }
}

const post = (message: RlBenchMessage) => self.postMessage(message)

self.onmessage = async () => {
  try {
    await init({ module_or_path: rlWasmUrl })
    post({
      type: "digests",
      results: [
        ...fixture.training.map((t) => ({
          name: `training (seed ${t.seed}, ${t.updates} updates)`,
          expected: t.digest,
          actual: trainingDigest(t.seed, t.updates),
        })),
        ...fixture.rollouts.map((r) => ({ name: `rollouts (seed ${r.seed})`, expected: r.digest, actual: rolloutDigest(r.seed) })),
        ...fixture.learning.map((l) => ({ name: `learning (seed ${l.seed})`, expected: l.digest, actual: learningDigest(l.seed) })),
      ],
    })
    const trainer = new Trainer("random", "snake/features.v1+relative3.v1", 0, "", "shaped")
    post({ type: "bench", result: { name: "env steps/s (Snake, random agent)", perSecond: rate((n) => trainer.train(n * 1000), 1000) } })
    trainer.free()
    const act = new Uint32Array(NETWORKS[0]![1])
    post({ type: "bench", result: { name: "act/s (11-64-64-3)", perSecond: rate((n) => benchForwards(act, ACTIVATIONS, n * 1000), 1000) } })
    for (const [name, layers] of NETWORKS) {
      const sizes = new Uint32Array(layers)
      post({
        type: "bench",
        result: { name: `updates/s (batch ${BATCH}, ${name})`, perSecond: rate((n) => benchUpdates(sizes, ACTIVATIONS, BATCH, n), 1) },
      })
    }
    post({ type: "done" })
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  }
}

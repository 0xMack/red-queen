// The WASM build of the RL core must reproduce the native build's determinism digests bit for bit
// (docs/design/0010 Decision 2): one seed, one training run, in a job and in a browser. Run in CI (frontend job):
//   node scripts/check-rl-determinism.mjs
// The digests come from libs/rl/tests/determinism.json, copied next to the module by libs/rl/build-wasm.py.
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { initSync, learningDigest, rolloutDigest, trainingDigest } from "../app/wasm/rl/rl.js"

const dir = new URL("../app/wasm/rl/", import.meta.url)
initSync({ module: readFileSync(fileURLToPath(new URL("rl_bg.wasm", dir))) })
const fixture = JSON.parse(readFileSync(fileURLToPath(new URL("determinism.json", dir)), "utf8"))

const results = [
  ...fixture.training.map((t) => [`training(seed ${t.seed}, ${t.updates} updates)`, t.digest, trainingDigest(t.seed, t.updates)]),
  ...fixture.rollouts.map((r) => [`rollouts(seed ${r.seed})`, r.digest, rolloutDigest(r.seed)]),
  ...fixture.learning.map((l) => [`learning(seed ${l.seed})`, l.digest, learningDigest(l.seed)]),
]
let failed = 0
for (const [name, expected, actual] of results) {
  const ok = expected === actual
  failed += ok ? 0 : 1
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: native ${expected}, wasm ${actual}`)
}
if (failed) {
  console.error(`${failed} digest(s) differ: the WASM build does not train like the native one`)
  process.exit(1)
}

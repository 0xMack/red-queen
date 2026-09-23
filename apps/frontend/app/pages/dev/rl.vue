<script setup lang="ts">
// Development page for reinforcement learning in the browser (docs/design/0010 Phase 0): does this browser's build of
// the RL core train exactly like the native one, and how fast is it? Live-training demos in Learn are shaped by these
// numbers. Not linked from the site's navigation.
import type { BenchResult, DigestResult, RlBenchMessage } from "~/workers/rlBench.worker"

useHead({ title: "Reinforcement learning (dev)" })

// jobs/rl_benchmark.py on the development machine (Windows, x86-64, release build), 2026-09-23 -- for comparison.
const NATIVE: Record<string, number> = {
  "env steps/s (Snake, random agent)": 7_054_437,
  "act/s (11-64-64-3)": 530_294,
  "updates/s (batch 32, 11-64-64-3)": 8_294,
  "updates/s (batch 32, 27-128-128-3)": 1_893,
  "updates/s (batch 32, 100-256-256-3)": 429,
}

const digests = ref<DigestResult[]>([])
const bench = ref<BenchResult[]>([])
const status = ref<"idle" | "running" | "done" | "error">("idle")
const error = ref<string | null>(null)
let worker: Worker | null = null

const deterministic = computed(() => digests.value.length > 0 && digests.value.every((d) => d.expected === d.actual))

function run() {
  worker?.terminate()
  digests.value = []
  bench.value = []
  error.value = null
  status.value = "running"
  worker = new Worker(new URL("~/workers/rlBench.worker.ts", import.meta.url), { type: "module" })
  worker.onmessage = (event: MessageEvent<RlBenchMessage>) => {
    const message = event.data
    if (message.type === "digests") digests.value = message.results
    else if (message.type === "bench") bench.value = [...bench.value, message.result]
    else if (message.type === "done") status.value = "done"
    else {
      error.value = message.message
      status.value = "error"
    }
  }
  worker.postMessage("run")
}

onMounted(run)
onUnmounted(() => worker?.terminate())

const format = (n: number) => Math.round(n).toLocaleString()
</script>

<template>
  <main class="mx-auto max-w-[1100px] px-4 py-8 sm:px-6 lg:px-8">
    <p class="eyebrow">Development</p>
    <h1 class="mt-2 text-3xl font-semibold">Reinforcement learning in the browser</h1>
    <p class="mt-2 max-w-3xl text-fg-muted">
      The RL core compiled to WebAssembly (docs/design/0010): does it train bit-for-bit like the native build, and how
      fast does it run here? Learn's live-training demos are sized from these numbers.
    </p>

    <section class="card mt-6 p-5 text-sm" data-rl-determinism>
      <div class="flex items-baseline justify-between gap-3">
        <h2 class="font-display font-semibold">Determinism</h2>
        <span v-if="digests.length" :class="deterministic ? 'text-life-300' : 'text-queen-300'" data-rl-deterministic>
          {{ deterministic ? "identical to the native build" : "differs from the native build" }}
        </span>
      </div>
      <p class="mt-1 text-fg-subtle">Digests of fixed training runs, against <code>libs/rl/tests/determinism.json</code> (computed natively).</p>
      <table v-if="digests.length" class="mt-3 w-full font-mono text-xs">
        <tr v-for="d in digests" :key="d.name" class="border-t border-line">
          <td class="py-1.5 font-sans">{{ d.name }}</td>
          <td>{{ d.expected }}</td>
          <td :class="d.expected === d.actual ? 'text-life-300' : 'text-queen-300'">{{ d.actual }}</td>
        </tr>
      </table>
      <p v-else-if="status === 'running'" class="mt-3 text-fg-subtle">Computing…</p>
    </section>

    <section class="card mt-6 p-5 text-sm" data-rl-bench>
      <div class="flex items-baseline justify-between gap-3">
        <h2 class="font-display font-semibold">Speed</h2>
        <button class="btn-ghost btn-sm" :disabled="status === 'running'" @click="run">Run again</button>
      </div>
      <table class="mt-3 w-full text-xs">
        <tr class="text-left text-fg-subtle">
          <th class="py-1.5 font-normal">Case</th>
          <th class="text-right font-normal">This browser</th>
          <th class="text-right font-normal">Native</th>
          <th class="text-right font-normal">Browser / native</th>
        </tr>
        <tr v-for="b in bench" :key="b.name" class="border-t border-line">
          <td class="py-1.5">{{ b.name }}</td>
          <td class="num text-right">{{ format(b.perSecond) }}</td>
          <td class="num text-right text-fg-subtle">{{ NATIVE[b.name] ? format(NATIVE[b.name]!) : "--" }}</td>
          <td class="num text-right">{{ NATIVE[b.name] ? `${((b.perSecond / NATIVE[b.name]!) * 100).toFixed(0)}%` : "--" }}</td>
        </tr>
      </table>
      <p v-if="status === 'running'" class="mt-3 text-fg-subtle">Measuring… ({{ bench.length }} of 5)</p>
      <p v-if="error" class="mt-3 text-queen-300">{{ error }}</p>
    </section>
  </main>
</template>

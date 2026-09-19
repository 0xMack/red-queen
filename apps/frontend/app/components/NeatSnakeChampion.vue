<script setup lang="ts">
// A real NEAT champion from this project's own training runs, playing Snake in the browser (the real
// Snake game core as WebAssembly, the genome as its ONNX package) with its evolved graph lighting up move by move.
// Prefers the untagged "flagship" run (the one on the leaderboard); falls back to any finished NEAT run.
import type { RunInfo } from "~/types/telemetry"

const config = useRuntimeConfig()
const run = ref<RunInfo | null>(null)

onMounted(async () => {
  try {
    const runs = await $fetch<RunInfo[]>("/runs", { baseURL: config.public.apiBase })
    const neat = runs.filter((r) => r.config?.representation === "neat" && r.status === "completed").sort((a, b) => b.created_at - a.created_at)
    run.value = neat.find((r) => !r.config?.experiment) ?? neat[0] ?? null
  } catch {
    // No backend: the chapter reads fine without the live champion.
  }
})
</script>

<template>
  <figure v-if="run" class="card my-8 p-5">
    <ClientOnly>
      <WatchChampion :run-id="run.run_id" />
    </ClientOnly>
    <figcaption class="mt-4 text-xs text-fg-subtle">
      A finished NEAT run's final champion (<NuxtLink :to="`/runs/${run.run_id}`">{{ shortId(run.run_id) }}</NuxtLink>), loaded from the telemetry store.
      Every node and edge here was added by mutation; the run began with 11 inputs wired straight to 3 outputs and nothing else. Nodes light up with
      their activation on the current board.
    </figcaption>
  </figure>
  <p v-else class="text-sm text-fg-subtle">(Start the backend, and run <code>jobs/snake_neat_run.py</code>, to see a real NEAT champion play here.)</p>
</template>

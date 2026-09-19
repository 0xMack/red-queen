<script setup lang="ts">
import type { GenerationStats, RunInfo } from "~/types/telemetry"

// The real SSE stream, rendered as a log: connects to GET /runs/{id}/metrics/stream the same way
// the run page does, asking to start a few generations back -- so it demonstrates both halves of
// backfill-then-live: the backlog arrives at once, then new generations as the job records them.
// Uses a training run if one is live, otherwise the latest run (which replays and then idles).
const config = useRuntimeConfig()
const run = ref<RunInfo | null>(null)
const events = ref<{ stats: GenerationStats; receivedAt: number }[]>([])
const status = ref<"connecting" | "open" | "error" | "none">("connecting")
let source: EventSource | null = null

onMounted(async () => {
  try {
    const runs = await $fetch<RunInfo[]>("/runs", { baseURL: config.public.apiBase })
    run.value = runs.find((r) => r.status === "running") ?? runs[0] ?? null
  } catch {
    status.value = "error"
    return
  }
  if (!run.value) {
    status.value = "none"
    return
  }
  const history = await $fetch<GenerationStats[]>(`/runs/${run.value.run_id}/metrics/history`, {
    baseURL: config.public.apiBase,
  }).catch(() => [])
  const since = Math.max(0, (history.at(-1)?.generation ?? 0) - 4)
  source = new EventSource(`${config.public.apiBase}/runs/${run.value.run_id}/metrics/stream?since_generation=${since}`)
  source.onopen = () => (status.value = "open")
  source.onerror = () => (status.value = "error")
  source.addEventListener("generation", (event) => {
    const stats = JSON.parse((event as MessageEvent).data) as GenerationStats
    events.value = [...events.value.slice(-11), { stats, receivedAt: Date.now() / 1000 }]
  })
})
onUnmounted(() => source?.close())

const live = computed(() => run.value?.status === "running")
</script>

<template>
  <figure class="card my-8 overflow-hidden">
    <div class="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2.5 text-xs">
      <span class="size-2 rounded-full" :class="status === 'open' ? 'animate-live-pulse bg-life-400' : 'bg-fg-subtle'" />
      <span class="font-mono text-fg-muted">GET /runs/{{ run ? shortId(run.run_id) : "…" }}/metrics/stream</span>
      <span class="ml-auto text-fg-subtle">
        {{ status === "open" ? (live ? "live -- this run is training right now" : "connected -- run finished, replayed the backlog") : status === "error" ? "backend unreachable" : status === "none" ? "no runs yet" : "connecting…" }}
      </span>
    </div>
    <ol class="max-h-72 overflow-y-auto bg-sunken font-mono text-[12px]">
      <li v-for="e in [...events].reverse()" :key="e.stats.generation" class="flex gap-4 border-b border-line/40 px-4 py-1.5">
        <span class="text-fg-subtle">event: generation</span>
        <span class="text-fg">gen {{ e.stats.generation }}</span>
        <span class="text-queen-300">best {{ e.stats.best_fitness.toFixed(3) }}</span>
        <span v-if="e.stats.held_out_score != null" class="text-life-300">held-out {{ e.stats.held_out_score.toFixed(2) }}</span>
        <span class="ml-auto text-fg-subtle" :title="'seconds between the job recording it and this page receiving it'">
          +{{ Math.max(0, e.receivedAt - e.stats.timestamp).toFixed(1) }}s
        </span>
      </li>
      <li v-if="!events.length" class="px-4 py-6 text-center text-fg-subtle">waiting for events…</li>
    </ol>
    <figcaption class="border-t border-line px-4 py-2 text-xs text-fg-subtle">
      The right column is the delay between the training job writing a generation and this page receiving it. Backlog
      events are old (they were recorded before you opened the page); live ones arrive within about a second.
    </figcaption>
  </figure>
</template>

<script setup lang="ts">
import type { GenerationStats } from "~/types/telemetry"

// A static (one-shot fetch, not live via SSE) preview for embedding a real run's fitness curve in
// Learn chapters -- reuses FitnessChart.vue exactly as-is, the same component the live run-detail
// page uses. Illustrating a chapter's point with a real, historical result rather than a synthetic
// chart or a screenshot -- if the backend/run data isn't reachable, this just doesn't render (the
// caller decides what to show instead), rather than adding its own error UI to a page that's
// mostly prose.
const props = defineProps<{ runId: string }>()

const api = useApi()
const history = ref<GenerationStats[] | null>(null)

onMounted(async () => {
  try {
    history.value = await api.fetch<GenerationStats[]>(`/runs/${props.runId}/metrics/history`)
  } catch {
    history.value = null
  }
})
</script>

<template>
  <figure v-if="history" class="card my-8 p-5">
    <FitnessChart :history="history" :height="260" />
    <figcaption class="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-fg-subtle">
      <span>A real recorded run -- {{ history.length }} generations, straight from the telemetry store.</span>
      <NuxtLink :to="`/runs/${runId}`" class="link">Open this run →</NuxtLink>
    </figcaption>
  </figure>
</template>

<script setup lang="ts">
import type { RunInfo } from "~/types/telemetry"

// Interaction modes 2 and 3 in one page (doc 0005 step 6): watch a trained policy play, driven
// entirely client-side. Which viewer depends on the run's game: WatchChampion (Snake, in a Web Worker
// with a model package) or CheckersWatch (Checkers, the champion playing an opponent on the board).
// Both own their metrics stream and champion loading; the run detail page embeds the same components
// next to its charts.
const route = useRoute()
const runId = route.params.runId as string
const api = useApi()
useHead({ title: "Watch" })

const { data: run } = await useAsyncData(`watch-run-${runId}`, () =>
  api.fetch<RunInfo>(`/runs/${runId}`).catch(() => null),
)
const game = computed(() => (typeof run.value?.config?.game === "string" ? run.value.config.game : null))
</script>

<template>
  <main class="mx-auto max-w-[1400px] px-4 py-8 sm:px-6 lg:px-8">
    <NuxtLink :to="`/runs/${runId}`" class="text-sm text-fg-subtle transition hover:text-fg">&larr; Run details</NuxtLink>
    <p class="eyebrow mt-4">Watch mode</p>
    <h1 class="mt-2 text-3xl font-semibold">A trained champion, playing live</h1>
    <p class="mt-2 max-w-2xl text-fg-muted">
      <template v-if="game === 'checkers'">
        The champion plays the opponent you pick; drag the slider to see any generation, or measure it in
        the arena. If the run is still training, the newest champion is swapped in the moment it's recorded.
      </template>
      <template v-else>
        No controls -- the network decides every move. If the run is still training, the newest
        champion is swapped in the moment it's recorded.
      </template>
      <span class="font-mono text-xs text-fg-subtle">{{ runId }}</span>
    </p>
    <div class="card mt-8 p-5">
      <ClientOnly>
        <CheckersWatch v-if="game === 'checkers'" :run-id="runId" scrubber />
        <WatchChampion v-else :run-id="runId" />
      </ClientOnly>
    </div>
  </main>
</template>

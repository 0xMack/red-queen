<script setup lang="ts">
// Interaction modes 2 and 3 in one page (doc 0005 step 6): watch a trained policy play, driven
// entirely client-side by the same Web Worker /play/[game].vue uses. WatchChampion.vue (built on
// useWatchSession) owns the worker session, the metrics-stream watcher, and champion loading; the
// run detail page embeds the same component next to its charts.
const route = useRoute()
const runId = route.params.runId as string
useHead({ title: "Watch" })
</script>

<template>
  <main class="mx-auto max-w-[1400px] px-4 py-8 sm:px-6 lg:px-8">
    <NuxtLink :to="`/runs/${runId}`" class="text-sm text-fg-subtle transition hover:text-fg">&larr; Run details</NuxtLink>
    <p class="eyebrow mt-4">Watch mode</p>
    <h1 class="mt-2 text-3xl font-semibold">A trained champion, playing live</h1>
    <p class="mt-2 max-w-2xl text-fg-muted">
      No controls -- the network decides every move. If the run is still training, the newest
      champion is swapped in the moment it's recorded.
      <span class="font-mono text-xs text-fg-subtle">{{ runId }}</span>
    </p>
    <div class="card mt-8 p-5">
      <ClientOnly>
        <WatchChampion :run-id="runId" />
      </ClientOnly>
    </div>
  </main>
</template>

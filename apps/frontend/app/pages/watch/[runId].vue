<script setup lang="ts">
// Interaction modes 2 and 3 in one page (doc 0005 step 6): watch a trained policy play, driven
// entirely client-side by the same Web Worker /play/[game].vue uses. useWatchSession()
// (app/composables/useWatchSession.ts) owns the worker session, the metrics-stream watcher, and
// the champion-loading logic; this page is just that composable plus markup.
const route = useRoute()
const runId = route.params.runId as string

const { loading, error, renderState, reward, done, stepCount, lastLoadedRef, start, retry } = useWatchSession(runId)

onMounted(start)
</script>

<template>
  <main class="mx-auto max-w-2xl p-6">
    <NuxtLink :to="`/runs/${runId}`" class="text-sm text-slate-500 hover:underline">&larr; run detail</NuxtLink>
    <h1 class="mt-2 text-2xl font-semibold text-slate-900">Watching <span class="font-mono text-lg">{{ runId }}</span></h1>
    <p class="mt-1 text-sm text-slate-500">
      A trained policy plays automatically -- no controls. Re-loads the current-best champion
      whenever a new one is recorded.
    </p>

    <p v-if="loading" class="mt-6 text-slate-500">Loading Python runtime...</p>
    <div v-else-if="error" class="mt-4">
      <p class="text-red-600">{{ error }}</p>
      <button
        class="mt-2 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
        @click="retry"
      >
        Retry
      </button>
    </div>

    <template v-else-if="renderState">
      <GridBoard :state="renderState" class="mt-4" />

      <div class="mt-4 flex items-center gap-6 text-sm">
        <GameStatRow :score="renderState.score" :step="stepCount" :reward="reward" />
        <span v-if="lastLoadedRef" class="font-mono text-xs text-slate-400">{{ lastLoadedRef }}</span>
      </div>

      <p v-if="done" class="mt-4 text-slate-600">
        Episode ended -- will restart automatically once a new champion is recorded.
      </p>
    </template>
  </main>
</template>

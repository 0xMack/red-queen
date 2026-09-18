<script setup lang="ts">
// Web Worker phase of client-side simulation (doc 0005 step 5): the actual Pyodide runtime, the
// games.snake.Snake instance, and the tick timer all live in app/workers/snakeGame.worker.ts, off
// this thread entirely. usePlaySession() (app/composables/usePlaySession.ts) owns the worker
// session and window-scoped keyboard capture; this page is just that composable plus markup --
// LiveSnakeDemo.vue is the same underlying pieces embedded elsewhere instead of on its own page.
const route = useRoute()
const game = route.params.game as string

// Only games.snake has a worker wired up so far -- games.reach1d doesn't implement Renderable yet
// (no render_state()), matching the same restriction as apis/backend's game registry.
const supported = game === "snake"

const { loading, error, renderState, reward, done, stepCount, start, restart } = usePlaySession()
if (!supported) {
  loading.value = false
  error.value = `unsupported game: ${game}`
}

onMounted(() => {
  if (supported) start()
})
</script>

<template>
  <main class="mx-auto max-w-2xl p-6">
    <NuxtLink to="/games" class="text-sm text-slate-500 hover:underline">&larr; games</NuxtLink>
    <h1 class="mt-2 text-2xl font-semibold capitalize text-slate-900">{{ game }}</h1>
    <p class="mt-1 text-sm text-slate-500">
      Running entirely in your browser via Pyodide, in a Web Worker -- use the arrow keys to steer.
    </p>

    <p v-if="loading" class="mt-6 text-slate-500">Loading Python runtime...</p>
    <div v-else-if="error" class="mt-4">
      <p class="text-red-600">{{ error }}</p>
      <button
        v-if="supported"
        class="mt-2 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
        @click="start"
      >
        Retry
      </button>
    </div>

    <template v-else-if="renderState">
      <GridBoard :state="renderState" class="mt-4" />
      <GameStatRow :score="renderState.score" :step="stepCount" :reward="reward" class="mt-4" />

      <div v-if="done" class="mt-4">
        <p class="text-slate-600">Game over.</p>
        <button
          class="mt-2 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
          @click="restart"
        >
          Play again
        </button>
      </div>
    </template>
  </main>
</template>

<script setup lang="ts">
// Web Worker phase of client-side simulation (doc 0005 step 5): the actual Pyodide runtime, the
// games.snake.Snake instance, and the tick timer all live in app/workers/snakeGame.worker.ts, off
// this thread entirely. usePlaySession() (app/composables/usePlaySession.ts) owns the worker
// session and window-scoped keyboard capture; this page is just that composable plus markup --
// LiveSnakeDemo.vue is the same underlying pieces embedded elsewhere instead of on its own page.
const route = useRoute()
const game = route.params.game as string
useHead({ title: `Play ${game}` })

// Only games.snake has a worker wired up so far -- games.reach1d doesn't implement Renderable yet
// (no render_state()), matching the same restriction as apis/backend's game registry.
const supported = game === "snake"

const { loading, error, renderState, reward, done, stepCount, episodes, bestScore, start, restart } = usePlaySession()
if (!supported) {
  loading.value = false
  error.value = `unsupported game: ${game}`
}

onMounted(() => {
  if (supported) start()
})

function onKey(event: KeyboardEvent) {
  if (done.value && (event.key === " " || event.key === "Enter")) {
    event.preventDefault()
    restart()
  }
}
onMounted(() => window.addEventListener("keydown", onKey))
onUnmounted(() => window.removeEventListener("keydown", onKey))
</script>

<template>
  <main class="mx-auto max-w-[1400px] px-4 py-8 sm:px-6 lg:px-8">
    <NuxtLink to="/games" class="text-sm text-fg-subtle transition hover:text-fg">&larr; Games</NuxtLink>

    <div class="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div class="mx-auto w-full max-w-[680px]">
        <div v-if="loading" class="card flex aspect-square flex-col items-center justify-center gap-3">
          <BrandMark class="size-10 animate-pulse" />
          <p class="text-sm text-fg-muted">Starting the Python runtime (first load ~10s)…</p>
        </div>
        <div v-else-if="error" class="card flex aspect-square flex-col items-center justify-center gap-3 p-6 text-center">
          <p class="text-queen-300">{{ error }}</p>
          <button v-if="supported" class="btn-ghost btn-sm" @click="start">Retry</button>
        </div>
        <div v-else-if="renderState" class="relative">
          <GridBoard :state="renderState" />
          <div v-if="done" class="absolute inset-0 flex flex-col items-center justify-center gap-4 rounded-xl bg-bg/70 backdrop-blur-sm">
            <p class="font-display text-3xl font-semibold">Game over</p>
            <p class="num text-fg-muted">score {{ renderState.score }} · {{ stepCount }} steps</p>
            <button class="btn-primary" @click="restart">Play again <span class="text-xs opacity-70">(space)</span></button>
          </div>
        </div>
      </div>

      <aside class="flex flex-col gap-4">
        <div>
          <p class="eyebrow">Play mode</p>
          <h1 class="mt-2 text-3xl font-semibold capitalize">{{ game }}</h1>
          <p class="mt-2 text-sm text-fg-muted">
            Running entirely in your browser -- the real <code class="chip">games.snake</code> Python
            module, in Pyodide, in a Web Worker. Zero server round trips per move.
          </p>
        </div>

        <div class="grid grid-cols-2 gap-2">
          <StatTile label="Score" :value="renderState?.score ?? 0" tone="life" />
          <StatTile label="Steps" :value="stepCount" />
          <StatTile label="Best this session" :value="bestScore" tone="gold" />
          <StatTile label="Reward" :value="reward.toFixed(2)" :hint="`${episodes} games played`" />
        </div>

        <div class="card p-4 text-sm">
          <p class="font-medium">Controls</p>
          <div class="mt-3 grid w-fit grid-cols-3 gap-1 font-mono text-xs">
            <span />
            <kbd class="rounded border border-line-strong bg-raised px-2 py-1 text-center">↑</kbd>
            <span />
            <kbd class="rounded border border-line-strong bg-raised px-2 py-1 text-center">←</kbd>
            <kbd class="rounded border border-line-strong bg-raised px-2 py-1 text-center">↓</kbd>
            <kbd class="rounded border border-line-strong bg-raised px-2 py-1 text-center">→</kbd>
          </div>
          <p class="mt-3 text-xs text-fg-subtle">
            Arrow keys are translated to Snake's <em>relative</em> action space (left / straight /
            right) -- the same three actions a trained policy chooses between.
          </p>
        </div>

        <NuxtLink to="/runs" class="card card-hover block p-4">
          <p class="text-sm font-medium">Think you can beat evolution? →</p>
          <p class="mt-1 text-xs text-fg-subtle">Open a Snake run to watch its trained champion play the same game.</p>
        </NuxtLink>
      </aside>
    </div>
  </main>
</template>

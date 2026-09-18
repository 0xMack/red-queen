<script setup lang="ts">
import type { RenderState } from "~/types/games"

// Web Worker phase of client-side simulation (doc 0005 step 5): the actual Pyodide runtime, the
// games.snake.Snake instance, and the tick timer all live in app/workers/snakeGame.worker.ts, off
// this thread entirely -- heavy Python/WASM work can never jank the page. This thread's only jobs
// are (a) translate absolute arrow-key presses into Snake's relative action space using a
// client-tracked heading, and (b) redraw from the state messages the worker posts back. Supersedes
// the main-thread-Pyodide version of this page (doc 0005 step 5's first phase, which proved the
// tick-loop and keypress translation worked before moving them off-thread).
const route = useRoute()
const game = route.params.game as string

// Only games.snake has a worker wired up so far -- games.reach1d doesn't implement Renderable yet
// (no render_state()), matching the same restriction as apis/backend's game registry.
const supported = game === "snake"

const loading = ref(true)
const error = ref<string | null>(supported ? null : `unsupported game: ${game}`)
const renderState = ref<RenderState | null>(null)
const reward = ref(0)
const done = ref(false)
const stepCount = ref(0)

// Clockwise order, must match games.snake._DIRECTIONS exactly -- doc 0005's worked example: this
// thread tracks heading purely to translate absolute key presses into the relative turns the
// worker's Snake.step() expects, and never runs Snake's own movement logic itself.
const HEADING_FOR_KEY: Record<string, number> = { ArrowRight: 0, ArrowDown: 1, ArrowLeft: 2, ArrowUp: 3 }
let headingIndex = 0

let worker: Worker | null = null

function onKeydown(event: KeyboardEvent) {
  const target = HEADING_FOR_KEY[event.key]
  if (target === undefined || !worker) return
  const delta = (target - headingIndex + 4) % 4
  const action = delta === 1 ? 1 : delta === 3 ? -1 : 0 // right turn, left turn, or already-there/reversal
  headingIndex = (headingIndex + action + 4) % 4
  worker.postMessage({ type: "input", action })
}

function restart() {
  headingIndex = 0
  worker?.postMessage({ type: "restart" })
}

function startSession() {
  loading.value = true
  error.value = null
  headingIndex = 0

  // A shared worker, not one created (and Pyodide-loaded) fresh per visit -- see
  // app/composables/useSnakeWorker.ts. Assigning onmessage here replaces whatever the previously
  // active page attached, which is all the "detach" a single-worker, one-visible-page-at-a-time
  // app needs.
  worker = getSnakeWorker()
  worker.onmessage = (event: MessageEvent) => {
    const message = event.data
    if (message.type === "ready") {
      loading.value = false
    } else if (message.type === "state") {
      renderState.value = message.renderState
      reward.value = message.reward
      done.value = message.done
      stepCount.value = message.step
    } else if (message.type === "error") {
      error.value = message.message
      loading.value = false
    }
  }
  worker.postMessage({ type: "start" })
}

onMounted(() => {
  if (!supported) {
    loading.value = false
    return
  }
  window.addEventListener("keydown", onKeydown)
  startSession()
})
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown)
  // Not terminate() -- the worker is shared and may be reused by the next page. "stop" just pauses
  // its tick loop so it doesn't keep running (or posting messages into the void) while unused.
  worker?.postMessage({ type: "stop" })
  worker = null
})
</script>

<template>
  <main class="mx-auto max-w-2xl p-6">
    <NuxtLink to="/" class="text-sm text-slate-500 hover:underline">&larr; all runs</NuxtLink>
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
        @click="startSession"
      >
        Retry
      </button>
    </div>

    <template v-else-if="renderState">
      <GridBoard :state="renderState" class="mt-4" />

      <div class="mt-4 flex items-center gap-6 text-sm">
        <span>score: <span class="font-mono">{{ renderState.score }}</span></span>
        <span>step: <span class="font-mono">{{ stepCount }}</span></span>
        <span>reward: <span class="font-mono">{{ reward.toFixed(2) }}</span></span>
      </div>

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

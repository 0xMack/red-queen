<script setup lang="ts">
import type { PyodideInterface } from "pyodide"
import type { PyProxy } from "pyodide/ffi"
import type { RenderState } from "~/types/games"

// Client-side simulation (doc 0005 step 5, main-thread phase): the whole games.snake.Snake
// instance and its step() loop run in-browser via Pyodide -- no backend round trip per tick. This
// replaces doc 0005 step 4's version of this same page, which drove the game over
// apis/backend/routers/games.py instead; that router and its tests are unchanged and still valid,
// just no longer this page's data source.
const route = useRoute()
const game = route.params.game as string

// Only games.snake is wired into the Pyodide bridge so far -- games.reach1d doesn't implement
// Renderable yet (no render_state()), matching the same restriction as apis/backend's game
// registry (see apis/backend/README.md).
const supported = game === "snake"

const loading = ref(true)
const error = ref<string | null>(supported ? null : `unsupported game: ${game}`)
const renderState = ref<RenderState | null>(null)
const reward = ref(0)
const done = ref(false)
const stepCount = ref(0)

// Clockwise order, must match games.snake._DIRECTIONS exactly -- the client tracks heading purely
// to translate absolute arrow-key presses into the relative turns Snake.step() expects; it never
// reimplements Snake's own movement logic (doc 0005's worked example).
const HEADING_FOR_KEY: Record<string, number> = { ArrowRight: 0, ArrowDown: 1, ArrowLeft: 2, ArrowUp: 3 }
let headingIndex = 0
let pendingAction = 0

let pyodide: PyodideInterface | null = null
let snake: PyProxy | null = null
let renderStateForJs: PyProxy | null = null
let timer: ReturnType<typeof setInterval> | null = null
const tickIntervalMs = 150

function stopTicking() {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}

function onKeydown(event: KeyboardEvent) {
  const target = HEADING_FOR_KEY[event.key]
  if (target === undefined) return
  const delta = (target - headingIndex + 4) % 4
  if (delta === 1) pendingAction = 1 // right turn
  else if (delta === 3) pendingAction = -1 // left turn
  else pendingAction = 0 // already heading there, or a same-tick reversal Snake can't do anyway
}

// render_state_for_js (defined in usePyodideGames.ts's Python bootstrap) already flattens the
// tuple-keyed `cells` dict -- dict_converter here just makes every level (including each cell)
// come back as a plain object instead of a Map, matching RenderState/GridCell field access.
function readRenderState(): RenderState {
  const stateProxy = renderStateForJs!(snake)
  const raw = stateProxy.toJs({ dict_converter: Object.fromEntries }) as RenderState
  stateProxy.destroy()
  return raw
}

function tick() {
  if (!snake || done.value) {
    stopTicking()
    return
  }
  // Explicit destroy() on every PyProxy the tick loop creates -- this runs every 150ms for as long
  // as the page is open, so leaving cleanup to Pyodide's FinalizationRegistry safety net alone
  // would grow the WASM heap needlessly instead of freeing it immediately.
  const resultProxy = snake.step(pendingAction)
  const [, stepReward, stepDone] = resultProxy.toJs() as [unknown, number, boolean]
  resultProxy.destroy()

  headingIndex = (headingIndex + pendingAction + 4) % 4
  pendingAction = 0
  reward.value = stepReward
  done.value = stepDone
  stepCount.value += 1
  renderState.value = readRenderState()
}

function restart() {
  stopTicking()
  if (!snake) return
  const obsProxy = snake.reset()
  obsProxy.destroy()
  headingIndex = 0
  pendingAction = 0
  stepCount.value = 0
  reward.value = 0
  done.value = false
  renderState.value = readRenderState()
  timer = setInterval(tick, tickIntervalMs)
}

onMounted(async () => {
  if (!supported) {
    loading.value = false
    return
  }
  window.addEventListener("keydown", onKeydown)
  try {
    pyodide = await loadGamesPyodide()
    renderStateForJs = pyodide.globals.get("render_state_for_js")
    snake = pyodide.runPython(`
import random
from games.snake import Snake
Snake(seed=random.randint(0, 2**31 - 1))
`)
    renderState.value = readRenderState()
    loading.value = false
    timer = setInterval(tick, tickIntervalMs)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    loading.value = false
  }
})
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown)
  stopTicking()
  snake?.destroy()
  renderStateForJs?.destroy()
})
</script>

<template>
  <main class="mx-auto max-w-2xl p-6">
    <NuxtLink to="/" class="text-sm text-slate-500 hover:underline">&larr; all runs</NuxtLink>
    <h1 class="mt-2 text-2xl font-semibold capitalize text-slate-900">{{ game }}</h1>
    <p class="mt-1 text-sm text-slate-500">
      Running entirely in your browser via Pyodide -- use the arrow keys to steer.
    </p>

    <p v-if="loading" class="mt-6 text-slate-500">Loading Python runtime...</p>
    <p v-else-if="error" class="mt-4 text-red-600">{{ error }}</p>

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

<script setup lang="ts">
import type { GameSessionState } from "~/types/games"

// Server-side simulation for now (doc 0005 step 4): each tick POSTs to apis/backend, which owns
// the actual Snake instance. Controls are Snake's native relative action space directly
// (ArrowLeft/ArrowRight -> turn, otherwise straight) rather than absolute-direction controls with
// client-tracked heading -- that translation is specifically the Pyodide/Web-Worker worked example
// in doc 0005 (step 5), where the game moves client-side and the server round-trip goes away.
const route = useRoute()
const game = route.params.game as string
const config = useRuntimeConfig()

const session = ref<GameSessionState | null>(null)
const error = ref<string | null>(null)
const pendingAction = ref(0)
const tickIntervalMs = 150

let timer: ReturnType<typeof setInterval> | null = null

function stopTicking() {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}

async function createSession() {
  try {
    session.value = await $fetch<GameSessionState>(`/games/${game}/sessions`, {
      method: "POST",
      baseURL: config.public.apiBase,
      body: { game, seed: null },
    })
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

async function tick() {
  if (!session.value || session.value.done) {
    stopTicking()
    return
  }
  try {
    session.value = await $fetch<GameSessionState>(
      `/games/${game}/sessions/${session.value.session_id}/actions`,
      { method: "POST", baseURL: config.public.apiBase, body: { action: pendingAction.value } },
    )
    pendingAction.value = 0
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    stopTicking()
  }
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "ArrowLeft") pendingAction.value = -1
  else if (event.key === "ArrowRight") pendingAction.value = 1
}

async function restart() {
  stopTicking()
  error.value = null
  await createSession()
  timer = setInterval(tick, tickIntervalMs)
}

onMounted(async () => {
  window.addEventListener("keydown", onKeydown)
  await restart()
})
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown)
  stopTicking()
})
</script>

<template>
  <main class="mx-auto max-w-2xl p-6">
    <NuxtLink to="/" class="text-sm text-slate-500 hover:underline">&larr; all runs</NuxtLink>
    <h1 class="mt-2 text-2xl font-semibold capitalize text-slate-900">{{ game }}</h1>
    <p class="mt-1 text-sm text-slate-500">Use &larr;/&rarr; to turn.</p>

    <p v-if="error" class="mt-4 text-red-600">{{ error }}</p>

    <template v-else-if="session">
      <GridBoard :state="session.render_state" class="mt-4" />

      <div class="mt-4 flex items-center gap-6 text-sm">
        <span>score: <span class="font-mono">{{ session.render_state.score }}</span></span>
        <span>step: <span class="font-mono">{{ session.step }}</span></span>
        <span>reward: <span class="font-mono">{{ session.reward.toFixed(2) }}</span></span>
      </div>

      <div v-if="session.done" class="mt-4">
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

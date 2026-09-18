<script setup lang="ts">
// An embeddable, self-contained "play Snake right here" widget -- the landing page and the
// "Teaching a Snake to Play Itself" Learn chapter both use this instead of duplicating
// /play/[game].vue's markup. Deliberately play-mode only for now, not parameterized for "watch a
// specific run" too: nothing embeds a watch-mode demo yet, and useWatchSession already exists and
// is proven (see /watch/[runId].vue) -- wiring it in here is a small, natural extension whenever
// something actually needs it, not speculative work to do now.
//
// Element-scoped keyboard capture (not window, unlike the dedicated /play page): this widget
// lives inside a page a visitor is reading, so it shouldn't hijack the whole page's arrow keys --
// only when the demo itself has focus.
const containerRef = ref<HTMLElement | null>(null)
const session = useSnakeSession()
const { loading, error, renderState, reward, done, stepCount, start, restart, sendInput } = session
const heading = createHeadingTracker()

function onKeydown(event: KeyboardEvent) {
  const action = heading.translate(event.key)
  if (action !== null) {
    event.preventDefault() // stop arrow keys from also scrolling the page while the demo has focus
    sendInput(action)
  }
}

function startDemo() {
  heading.reset()
  start()
}

onMounted(startDemo)
onUnmounted(() => session.stop())
</script>

<template>
  <div
    ref="containerRef"
    tabindex="0"
    class="inline-block rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
    @keydown="onKeydown"
  >
    <p v-if="loading" class="text-sm text-slate-500">Loading Python runtime...</p>
    <div v-else-if="error" class="text-sm">
      <p class="text-red-600">{{ error }}</p>
      <button
        class="mt-2 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
        @click="startDemo"
      >
        Retry
      </button>
    </div>

    <template v-else-if="renderState">
      <GridBoard :state="renderState" />
      <div class="mt-2 flex items-center justify-between gap-4">
        <GameStatRow :score="renderState.score" :step="stepCount" :reward="reward" />
        <button
          v-if="done"
          class="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
          @click="restart"
        >
          Play again
        </button>
        <p v-else class="text-xs text-slate-400">click the board, then use the arrow keys</p>
      </div>
    </template>
  </div>
</template>

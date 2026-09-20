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

// Started on the first click, not on mount: the demo is usually below the fold, and a game that
// starts ticking on page load is already over by the time a reader scrolls to it.
const started = ref(false)
function startDemo() {
  started.value = true
  heading.reset()
  start()
  containerRef.value?.focus()
}

onUnmounted(() => session.stop())
</script>

<template>
  <div
    ref="containerRef"
    tabindex="0"
    class="group/demo block w-full max-w-[420px] rounded-xl outline-none"
    @keydown="onKeydown"
  >
    <button
      v-if="!started"
      class="card group/start flex aspect-square w-full flex-col items-center justify-center gap-3 bg-[radial-gradient(circle_at_center,rgb(74_222_128/0.08),transparent_65%)]"
      @click="startDemo"
    >
      <span class="flex size-14 items-center justify-center rounded-full border border-life-400/40 bg-life-400/10 text-2xl text-life-300 transition group-hover/start:scale-110">▶</span>
      <span class="font-display font-semibold">Click to play Snake</span>
      <span class="text-xs text-fg-subtle">arrow keys to steer</span>
    </button>
    <div v-else-if="loading" class="card flex aspect-square flex-col items-center justify-center gap-3">
      <BrandMark class="size-9 animate-pulse" />
      <p class="text-sm text-fg-muted">Starting the game…</p>
    </div>
    <div v-else-if="error" class="card flex aspect-square flex-col items-center justify-center gap-3 p-6 text-center text-sm">
      <p class="text-queen-300">{{ error }}</p>
      <button class="btn-ghost btn-sm" @click="startDemo">Retry</button>
    </div>

    <template v-else-if="renderState">
      <div class="relative rounded-xl ring-queen-400/70 transition group-focus-visible/demo:ring-2 group-focus/demo:ring-2">
        <GridBoard :state="renderState" />
        <!-- The same result overlay every game uses (BoardResult); the demo's action sits below the board. -->
        <BoardResult v-if="done" title="Game over" :sub="`${renderState.score} 🍎`" tone="draw" />
      </div>
      <div class="mt-3 flex items-center justify-between gap-4">
        <GameStatRow :score="renderState.score" :step="stepCount" :reward="reward" />
        <button v-if="done" class="btn-primary btn-sm" @click="restart">Play again</button>
        <p v-else class="text-[11px] text-fg-subtle">click the board, then arrow keys</p>
      </div>
    </template>
  </div>
</template>

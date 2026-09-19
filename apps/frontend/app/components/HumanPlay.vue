<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"
import type { HumanHistory } from "~/utils/leaderboard"

// "Play it yourself" on the game page: the same board and the same game the algorithms play (same
// Snake, same 10x10 board, same session worker and WebAssembly game core), with a human steering instead of a policy. A short
// countdown, a live race against every entrant's mean while playing, then HumanResults.
// Keyboard capture is window-scoped (this component owns the stage while mounted); arrow keys and
// space are preventDefault'ed so the page doesn't scroll mid-game.
const props = defineProps<{ game: string; entries: EvaluationRecord[] }>()
const emit = defineEmits<{ score: [score: number, live: boolean]; exit: [] }>()

const session = useSnakeSession()
const { loading, error, renderState, stepCount, done } = session
const heading = createHeadingTracker()

type Phase = "countdown" | "playing" | "over"
const phase = ref<Phase>("countdown")
const countdown = ref(3)
const started = ref(false)
const history = ref<HumanHistory>({ best: 0, games: [] })
const newBest = ref(false)
const finalScore = ref(0)
let countdownTimer: ReturnType<typeof setInterval> | null = null

function begin() {
  phase.value = "countdown"
  countdown.value = 3
  newBest.value = false
  heading.reset()
  countdownTimer && clearInterval(countdownTimer)
  countdownTimer = setInterval(() => {
    countdown.value -= 1
    if (countdown.value > 0) return
    clearInterval(countdownTimer!)
    countdownTimer = null
    phase.value = "playing"
    if (started.value) session.restart()
    else {
      started.value = true
      session.start() // no policy = play mode
    }
  }, 650)
}

const score = computed(() => renderState.value?.score ?? 0)
watch(score, (s) => phase.value === "playing" && emit("score", s, true))

watch(done, (isDone) => {
  if (!isDone || phase.value !== "playing") return
  const before = history.value.best
  finalScore.value = score.value
  history.value = saveHumanGame(props.game, finalScore.value)
  newBest.value = finalScore.value > before
  phase.value = "over"
  emit("score", finalScore.value, false)
})

// Touch screens have no arrow keys: an on-screen pad (shown only for coarse pointers) sends the same
// key names through the same translation.
function press(key: string) {
  if (phase.value !== "playing") return
  const action = heading.translate(key)
  if (action !== null) session.sendInput(action)
}

const KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " ", "Enter"])
function onKeydown(event: KeyboardEvent) {
  if (!KEYS.has(event.key)) return
  if ((event.target as HTMLElement | null)?.closest("input, textarea, select")) return
  event.preventDefault()
  if (phase.value === "over" && (event.key === " " || event.key === "Enter")) return begin()
  if (phase.value !== "playing") return
  const action = heading.translate(event.key)
  if (action !== null) session.sendInput(action)
}

// Count down only once the runtime is loaded (instant if it's already warm from watching).
const warming = ref(true)
watch(loading, (isLoading) => {
  if (!isLoading && warming.value && !error.value) {
    warming.value = false
    begin()
  }
})

onMounted(() => {
  history.value = loadHumanHistory(props.game)
  window.addEventListener("keydown", onKeydown)
  session.warmup()
})
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown)
  countdownTimer && clearInterval(countdownTimer)
  session.stop()
})
</script>

<template>
  <div class="@container">
    <div class="grid gap-5 @3xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
      <div class="relative mx-auto w-full max-w-[520px] @3xl:max-w-none">
        <GridBoard v-if="renderState" :state="renderState" :class="phase === 'countdown' ? 'opacity-40 blur-[2px]' : ''" class="transition" />
        <div v-else class="card aspect-square" />

        <div v-if="phase === 'countdown'" class="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <p v-if="error" class="text-sm text-queen-300">{{ error }}</p>
          <template v-else>
            <template v-if="warming">
              <BrandMark class="size-10 animate-pulse" />
              <p class="text-sm text-fg-muted">Loading the game…</p>
            </template>
            <template v-else>
              <p class="text-sm tracking-widest text-fg-muted uppercase">Get ready</p>
              <Transition name="count" mode="out-in">
                <p :key="countdown" class="num font-display text-8xl font-bold text-queen-300">{{ countdown }}</p>
              </Transition>
            </template>
            <p class="text-xs text-fg-subtle">steer with the arrow keys (or the pad below on touch screens)</p>
          </template>
        </div>
        <div v-if="phase === 'playing' && loading" class="absolute inset-0 flex items-center justify-center text-sm text-fg-muted">
          Starting the game…
        </div>
        <div v-if="phase !== 'countdown'" class="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between">
          <span class="chip border-queen-400/40 bg-bg/80 text-queen-200 backdrop-blur">🎮 you're playing</span>
          <span class="num rounded-md bg-bg/80 px-2 py-0.5 text-sm font-semibold text-life-300 backdrop-blur">{{ score }} 🍎</span>
        </div>
      </div>

      <div class="mx-auto hidden w-fit grid-cols-3 gap-2 pointer-coarse:grid @3xl:col-start-1">
        <span />
        <button class="btn-ghost size-14 text-xl" aria-label="up" @click="press('ArrowUp')">↑</button>
        <span />
        <button class="btn-ghost size-14 text-xl" aria-label="left" @click="press('ArrowLeft')">←</button>
        <button class="btn-ghost size-14 text-xl" aria-label="down" @click="press('ArrowDown')">↓</button>
        <button class="btn-ghost size-14 text-xl" aria-label="right" @click="press('ArrowRight')">→</button>
      </div>

      <div class="flex min-w-0 flex-col gap-4 @3xl:col-start-2 @3xl:row-start-1 @3xl:row-span-2">
        <HumanResults
          v-if="phase === 'over'"
          :entries="entries"
          :score="finalScore"
          :history="history"
          :new-best="newBest"
          @again="begin"
          @watch="emit('exit')"
        />
        <template v-else>
          <HumanRace :entries="entries" :score="score" />
          <div class="grid grid-cols-3 gap-2">
            <div class="rounded-lg border border-line bg-sunken px-3 py-2">
              <p class="text-[10px] tracking-wide text-fg-subtle uppercase">steps</p>
              <p class="num text-lg">{{ stepCount }}</p>
            </div>
            <div class="rounded-lg border border-line bg-sunken px-3 py-2">
              <p class="text-[10px] tracking-wide text-fg-subtle uppercase">your best</p>
              <p class="num text-lg text-gold-300">{{ history.best }}</p>
            </div>
            <div class="rounded-lg border border-line bg-sunken px-3 py-2">
              <p class="text-[10px] tracking-wide text-fg-subtle uppercase">games</p>
              <p class="num text-lg">{{ history.games.length }}</p>
            </div>
          </div>
          <p class="text-xs text-fg-subtle">
            Same board and rules the algorithms play. Arrow keys are absolute directions for you;
            they're translated into the same left / straight / right turns a policy chooses between.
          </p>
          <button class="btn-ghost btn-sm w-fit" @click="emit('exit')">← Back to watching</button>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.count-enter-active,
.count-leave-active {
  transition: all 0.25s ease;
}
.count-enter-from {
  opacity: 0;
  transform: scale(1.6);
}
.count-leave-to {
  opacity: 0;
  transform: scale(0.6);
}
</style>

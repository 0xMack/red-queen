<script setup lang="ts">
// A strategy playing bandit games, one pull per tick (docs/design/0011) -- what the game page's Watch stage and the
// Learn chapter's labs both show: the floor (the machine it picks is outlined, then pulled), playback controls, the
// table of values it keeps, every pull on a tape, Thompson sampling's belief curves, and the reveal when a game ends
// (then a new game starts). Changing any prop starts a new game.
const props = withDefaults(
  defineProps<{
    scenario: string
    strategy: string
    params?: string
    /** "none.v1" (one row) or "lamp.v1" (a row per lamp colour). */
    observer?: string
    label?: string
  }>(),
  { params: "", observer: "none.v1", label: "strategy" },
)

const info = computed(() => scenarioById(props.scenario))
const run = useBanditRun()
const paused = ref(false)
const speed = ref(1)
const tickMs = computed(() => 520 / speed.value)
const chosen = ref<number | null>(null)
let timer: ReturnType<typeof setTimeout> | null = null

function newGame() {
  if (timer) clearTimeout(timer)
  chosen.value = null
  run
    .start({ scenario: props.scenario, observer: props.observer, strategy: props.strategy, params: props.params, seed: 1 + Math.floor(Math.random() * 1e9) })
    .then(schedule)
}

// Pick (outlined for a moment), then pull. A finished game shows its reveal, then starts over.
function schedule() {
  if (timer) clearTimeout(timer)
  timer = setTimeout(tick, run.done.value ? 3400 : tickMs.value)
}
function tick() {
  if (paused.value) return schedule()
  if (run.done.value) return newGame()
  const arm = run.choose()
  if (arm === null) return
  chosen.value = arm
  timer = setTimeout(() => {
    run.pull(arm)
    chosen.value = null
    schedule()
  }, tickMs.value * 0.35)
}

onMounted(newGame)
onUnmounted(() => timer && clearTimeout(timer))
watch(() => [props.scenario, props.strategy, props.params, props.observer], newGame)

const rowLabels = computed(() => rowNames(info.value, props.observer))
// A strategy that values what comes next (a Q-table with gamma > 0).
const looksAhead = computed(() => Number(/gamma=([\d.]+)/.exec(props.params)?.[1] ?? 0) > 0)
const kind = computed(() => (props.strategy === "gradient" ? "preference" : "estimate"))
const curves = computed(() => {
  if (props.strategy !== "thompson" || !info.value.binary) return null
  const b = run.beliefs.value[run.currentRow()]
  if (!b) return null
  // Thompson reports the posterior mean (1 + wins) / (2 + pulls): back to wins
  return { wins: b.counts.map((n, a) => Math.round(b.values[a]! * (n + 2) - 1)), pulls: b.counts }
})
</script>

<template>
  <!-- A container, not the viewport: the player sits in a full-width stage, a chapter column and a run page's side panel. -->
  <div class="@container space-y-5">
    <UiEmpty v-if="run.error.value">{{ run.error.value }}</UiEmpty>
    <template v-else>
      <BanditFloor :run="run" :scenario="info" :chosen="chosen" show-beliefs :spin-ms="Math.min(360, tickMs * 0.6)" :revealed="run.done.value" />
      <PlaybackControls :paused="paused" :speed="speed" new-game-label="New game" @update:paused="(p) => (paused = p)" @update:speed="(s) => (speed = s)" @new-game="newGame" />

      <div class="grid gap-6 @3xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div class="min-w-0">
          <p class="label mb-2">What it believes · the table it keeps</p>
          <BanditTable
            :beliefs="run.beliefs.value"
            :current="run.currentRow()"
            :arm="chosen ?? run.history.value.at(-1)?.arm ?? null"
            :row-labels="rowLabels"
            :binary="info.binary"
            :kind="kind"
          />
          <p class="mt-2 text-xs leading-relaxed text-fg-subtle">
            <template v-if="info.sequential && looksAhead">
              A row per room. Each value is the payout <em>plus γ times the best value of the room it leads to</em> -- so the
              detour's value can be high although it never pays a thing. That is the Bellman update.
            </template>
            <template v-else-if="info.sequential">A row per room -- but each value is only what the machine pays, so the detour (which pays nothing) looks worthless.</template>
            <template v-else-if="observer === 'lamp.v1'">Two rows: separate values for each lamp colour -- a contextual bandit.</template>
            <template v-else-if="info.contexts > 1">One row: it can't tell the lamps apart, so each value averages both colours -- and every machine averages the same.</template>
            <template v-else>One row, because every pull is the same situation. A Q-table is this with a row per situation: Snake's has 2,048.</template>
          </p>
        </div>
        <div class="min-w-0">
          <p class="label mb-2">Every pull · tall = a win</p>
          <BanditTape
            :pulls="run.history.value"
            :budget="run.budget.value"
            :binary="info.binary"
            :lamps="info.contexts > 1"
            :drift="run.done.value ? (run.reveal.value?.drift?.at ?? null) : null"
          />
          <template v-if="curves">
            <p class="label mt-4 mb-2">Thompson's beliefs · where each win rate could be</p>
            <BeliefCurves :wins="curves.wins" :pulls="curves.pulls" :truth="run.done.value ? run.means() : null" />
          </template>
          <template v-if="run.done.value && run.reveal.value">
            <p class="label mt-4 mb-2">The reveal</p>
            <BanditReveal :reveal="run.reveal.value" :scenario="info" :players="[{ label, counts: run.counts.value }]" />
          </template>
        </div>
      </div>
    </template>
  </div>
</template>

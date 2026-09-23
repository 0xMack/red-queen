<script setup lang="ts">
import { brainFromApi, cellOf, CheckersEngine, squareOf, STATIC_STRATEGIES, wasmStrategy, type BrainPayload, type CheckersBrain, type CheckersPosition } from "~/utils/checkersEngine"
import { useMetricsStreamStore } from "~/stores/metricsStream"
import type { RunInfo } from "~/types/telemetry"

// The Checkers viewer for a training run (docs/design/0006): the run's champion -- the newest one, or a
// pinned generation -- playing a chosen opponent on the board, with the same stage the game page uses
// (VersusStage), so it can also be measured in the arena and taken over by a human. The Snake
// counterpart is WatchChampion; this is what makes a checkers run watchable instead of "no viewer for
// this representation yet". A champion is a position evaluator (layered or NEAT) searched to the run's
// `search_depth`: the backend serves it as plain numbers (`.../brain`) and the Rust `evaluator` plays it.
const props = withDefaults(
  defineProps<{
    runId: string
    // The run page already runs the metrics stream for its charts; the watch page has to start it.
    manageStream?: boolean
    pinnedGeneration?: number | null
    // A generation slider (the watch page has no chart to click).
    scrubber?: boolean
    compact?: boolean
  }>(),
  { manageStream: true, pinnedGeneration: null, scrubber: false, compact: false },
)
const emit = defineEmits<{ unpin: [] }>()

const api = useApi()
const stream = useMetricsStreamStore()
const run = ref<RunInfo | null>(null)
const loadError = ref<string | null>(null)

const history = computed(() => (stream.runId === props.runId ? stream.history : []))
const latest = computed(() => history.value.at(-1) ?? null)
// The slider's own choice, until the user drags back to the end (follow the newest).
const scrubbed = ref<number | null>(null)
const generation = computed(() => props.pinnedGeneration ?? scrubbed.value ?? latest.value?.generation ?? null)
const stats = computed(() => history.value.find((h) => h.generation === generation.value) ?? null)
const following = computed(() => props.pinnedGeneration === null && scrubbed.value === null)

// The champion's brain, fetched once per artifact.
const champions = shallowRef<Record<string, CheckersBrain>>({})
const championError = ref<string | null>(null)
async function ensureChampion(ref: string) {
  if (champions.value[ref]) return
  try {
    const payload = await api.fetch<BrainPayload>(`/runs/${props.runId}/artifacts/${ref}/brain`)
    champions.value = { ...champions.value, [ref]: brainFromApi(payload) }
    championError.value = null
  } catch (e) {
    championError.value = `couldn't load champion ${ref}: ${e instanceof Error ? e.message : String(e)}`
  }
}
watch(stats, (s) => s && ensureChampion(s.champion_ref), { immediate: true })

// The champion sits under one stable id, so a seat that holds it keeps holding "the champion" as the
// generation changes -- the session rebuilds the bot and the game in progress carries on.
const strategies = computed(() => {
  const s = stats.value
  const brain = s && champions.value[s.champion_ref]
  if (!s || !brain) return STATIC_STRATEGIES
  const depth = Number(run.value?.config?.search_depth ?? 1)
  const shape = brain.kind === "layered" ? `${brain.layerSizes.join("→")} network` : "NEAT graph"
  const champion = wasmStrategy({
    id: "champion",
    kind: "evaluator",
    label: `Champion (gen ${s.generation})`,
    description: `Generation ${s.generation} of run ${props.runId.slice(0, 8)}: a ${shape}${depth > 1 ? `, searching ${depth} plies` : ""}, training fitness ${s.best_fitness.toFixed(2)}${
      s.held_out_score != null ? `, ${s.held_out_score.toFixed(2)} on games it never trained on` : ""
    }.`,
    brain,
    depth,
  })
  return [champion, ...STATIC_STRATEGIES]
})

// The champion (Red) against the first opponent it should beat; games repeat until you stop them.
const session = useVersusSession<CheckersPosition>(() => CheckersEngine.create(), strategies, {
  defaultSeats: ["champion", "material-1"],
  autoRestartMs: 1800,
})
const { seats, path, starts, targets } = session

const flipped = computed(() => seats.value[0] === "human" && seats.value[1] !== "human")
const asSquares = (cells: number[]) => cells.map(squareOf)
const { board, trail, turn, banner, pieceCounts: scores } = useCheckersBoardView(session)

async function start() {
  try {
    run.value = await api.fetch<RunInfo>(`/runs/${props.runId}`)
    if (run.value.config?.game !== "checkers") {
      loadError.value = `run ${props.runId} isn't a checkers run (config.game = ${String(run.value.config?.game ?? "unset")})`
      return
    }
    if (props.manageStream) await stream.start(props.runId)
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e)
  }
}
onMounted(() => {
  start()
  session.load()
})
onUnmounted(() => props.manageStream && stream.stop())

const generations = computed(() => history.value.map((h) => h.generation))
function onScrub(event: Event) {
  const value = Number((event.target as HTMLInputElement).value)
  scrubbed.value = value >= (latest.value?.generation ?? 0) ? null : value
  if (props.pinnedGeneration !== null) emit("unpin")
}
</script>

<template>
  <div>
    <p v-if="loadError" class="card border-queen-500/40 p-4 text-sm text-queen-300">{{ loadError }}</p>
    <template v-else>
      <div class="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <span v-if="stats" class="chip">generation {{ stats.generation }}</span>
        <span v-if="stats" class="text-fg-muted">
          training fitness <span class="num font-semibold text-fg">{{ stats.best_fitness.toFixed(2) }}</span>
          <template v-if="stats.held_out_score != null">
            · unseen games <span class="num font-semibold text-fg">{{ stats.held_out_score.toFixed(2) }}</span>
          </template>
        </span>
        <span v-else class="text-fg-subtle">Waiting for the first generation…</span>
        <span v-if="run?.status === 'running' && following" class="chip text-life-300">live: newest champion</span>
        <button v-if="!following" class="link ml-auto text-xs" @click="scrubbed = null; emit('unpin')">follow the newest</button>
      </div>

      <label v-if="scrubber && generations.length > 1" class="mb-4 flex items-center gap-3 text-xs text-fg-subtle">
        Generation
        <input
          type="range"
          class="flex-1 accent-queen-500"
          :min="generations[0]"
          :max="generations.at(-1)"
          :value="generation ?? 0"
          @input="onScrub"
        />
        <span class="num w-24 text-right">{{ generation }} / {{ generations.at(-1) }}</span>
      </label>
      <p v-if="championError" class="mb-3 text-xs text-queen-300">{{ championError }}</p>

      <VersusStage :session="session" :scores="scores" :player-colors="['#ef3b5d', '#e9ebf1']" :arena-defaults="['champion', 'material-1']" :compact="compact">
        <template #board>
          <CheckersBoard
            :state="board"
            :flipped="flipped"
            :selected="path.length ? squareOf(path.at(-1)!) : null"
            :starts="path.length ? [] : asSquares(starts)"
            :targets="asSquares(targets)"
            :trail="trail"
            :turn="turn"
            :banner="banner"
            @square="(x, y) => session.clickCell(cellOf(x, y))"
          />
        </template>
        <template #arena-note>Pick <em>Champion</em> as A to measure this generation.</template>
      </VersusStage>
    </template>
  </div>
</template>

<script setup lang="ts">
import type { GameEntry } from "~/data/games"
import { noDevice } from "~/games/device"
import type { GameModule } from "~/games/types"
import type { GameBoard } from "~/composables/useGameBoard"
import type { DeviceFit } from "~/types/modelpack"

// The one game page (docs/design/0007), for every game: header and Watch/Play toggle; a live stage with a
// ranked side leaderboard; then the full leaderboard, cost tradeoffs, head-to-head and the
// representations. What differs per game -- what a score means, the stage itself, extra columns, wording --
// is the game's `GameModule` (app/games/); this file has no game in it. Algorithms first: the leaderboard
// (produced by an evaluation job on held-out games) decides which entrant is on the stage, and clicking
// any entrant -- in the side list, the full table, the cost chart or the matrix -- puts it on. State
// lives in the URL (?watch=<entrant>&mode=play) so a view can be linked.
const props = defineProps<{ game: GameEntry; module: GameModule; board: GameBoard }>()

const route = useRoute()
const router = useRouter()
const { entries, protocol, protocolInfo, interfaces, interfacesById, error } = props.board

// --- What this device can run (docs/design/0009) -------------------------------------------------------
const device = (props.module.device ?? (() => noDevice()))(props.game.slug, entries)
onMounted(device.load)
const onlyRunnable = ref(false)
const runsHere = device.runsHere as Ref<Record<string, DeviceFit>>
const listed = computed(() => (onlyRunnable.value ? entries.value.filter((r) => runsHere.value[r.entrant_id]?.ok !== false) : entries.value))

// --- What's on the stage -------------------------------------------------------------------------------
const mode = computed<"watch" | "play">(() => (route.query.mode === "play" || entries.value.length === 0 ? "play" : "watch"))
// Snake defaults to the best *trained* entrant this device can run -- the point is watching something that
// learned -- falling back to the best trained one, then the top entrant; a versus game shows its strongest
// player. The list shows plainly when a baseline ranks above the one on stage.
const defaultEntrant = computed(() => {
  if (props.module.defaultEntrant === "top") return entries.value[0] ?? null
  return (
    entries.value.find((r) => r.entrant_kind === "champion" && runsHere.value[r.entrant_id]?.ok !== false) ??
    entries.value.find((r) => r.entrant_kind === "champion") ??
    entries.value[0] ??
    null
  )
})
const selected = computed(() => entries.value.find((r) => r.entrant_id === route.query.watch) ?? defaultEntrant.value)
const selectedRank = computed(() => (selected.value ? entries.value.indexOf(selected.value) + 1 : null))
// Is the selected entrant what the stage is about? Always when watching; when playing only if Play is
// against an entrant (versus games).
const entrantOnStage = computed(() => mode.value === "watch" || props.module.selectInPlay === "stay")

const stage = ref<HTMLElement | null>(null)
// The list scrolls now, so whoever is selected (from the table, the cost chart, a link) must be brought into view.
const aside = ref<HTMLElement | null>(null)
watch(
  () => selected.value?.entrant_id,
  () => nextTick(() => aside.value?.querySelector('[data-selected="true"]')?.scrollIntoView({ block: "nearest" })),
)
function selectEntrant(entrantId: string) {
  const stay = mode.value === "play" && props.module.selectInPlay === "stay"
  router.replace({ query: { ...route.query, watch: entrantId, mode: stay ? "play" : undefined } })
  stage.value?.scrollIntoView({ behavior: "smooth", block: "start" })
}
function setMode(next: "watch" | "play") {
  router.replace({ query: { ...route.query, mode: next === "play" ? "play" : undefined } })
}

// --- The human, for the side leaderboard ---------------------------------------------------------------
const human = ref<{ score: number; label: string; live: boolean } | null>(null)
onMounted(() => {
  human.value = props.module.initialHuman?.(props.game.slug) ?? null
})
function onHumanScore(score: number, live: boolean, label?: string) {
  human.value = { score, label: label ?? "You", live }
}
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 pt-8 pb-6 sm:px-6 lg:px-8">
    <!-- Header -->
    <UiSectionHeader :level="1" :eyebrow="game.tagline" :title="game.title" :back="{ to: '/games', label: 'Games' }">
      {{ mode === "watch" ? module.copy.watch : module.copy.play }}
      <template #actions>
        <UiSegmented
          size="md"
          :model-value="mode"
          accent="play"
          aria-label="Watch or play"
          :options="[
            { value: 'watch', label: 'Watch the models', disabled: entries.length === 0 },
            { value: 'play', label: 'Play it yourself' },
          ]"
          @update:model-value="setMode"
        />
      </template>
    </UiSectionHeader>

    <UiPanel v-if="error" class="mt-6 border-queen-500/40 text-sm text-queen-200">
      Couldn't load the leaderboard ({{ error.message }}), so there's nothing to rank or watch -- you can still play. It's
      produced by an evaluation job (<code class="chip">jobs/evaluate*.py</code>) and served by apis/backend.
    </UiPanel>

    <!-- Stage + side leaderboard -->
    <div ref="stage" class="mt-8 grid scroll-mt-20 grid-cols-[minmax(0,1fr)] gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
      <UiPanel ticks pad="md" class="min-w-0">
        <template v-if="entrantOnStage && selected" #header>
          <EntrantHeader
            :entry="selected"
            :rank="selectedRank!"
            :total="entries.length"
            :score="module.score"
            :interfaces-by-id="interfacesById"
          />
        </template>
        <template v-else #header>
          <span class="label text-queen-300">{{ mode === "play" ? "Your game" : "Stage" }}</span>
        </template>
        <ClientOnly>
          <component
            :is="mode === 'watch' ? module.Watch : module.Play"
            :key="`${mode}:${selected?.entrant_id ?? 'none'}`"
            :mode="mode"
            :entry="selected"
            :entries="entries"
            :device="device"
            @score="onHumanScore"
            @exit="setMode('watch')"
          />
        </ClientOnly>
      </UiPanel>

      <!-- Bounded: the list scrolls inside the panel, so a long leaderboard doesn't stretch the row (and with it the
           stage beside it) -- the page is only as tall as the stage's own content makes it. -->
      <UiPanel
        label="Standings"
        pad="none"
        fill
        class="h-fit max-h-[36rem] xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)]"
      >
        <template #actions>
          <a href="#leaderboard" class="text-xs text-fg-subtle transition hover:text-fg">full table ↓</a>
        </template>
        <div ref="aside" class="flex min-h-0 flex-1 flex-col p-3">
          <label v-if="entries.length && device.filterable.value" class="mb-2 flex shrink-0 items-center gap-2 px-1 text-xs text-fg-subtle">
            <input v-model="onlyRunnable" type="checkbox">
            Only what runs on this device
            <span v-if="device.summary.value" class="ml-auto truncate font-mono text-[10px]">{{ device.summary.value }}</span>
          </label>
          <div v-if="listed.length" class="-mr-2 min-h-0 flex-1 overflow-y-auto pr-2">
            <LeaderboardList
              :entries="listed"
              :score="module.score"
              :runs-here="runsHere"
              :selected-id="entrantOnStage ? selected?.entrant_id : null"
              :human="human"
              @select="selectEntrant"
            />
          </div>
          <UiEmpty v-else compact>No evaluations yet.</UiEmpty>
        </div>
        <template #footer>
          {{ module.copy.scoreNote({ episodes: protocolInfo?.episodes, protocol }) }}
          Click an entrant to {{ mode === "play" && module.selectInPlay === "stay" ? "play against" : "watch" }} it.
          <template v-if="human">Your row is this browser only.</template>
        </template>
      </UiPanel>
    </div>

    <!-- Full leaderboard -->
    <section v-if="entries.length" id="leaderboard" class="mt-24 scroll-mt-20">
      <UiSectionHeader eyebrow="Leaderboard" :index="2" title="Every entrant, every measurement">
        {{ module.copy.leaderboardIntro({ episodes: protocolInfo?.episodes, seeds: protocolInfo?.held_out_seeds }) }}
        Baselines are always included, so you can tell whether a score is actually good.
      </UiSectionHeader>
      <LeaderboardTable
        class="mt-8"
        :entries="entries"
        :interfaces-by-id="interfacesById"
        :score="module.score"
        :columns="module.columns"
        :selected-id="selected?.entrant_id"
        @select="selectEntrant"
      />
      <p class="mt-3 max-w-4xl text-xs leading-relaxed text-fg-subtle">
        "~" marks training cost estimated for runs recorded before cost tracking existed; "≈" marks a rank whose 95% interval
        overlaps the one above it. Timings are from
        {{ [...new Set(entries.map((e) => e.hardware.hardware_class).filter(Boolean))].join(", ") || "unknown hardware" }}
        -- only compare them within the same hardware class.
      </p>

      <div v-if="module.sections.headToHead" class="mt-8">
        <LeaderboardMatrix :entries="entries" :selected-id="selected?.entrant_id" @select="selectEntrant" />
      </div>
      <div v-if="module.sections.pareto" class="mt-8">
        <LeaderboardPareto :entries="entries" @select="selectEntrant" />
      </div>

      <div v-if="module.sections.representations && interfaces.length" class="mt-24">
        <UiSectionHeader eyebrow="Representations" :index="3" title="How entrants see the game">
          An interface is an observer (what the model sees) plus an action adapter (how its outputs become moves). A model
          only runs under the interface it was trained for, so representations are compared by training separate models, not
          by swapping weights.
        </UiSectionHeader>
        <RepresentationCards class="mt-8" :interfaces="interfaces" :entries="entries" />
      </div>
    </section>
  </main>
</template>

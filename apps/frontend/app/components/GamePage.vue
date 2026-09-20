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
  <main class="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8 xl:pl-16">
    <!-- Back to the games hub. On wide screens it floats in the left margin, out of the content's vertical
         flow (it used to be a row of its own above the header); on narrow ones it rides on the eyebrow line. -->
    <NuxtLink
      to="/games"
      class="fixed top-20 left-3 z-20 hidden size-10 items-center justify-center rounded-full border border-line bg-surface/90 text-fg-muted shadow backdrop-blur transition hover:border-line-strong hover:text-fg xl:flex"
      title="All games"
      aria-label="Back to all games"
    >
      &larr;
    </NuxtLink>

    <!-- Header -->
    <div class="flex flex-wrap items-end justify-between gap-4">
      <div>
        <p class="eyebrow">
          <NuxtLink to="/games" class="mr-2 text-fg-subtle normal-case transition hover:text-fg xl:hidden">&larr; Games</NuxtLink>
          {{ game.tagline }}
        </p>
        <h1 class="mt-2 text-3xl font-semibold sm:text-4xl">{{ game.title }}</h1>
        <p class="mt-2 max-w-3xl text-fg-muted">{{ mode === "watch" ? module.copy.watch : module.copy.play }}</p>
      </div>
      <div class="flex rounded-xl border border-line bg-surface p-1">
        <button
          class="rounded-lg px-4 py-2 text-sm font-medium transition"
          :class="mode === 'watch' ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
          :disabled="entries.length === 0"
          @click="setMode('watch')"
        >
          👁 Watch the algorithms
        </button>
        <button
          class="rounded-lg px-4 py-2 text-sm font-medium transition"
          :class="mode === 'play' ? 'bg-queen-500 text-white shadow-[0_0_20px_-4px_rgb(239_59_93/0.8)]' : 'text-fg-muted hover:text-fg'"
          @click="setMode('play')"
        >
          🎮 Play it yourself
        </button>
      </div>
    </div>

    <p v-if="error" class="card mt-6 border-queen-500/40 p-4 text-sm text-queen-300">
      Couldn't load the leaderboard ({{ error.message }}), so there's nothing to rank or watch -- you can still play. It's
      produced by an evaluation job (<code class="chip">jobs/evaluate*.py</code>) and served by apis/backend.
    </p>

    <!-- Stage + side leaderboard -->
    <div ref="stage" class="mt-6 grid scroll-mt-20 gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
      <section class="card p-5">
        <EntrantHeader
          v-if="entrantOnStage && selected"
          :entry="selected"
          :rank="selectedRank!"
          :total="entries.length"
          :score="module.score"
          :interfaces-by-id="interfacesById"
        />
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
      </section>

      <!-- Bounded: the list scrolls inside the card, so a long leaderboard doesn't stretch the row (and with it the
           stage card beside it) -- the page is only as tall as the stage's own content makes it. -->
      <aside ref="aside" class="card flex h-fit max-h-[34rem] flex-col p-4 xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)]">
        <div class="mb-3 flex shrink-0 items-baseline justify-between px-1">
          <h2 class="font-display text-base font-semibold">🏆 Leaderboard</h2>
          <a href="#leaderboard" class="text-xs text-fg-subtle hover:text-fg">full table ↓</a>
        </div>
        <label v-if="entries.length && device.filterable.value" class="mb-2 flex shrink-0 items-center gap-2 px-1 text-xs text-fg-subtle">
          <input v-model="onlyRunnable" type="checkbox" class="accent-queen-500" />
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
        <p v-else class="px-1 text-sm text-fg-subtle">No evaluations yet.</p>
        <p class="mt-3 shrink-0 px-1 text-[11px] leading-relaxed text-fg-subtle">
          {{ module.copy.scoreNote({ episodes: protocolInfo?.episodes, protocol }) }}
          Click an entrant to {{ mode === "play" && module.selectInPlay === "stay" ? "play against" : "watch" }} it.
          <template v-if="human">Your row is this browser only.</template>
        </p>
      </aside>
    </div>

    <!-- Full leaderboard -->
    <section v-if="entries.length" id="leaderboard" class="mt-16 scroll-mt-20">
      <p class="eyebrow">Leaderboard</p>
      <h2 class="mt-2 text-2xl font-semibold">Every entrant, every measurement</h2>
      <p class="mt-2 max-w-3xl text-fg-muted">
        {{ module.copy.leaderboardIntro({ episodes: protocolInfo?.episodes, seeds: protocolInfo?.held_out_seeds }) }}
        Baselines are always included, so you can tell whether a score is actually good.
      </p>
      <LeaderboardTable
        class="mt-6"
        :entries="entries"
        :interfaces-by-id="interfacesById"
        :score="module.score"
        :columns="module.columns"
        :selected-id="selected?.entrant_id"
        @select="selectEntrant"
      />
      <p class="mt-3 text-xs text-fg-subtle">
        "~" marks training cost estimated for runs recorded before cost tracking existed; "≈" marks a rank whose 95% interval
        overlaps the one above it. Timings are from
        {{ [...new Set(entries.map((e) => e.hardware.hardware_class).filter(Boolean))].join(", ") || "unknown hardware" }}
        -- only compare them within the same hardware class.
      </p>

      <div v-if="module.sections.headToHead" class="mt-6">
        <LeaderboardMatrix :entries="entries" :selected-id="selected?.entrant_id" @select="selectEntrant" />
      </div>
      <div v-if="module.sections.pareto" class="mt-6">
        <LeaderboardPareto :entries="entries" @select="selectEntrant" />
      </div>

      <div v-if="module.sections.representations && interfaces.length" class="mt-12">
        <p class="eyebrow">Representations</p>
        <h2 class="mt-2 text-2xl font-semibold">How entrants see the game</h2>
        <p class="mt-2 max-w-3xl text-fg-muted">
          An interface is an observer (what the model sees) plus an action adapter (how its outputs become moves). A model
          only runs under the interface it was trained for, so representations are compared by training separate models, not
          by swapping weights.
        </p>
        <RepresentationCards class="mt-6" :interfaces="interfaces" :entries="entries" />
      </div>
    </section>
  </main>
</template>

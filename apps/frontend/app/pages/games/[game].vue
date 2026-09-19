<script setup lang="ts">
import { checkersSnapshot } from "~/data/checkersSnapshot"
import { games } from "~/data/games"
import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"
import type { DeviceFit } from "~/types/modelpack"

// One page per game (docs/design/0007). Watching algorithms comes first: the leaderboard (evaluated
// by jobs/evaluate.py on held-out games) decides which entrant is on the board, and clicking any
// entrant -- in the side list, the full table, or the cost chart -- puts it on instead. "Play it
// yourself" swaps the same stage to human control, races the player against every entrant's mean,
// and ends with how the score compares. State lives in the URL (?watch=<entrant>&mode=play) so a
// view can be linked.
const route = useRoute()
const router = useRouter()
const slug = route.params.game as string
const game = games.find((g) => g.slug === slug)
const config = useRuntimeConfig()
useHead({ title: game?.title ?? slug })

// Only Snake has a playable/watchable implementation (and an evaluation protocol) so far.
const playable = slug === "snake"

const { data: records, error: leaderboardError } = await useAsyncData(`leaderboard-${slug}`, () =>
  playable ? $fetch<EvaluationRecord[]>(`/games/${slug}/leaderboard`, { baseURL: config.public.apiBase }) : Promise.resolve([]),
)
const { data: interfaceList } = await useAsyncData(`interfaces-${slug}`, () =>
  playable
    ? $fetch<InterfaceInfo[]>(`/games/${slug}/interfaces`, { baseURL: config.public.apiBase }).catch(() => [])
    : Promise.resolve([]),
)

// Records from different protocol versions aren't comparable -- use the newest one.
const protocol = computed(() => [...new Set((records.value ?? []).map((r) => r.protocol))].sort().at(-1) ?? null)
const entries = computed(() => (records.value ?? []).filter((r) => r.protocol === protocol.value))
const interfacesById = computed(() => Object.fromEntries((interfaceList.value ?? []).map((i) => [i.id, i])))
const protocolInfo = computed(() => entries.value[0]?.metrics.protocol)

// --- Which models this device can run (docs/design/0009) ---------------------------------------
// Published champions run as model packages in ONNX Runtime; the catalog + a device probe say which
// ones can, and why not for the rest. Baselines and unpublished champions run the Python code.
const models = useModelCatalog(slug)
onMounted(models.load)
const onlyRunnable = ref(false)

const runsHere = computed<Record<string, DeviceFit>>(() => {
  const fits: Record<string, DeviceFit> = {}
  for (const r of entries.value) {
    const available = models.availability.value[r.entrant_id]
    if (available?.match?.ok) fits[r.entrant_id] = { ok: true, note: `${available.match.variant.id} · ${available.match.backend}` }
    else if (available?.match) fits[r.entrant_id] = { ok: false, note: available.match.summary }
    else if (!available && models.catalog.value) fits[r.entrant_id] = { ok: true, note: "Python (not a published package)" }
  }
  return fits
})
const listed = computed(() => (onlyRunnable.value ? entries.value.filter((r) => runsHere.value[r.entrant_id]?.ok !== false) : entries.value))

// --- What's on the stage ------------------------------------------------------------------------
const mode = computed<"watch" | "play">(() => (route.query.mode === "play" || entries.value.length === 0 ? "play" : "watch"))
// Default: the best *trained* model this device can run -- the point is watching something that
// learned -- falling back to the best trained model, then the top entrant. The list shows plainly
// when a baseline ranks above it.
const defaultEntrant = computed(
  () =>
    entries.value.find((r) => r.entrant_kind === "champion" && runsHere.value[r.entrant_id]?.ok !== false) ??
    entries.value.find((r) => r.entrant_kind === "champion") ??
    entries.value[0] ??
    null,
)
const selected = computed(
  () => entries.value.find((r) => r.entrant_id === route.query.watch) ?? defaultEntrant.value,
)
const selectedRank = computed(() => (selected.value ? entries.value.indexOf(selected.value) + 1 : null))

const stage = ref<HTMLElement | null>(null)
function watchEntrant(entrantId: string) {
  router.replace({ query: { ...route.query, watch: entrantId, mode: undefined } })
  stage.value?.scrollIntoView({ behavior: "smooth", block: "start" })
}
function setMode(next: "watch" | "play") {
  router.replace({ query: { ...route.query, mode: next === "play" ? "play" : undefined } })
}

// --- The human, for the side leaderboard -------------------------------------------------------
const human = ref<{ score: number; label: string; live: boolean } | null>(null)
onMounted(() => {
  const history = loadHumanHistory(slug)
  if (history.games.length) human.value = { score: history.best, label: "You (best)", live: false }
})
function onHumanScore(score: number, live: boolean) {
  if (live) human.value = { score, label: "You", live: true }
  else {
    const history = loadHumanHistory(slug)
    human.value = { score, label: score >= history.best ? "You (best)" : "You", live: false }
  }
}
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 py-8 sm:px-6 lg:px-8">
    <NuxtLink to="/games" class="text-sm text-fg-subtle transition hover:text-fg">&larr; Games</NuxtLink>

    <!-- Unknown / not-yet-playable game -->
    <div v-if="!game" class="card mt-8 p-8 text-center text-fg-muted">
      No game called “{{ slug }}”. <NuxtLink to="/games" class="link">See all games</NuxtLink>.
    </div>
    <template v-else-if="!playable">
      <div class="mt-4">
        <p class="eyebrow">{{ game.tagline }}</p>
        <h1 class="mt-2 text-3xl font-semibold sm:text-4xl">{{ game.title }}</h1>
        <p class="mt-2 max-w-3xl text-fg-muted">{{ game.summary }}</p>
      </div>
      <div class="card mt-8 grid items-center gap-8 p-6 md:grid-cols-[minmax(0,360px)_minmax(0,1fr)]">
        <CheckersBoard v-if="game.art === 'checkers'" :state="checkersSnapshot" />
        <p class="text-fg-muted">
          Rules and the strategy-vs-strategy match framework are built and tested; watching, playing,
          and a rating-based leaderboard for this game are next (docs/design/0006, 0007).
        </p>
      </div>
    </template>

    <template v-else>
      <!-- Header -->
      <div class="mt-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p class="eyebrow">{{ game.tagline }}</p>
          <h1 class="mt-2 text-3xl font-semibold sm:text-4xl">{{ game.title }}</h1>
          <p class="mt-2 max-w-3xl text-fg-muted">
            <template v-if="mode === 'watch'">
              Algorithms first: pick any entrant on the leaderboard to watch it play, then see if you can
              beat them.
            </template>
            <template v-else>Same board, same rules, same game -- you're steering this time.</template>
          </p>
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

      <p v-if="leaderboardError" class="card mt-6 border-queen-500/40 p-4 text-sm text-queen-300">
        Couldn't load the leaderboard ({{ leaderboardError.message }}), so there's nothing to watch --
        you can still play. It's produced by <code class="chip">uv run python jobs/evaluate.py</code> and
        served by apis/backend.
      </p>

      <!-- Stage + side leaderboard -->
      <div ref="stage" class="mt-6 grid scroll-mt-20 gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section class="card p-5">
          <template v-if="mode === 'watch' && selected">
            <div class="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2">
              <span class="flex items-center gap-2">
                <span class="size-2.5 rounded-full" :style="{ background: entrantColor(selected) }" />
                <span class="font-display text-lg font-semibold">{{ selected.label }}</span>
              </span>
              <span class="chip">#{{ selectedRank }} of {{ entries.length }}</span>
              <span class="chip" :style="{ color: LEVEL_COLORS[selected.metrics.model.observer_level] }">
                L{{ selected.metrics.model.observer_level }} {{ interfacesById[selected.interface]?.observer.level_name }}
              </span>
              <span class="text-sm text-fg-muted">
                held-out <span class="num font-semibold text-fg">{{ selected.metrics.quality.mean.toFixed(2) }}</span>
                <span class="num text-fg-subtle"> ± {{ selected.metrics.quality.ci95.toFixed(2) }}</span>
              </span>
              <NuxtLink v-if="selected.run_id" :to="`/runs/${selected.run_id}`" class="link ml-auto text-sm">Training run →</NuxtLink>
            </div>
            <ClientOnly>
              <WatchChampion
                :key="selected.entrant_id"
                :run-id="selected.run_id ?? undefined"
                :packaged="models.availability.value[selected.entrant_id] ?? null"
                :packaged-loading="!models.catalog.value && !models.error.value"
                @model-failed="(f) => models.reportFailure(f.packageId, f.variantId, f.backend as 'wasm' | 'webgpu', f.message)"
                @retry-failed="models.retryFailed"
                :baseline="
                  baselineName(selected)
                    ? { name: baselineName(selected)!, interface: selected.interface, description: selected.metrics.model.description }
                    : undefined
                "
              />
            </ClientOnly>
          </template>
          <ClientOnly v-else>
            <HumanPlay :game="slug" :entries="entries" @score="onHumanScore" @exit="setMode('watch')" />
          </ClientOnly>
        </section>

        <aside class="card h-fit p-4 xl:sticky xl:top-20">
          <div class="mb-3 flex items-baseline justify-between px-1">
            <h2 class="font-display text-base font-semibold">🏆 Leaderboard</h2>
            <a href="#leaderboard" class="text-xs text-fg-subtle hover:text-fg">full table ↓</a>
          </div>
          <label v-if="entries.length && models.catalog.value" class="mb-2 flex items-center gap-2 px-1 text-xs text-fg-subtle">
            <input v-model="onlyRunnable" type="checkbox" class="accent-queen-500" />
            Only what runs on this device
            <span v-if="models.profile.value" class="ml-auto truncate font-mono text-[10px]" :title="models.profile.value.webgpu.reason ?? ''">
              {{ models.profile.value.webgpu.available ? "WebGPU" : "no WebGPU" }} ·
              {{ models.profile.value.wasm ? `WASM${models.profile.value.threads ? " threads" : ""}` : "no WASM" }}
            </span>
          </label>
          <LeaderboardList
            v-if="listed.length"
            :entries="listed"
            :runs-here="runsHere"
            :selected-id="mode === 'watch' ? selected?.entrant_id : null"
            :human="human"
            @select="watchEntrant"
          />
          <p v-else class="px-1 text-sm text-fg-subtle">No evaluations yet.</p>
          <p class="mt-3 px-1 text-[11px] leading-relaxed text-fg-subtle">
            Mean score over {{ protocolInfo?.episodes ?? 200 }} held-out games each ({{ protocol }}).
            Click an entrant to watch it.
            <template v-if="human">Your row is this browser only.</template>
          </p>
        </aside>
      </div>

      <!-- Full leaderboard -->
      <section v-if="entries.length" id="leaderboard" class="mt-16 scroll-mt-20">
        <p class="eyebrow">Leaderboard</p>
        <h2 class="mt-2 text-2xl font-semibold">Every entrant, every measurement</h2>
        <p class="mt-2 max-w-3xl text-fg-muted">
          Ranked by mean score on {{ protocolInfo?.episodes }} held-out games (seeds
          {{ protocolInfo?.held_out_seeds[0] }}–{{ protocolInfo?.held_out_seeds[1] }}) -- never by
          training fitness -- alongside what each cost to train and to run. Baselines are always
          included, so you can tell whether a score is actually good.
        </p>
        <LeaderboardTable
          class="mt-6"
          :entries="entries"
          :interfaces-by-id="interfacesById"
          :selected-id="mode === 'watch' ? selected?.entrant_id : null"
          @select="watchEntrant"
        />
        <p class="mt-3 text-xs text-fg-subtle">
          "~" marks training cost estimated for runs recorded before cost tracking existed; "≈" marks a
          rank whose 95% interval overlaps the one above it. Timings are from
          {{ [...new Set(entries.map((e) => e.hardware.hardware_class).filter(Boolean))].join(", ") || "unknown hardware" }}
          -- only compare them within the same hardware class.
        </p>

        <div class="mt-6">
          <LeaderboardPareto :entries="entries" @select="watchEntrant" />
        </div>

        <div class="mt-12">
          <p class="eyebrow">Representations</p>
          <h2 class="mt-2 text-2xl font-semibold">How entrants see the game</h2>
          <p class="mt-2 max-w-3xl text-fg-muted">
            An interface is an observer (what the model sees) plus an action adapter (how its outputs
            become moves). A model only runs under the interface it was trained for, so representations
            are compared by training separate models, not by swapping weights.
          </p>
          <RepresentationCards class="mt-6" :interfaces="interfaceList ?? []" :entries="entries" />
        </div>
      </section>
    </template>
  </main>
</template>

<script setup lang="ts">
import { games } from "~/data/games"
import type { UiStat } from "~/types/ui"
import { chapterForRepresentation, learnChapters } from "~/data/learnChapters"
import { gameModules } from "~/games/registry"

useHead({ title: "" })

const runsStore = useRunsStore()
await useAsyncData("runs", () => runsStore.ensureLoaded().then(() => true))
const [snake, checkers, bandit] = await Promise.all([useGameBoard("snake"), useGameBoard("checkers"), useGameBoard("bandit")])
const now = useNow()

// Fig. 1 is the Snake leaderboard's best *trained* entrant -- ranked on held-out games, not by its own training
// fitness (which isn't comparable across algorithms). Without a leaderboard, the best recorded Snake run.
const topEntry = computed(() => snake.entries.value.find((r) => r.entrant_kind === "champion" && r.run_id) ?? null)
const featuredRunId = computed(() => {
  if (topEntry.value) return topEntry.value.run_id!
  const snakeRuns = runsStore.runs.filter((r) => describeRun(r).game === "snake" && describeRun(r).watchable && bestFitnessOf(r) !== null)
  return [...snakeRuns].sort((a, b) => (bestFitnessOf(b) ?? -Infinity) - (bestFitnessOf(a) ?? -Infinity))[0]?.run_id ?? null
})
const featuredRun = computed(() => runsStore.runs.find((r) => r.run_id === featuredRunId.value) ?? null)
const featuredMeta = computed(() => (featuredRun.value ? describeRun(featuredRun.value) : null))
const featuredChapter = computed(() => (featuredMeta.value ? chapterForRepresentation(featuredMeta.value.representation) : null))

// WatchChampion owns the metrics stream for the featured run, so its training curve is right here.
const metricsStream = useMetricsStreamStore()
const featuredHistory = computed(() => (metricsStream.runId === featuredRunId.value ? metricsStream.history : []))

const featuredFacts = computed<UiStat[]>(() => {
  const e = topEntry.value
  const m = featuredMeta.value
  const facts: UiStat[] = []
  if (e) {
    facts.push(
      { label: "Rank", value: `#${snake.entries.value.indexOf(e) + 1} of ${snake.entries.value.length}` },
      { label: "Held-out score", value: `${e.metrics.quality.mean.toFixed(1)} ± ${e.metrics.quality.ci95.toFixed(1)}`, tone: "life" },
      { label: "Sees", value: e.interface.split("/")[1]?.split("+")[0] ?? e.interface },
    )
  }
  if (m) {
    facts.push({ label: "Trained by", value: m.representationLabel })
    if (m.network) facts.push({ label: "Network", value: m.network, wide: !e })
    if (e?.metrics.inference.parameters ?? m.parameterCount) facts.push({ label: "Weights", value: (e?.metrics.inference.parameters ?? m.parameterCount)!.toLocaleString() })
  }
  const steps = e?.metrics.training.env_steps
  if (steps) facts.push({ label: "Experience", value: `${formatSteps(steps)} moves` })
  const wall = e?.metrics.training.active_s ?? e?.metrics.training.wall_s
  if (wall) facts.push({ label: "Training time", value: formatDuration(wall) })
  return facts
})

// The hero's results: each game's leader against its baselines. Every board's fetch starts before any is awaited.
const boards = { snake, checkers, bandit }
const results = computed(() =>
  games
    .filter((g) => gameModules[g.slug] && boards[g.slug as keyof typeof boards])
    .map((g) => ({ game: g, entries: boards[g.slug as keyof typeof boards].entries.value, score: gameModules[g.slug]!.score })),
)

const principles = [
  {
    title: "Built from scratch",
    body: "Genetic programming, neuroevolution, NEAT, Q-learning to PPO, automatic differentiation and a transformer -- no ML framework underneath any of it, so every mechanism is something you can read.",
  },
  {
    title: "Measured honestly",
    body: "Every model is ranked on games it never trained on, next to hand-written baselines -- so a score says whether something actually learned, and what it cost to get there.",
  },
  {
    title: "Running in your browser",
    body: "The same Rust game core the training uses, compiled to WebAssembly, with models in ONNX Runtime on your own device. Watch them play, train one yourself, or take them on.",
  },
]

const recentRuns = computed(() => runsStore.runs.slice(0, 6))
</script>

<template>
  <main>
    <!-- Hero: what this is (left), how to start (right), then where things stand -- each game's leader against its
         baselines, live from the leaderboards. -->
    <section class="mx-auto max-w-[1600px] px-4 pt-10 sm:px-6 lg:px-8 lg:pt-14">
      <div class="grid gap-x-12 gap-y-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:items-end">
        <div>
          <p class="eyebrow">A laboratory for evolution &amp; learning</p>
          <h1 class="mt-4 max-w-[17ch] text-[2.5rem] leading-[1.02] tracking-[-0.02em] sm:text-[3.4rem] xl:text-[4.25rem]">
            Watch algorithms <em class="text-queen-400">learn to play</em>, live.
          </h1>
        </div>
        <div class="lg:pb-1.5">
          <p class="text-[17px] leading-relaxed text-fg-muted">
            Evolution and reinforcement learning, built from first principles -- no ML frameworks -- and pitted against each
            other on purpose-built games. Every model runs in your browser, drawn as it thinks.
          </p>
          <div class="mt-5 flex flex-wrap items-center gap-3">
            <NuxtLink to="/games/snake" class="btn-primary">Watch them play →</NuxtLink>
            <NuxtLink to="/learn" class="btn-ghost">Learn how they work</NuxtLink>
          </div>
        </div>
      </div>

      <div class="mt-10">
        <div class="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <p class="label">Where things stand · live from the leaderboards</p>
          <p class="text-xs text-fg-subtle">Ranked on held-out games, never on training fitness -- against hand-written baselines.</p>
        </div>
        <div class="grid gap-4 md:grid-cols-3">
          <LeaderResult v-for="r in results" :key="r.game.slug" :game="r.game" :entries="r.entries" :score="r.score" />
        </div>
      </div>
    </section>

    <div class="mx-auto max-w-[1600px] px-4 sm:px-6 lg:px-8">
      <!-- Fig. 1 -->
      <section class="mt-12">
        <UiPanel ticks pad="none">
          <template #header>
            <span class="flex items-center gap-2">
              <span class="size-1.5 animate-live-pulse rounded-full bg-queen-400" />
              <span class="label text-queen-300">Fig. 1 · Now playing</span>
            </span>
            <span v-if="topEntry" class="text-sm font-semibold">{{ entrantShortLabel(topEntry) }}</span>
            <span v-if="topEntry" class="hidden text-sm text-fg-subtle sm:inline">-- the best-ranked model, playing games it has never seen</span>
          </template>
          <template #actions>
            <NuxtLink to="/games/snake" class="btn-ghost btn-sm">Leaderboard →</NuxtLink>
          </template>

          <div class="grid xl:grid-cols-[minmax(0,1fr)_320px]">
            <div class="p-4 sm:p-5">
              <ClientOnly v-if="featuredRunId">
                <WatchChampion :run-id="featuredRunId" />
              </ClientOnly>
              <div v-else class="grid items-center gap-8 md:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
                <LiveSnakeDemo />
                <div class="text-sm text-fg-muted">
                  <p class="font-display text-2xl text-fg">No trained champion to show yet</p>
                  <p class="mt-2">
                    {{ runsStore.error ? "The backend isn't reachable" : "No Snake runs have been recorded" }},
                    so here's the game itself -- click it and steer with the arrow keys. Start the backend
                    and run <code class="chip">uv run python jobs/snake_neuro_run.py</code> to watch a trained
                    policy here instead.
                  </p>
                </div>
              </div>
            </div>

            <aside v-if="featuredRunId" class="flex flex-col gap-5 border-t border-line p-5 xl:border-t-0 xl:border-l">
              <div>
                <p class="label">Specimen</p>
                <p class="mt-1.5 font-display text-2xl leading-tight">{{ topEntry ? entrantShortLabel(topEntry) : featuredMeta?.title }}</p>
                <p class="font-mono text-[11px] text-fg-subtle">
                  run {{ shortId(featuredRunId) }}<template v-if="featuredRun"> · {{ formatRelative(featuredRun.created_at, now) }}</template>
                </p>
              </div>
              <UiStats :items="featuredFacts" :cols="2" :ruled="false" />
              <div v-if="featuredHistory.length > 1">
                <p class="label mb-2">Training curve · {{ featuredMeta?.terms.fitness }}</p>
                <Sparkline :values="featuredHistory.map((h) => h.best_fitness)" class="h-14 w-full" />
              </div>
              <div class="mt-auto flex flex-col gap-2 border-t border-line pt-4 text-sm">
                <NuxtLink :to="`/runs/${featuredRunId}`" class="link w-fit">Open the training run</NuxtLink>
                <NuxtLink v-if="featuredChapter" :to="featuredChapter.path" class="link w-fit">How {{ featuredMeta?.representationLabel }} works</NuxtLink>
              </div>
            </aside>
          </div>
        </UiPanel>
      </section>

      <!-- Principles -->
      <section class="mt-24 grid gap-10 md:grid-cols-3 md:gap-8">
        <div v-for="(p, i) in principles" :key="p.title" class="border-t border-line-strong pt-5">
          <p class="font-display text-lg text-queen-400 italic">{{ ["i.", "ii.", "iii."][i] }}</p>
          <h3 class="mt-1 font-display text-[1.75rem] font-normal">{{ p.title }}</h3>
          <p class="mt-2 text-[15px] leading-relaxed text-fg-muted">{{ p.body }}</p>
        </div>
      </section>

      <!-- Games + recent runs -->
      <section class="mt-28 grid gap-12 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div>
          <UiSectionHeader eyebrow="The arena" :index="2" title="Games" to="/games" link-label="All games" />
          <div class="mt-6 grid gap-5 sm:grid-cols-2">
            <GameCard v-for="game in games" :key="game.slug" :game="game" compact />
          </div>
        </div>
        <div>
          <UiSectionHeader eyebrow="Telemetry" :index="3" title="Recent runs" to="/runs" link-label="All runs" />
          <UiPanel class="mt-6" pad="none">
            <div class="divide-y divide-line">
              <RunListItem
                v-for="run in recentRuns"
                :key="run.run_id"
                :run="run"
                :trend="runsStore.summaries[run.run_id]?.trend"
                :best="bestFitnessOf(run) ?? runsStore.summaries[run.run_id]?.best_fitness ?? null"
                :now="now"
              />
              <p v-if="recentRuns.length === 0" class="px-4 py-10 text-center text-sm text-fg-subtle">
                {{ runsStore.error ? "Backend offline -- start apis/backend to see runs." : "No runs recorded yet." }}
              </p>
            </div>
          </UiPanel>
        </div>
      </section>

      <!-- Learn -->
      <section class="mt-28">
        <UiSectionHeader eyebrow="The interactive textbook" :index="4" title="Learn how it works" to="/learn" link-label="All chapters">
          Foundations first. Each chapter cites the real code and the real results -- including what didn't work.
        </UiSectionHeader>
        <div class="mt-8 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
          <ChapterCard v-for="(chapter, i) in learnChapters.slice(0, 4)" :key="chapter.slug" :chapter="chapter" :number="i + 1" />
        </div>
      </section>
    </div>
  </main>
</template>

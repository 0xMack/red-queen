<script setup lang="ts">
import { games } from "~/data/games"
import { learnChapters } from "~/data/learnChapters"

useHead({ title: "" })

const runsStore = useRunsStore()
await useAsyncData("runs", () => runsStore.fetchRuns().then(() => runsStore.runs))

// The featured champion: a Snake run that's training right now if there is one (you get to watch
// it improve live), otherwise the best finished one.
const featured = computed(() => {
  const snakeRuns = runsStore.runs.filter((r) => describeRun(r).watchable && bestFitnessOf(r) !== null)
  const live = snakeRuns.find((r) => r.status === "running" && !isStale(r))
  if (live) return { run: live, live: true }
  const best = [...snakeRuns].sort((a, b) => (bestFitnessOf(b) ?? -Infinity) - (bestFitnessOf(a) ?? -Infinity))[0]
  return best ? { run: best, live: false } : null
})
const featuredMeta = computed(() => (featured.value ? describeRun(featured.value.run) : null))

// WatchChampion owns the metrics stream for the featured run, so its history is right here.
const metricsStream = useMetricsStreamStore()
const featuredHistory = computed(() =>
  featured.value && metricsStream.runId === featured.value.run.run_id ? metricsStream.history : [],
)

const recentRuns = computed(() => runsStore.runs.slice(0, 5))
onMounted(() => runsStore.fetchHistories(recentRuns.value.map((r) => r.run_id)))

const stats = computed(() => ({
  runs: runsStore.runs.length,
  bestSnake: runsStore.runs
    .filter((r) => describeRun(r).game === "snake")
    .reduce<number | null>((best, r) => {
      const f = bestFitnessOf(r)
      return f !== null && (best === null || f > best) ? f : best
    }, null),
  chapters: learnChapters.filter((c) => c.status === "available").length,
}))

const highlights = [
  {
    icon: "{ }",
    title: "Built from scratch",
    body: "Genetic algorithms, neuroevolution, automatic differentiation, and a transformer -- no ML framework underneath any of it, so every mechanism is something you can actually read.",
  },
  {
    icon: "◎",
    title: "Tested against real games",
    body: "Not toy benchmarks alone -- a policy either learns to eat food and avoid walls in Snake, or it honestly doesn't, and the site says which.",
  },
  {
    icon: "◉",
    title: "Watch it happen, live",
    body: "Every training run streams its progress in real time, and a trained policy plays its game right in the browser -- the same Python code, via Pyodide, not a JS reimplementation.",
  },
]

const now = Date.now() / 1000
</script>

<template>
  <main>
    <!-- Hero -->
    <section class="relative overflow-hidden border-b border-line">
      <div class="pointer-events-none absolute -top-48 left-1/2 h-[520px] w-[900px] -translate-x-1/2 rounded-full bg-queen-500/10 blur-3xl" />
      <div class="relative mx-auto grid max-w-[1600px] gap-10 px-4 pt-16 pb-12 sm:px-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:items-end lg:px-8 lg:pt-24">
        <div>
          <p class="eyebrow">Evolution &amp; learning, from scratch</p>
          <h1 class="mt-4 text-4xl leading-[1.05] font-semibold sm:text-6xl">
            Watch algorithms <span class="bg-gradient-to-r from-queen-300 to-queen-500 bg-clip-text text-transparent">learn to play</span>, live.
          </h1>
          <p class="mt-6 max-w-2xl text-lg text-fg-muted">
            Red Queen is a lab for reinforcement learning and evolutionary algorithms -- custom
            implementations, tested against purpose-built games, with real-time visualization of
            what's actually happening inside them.
          </p>
          <div class="mt-8 flex flex-wrap items-center gap-3">
            <NuxtLink to="/learn" class="btn-primary">Start learning →</NuxtLink>
            <NuxtLink to="/runs" class="btn-ghost">Browse training runs</NuxtLink>
            <NuxtLink to="/games/snake" class="px-2 text-sm text-fg-subtle transition hover:text-fg">or take on the algorithms at Snake</NuxtLink>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <StatTile label="Training runs" :value="stats.runs || '--'" hint="recorded in telemetry" />
          <StatTile label="Best Snake fitness" :value="formatFitness(stats.bestSnake, 2)" tone="queen" hint="evolved, not hand-coded" />
          <StatTile label="Chapters" :value="stats.chapters" hint="of the interactive textbook" />
          <StatTile label="ML frameworks" value="0" tone="life" hint="every gradient hand-rolled" />
        </div>
      </div>
    </section>

    <div class="mx-auto max-w-[1600px] px-4 sm:px-6 lg:px-8">
      <!-- Now playing -->
      <section class="mt-12">
        <div class="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p class="eyebrow flex items-center gap-2">
              <span class="size-1.5 animate-live-pulse rounded-full bg-life-400" />
              {{ featured?.live ? "Training right now" : "Now playing" }}
            </p>
            <h2 class="mt-2 text-2xl font-semibold sm:text-3xl">An evolved neural network, playing Snake</h2>
            <p class="mt-2 max-w-3xl text-fg-muted">
              Nobody wrote this policy. It's the champion of a population of networks bred over
              hundreds of generations -- no gradients, just mutation and selection. Its network is drawn alongside,
              lighting up with every decision.
            </p>
          </div>
          <NuxtLink v-if="featured" :to="`/runs/${featured.run.run_id}`" class="btn-ghost btn-sm">Open the full run →</NuxtLink>
        </div>

        <div class="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div class="card p-5">
            <ClientOnly v-if="featured">
              <WatchChampion :run-id="featured.run.run_id" />
            </ClientOnly>
            <div v-else class="grid items-center gap-8 md:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
              <LiveSnakeDemo />
              <div class="text-sm text-fg-muted">
                <p class="font-display text-lg font-semibold text-fg">No trained champion to show yet</p>
                <p class="mt-2">
                  {{ runsStore.error ? "The backend isn't reachable" : "No Snake runs have been recorded" }},
                  so here's the game itself -- click it and steer with the arrow keys. Start the backend
                  and run <code class="chip">uv run python jobs/snake_neuro_run.py</code> to watch an evolved
                  policy here instead.
                </p>
              </div>
            </div>
          </div>

          <!-- About this champion -->
          <aside v-if="featured && featuredMeta" class="card flex flex-col gap-5 p-5 xl:self-start">
            <div>
              <p class="text-[11px] tracking-wide text-fg-subtle uppercase">About this champion</p>
              <p class="mt-1 text-lg font-semibold">{{ featuredMeta.title }}</p>
              <p class="font-mono text-[11px] text-fg-subtle">{{ shortId(featured.run.run_id) }} · {{ formatRelative(featured.run.created_at, now) }}</p>
            </div>
            <dl class="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
              <div>
                <dt class="text-[11px] text-fg-subtle">Network</dt>
                <dd class="font-mono text-fg">{{ featuredMeta.network ?? "--" }}</dd>
              </div>
              <div>
                <dt class="text-[11px] text-fg-subtle">Parameters</dt>
                <dd class="num text-fg">{{ featuredMeta.parameterCount ?? "--" }}</dd>
              </div>
              <div>
                <dt class="text-[11px] text-fg-subtle">Selection</dt>
                <dd class="font-mono text-fg">{{ featuredMeta.selection ?? "--" }}</dd>
              </div>
              <div>
                <dt class="text-[11px] text-fg-subtle">Population</dt>
                <dd class="num text-fg">{{ featuredMeta.populationSize ?? "--" }}</dd>
              </div>
              <div class="col-span-2">
                <dt class="text-[11px] text-fg-subtle">Mutation</dt>
                <dd class="font-mono text-fg">{{ featuredMeta.variation ?? "--" }}</dd>
              </div>
              <div>
                <dt class="text-[11px] text-fg-subtle">Generations</dt>
                <dd class="num text-fg">{{ featuredHistory.length || featuredMeta.targetGenerations || "--" }}</dd>
              </div>
              <div>
                <dt class="text-[11px] text-fg-subtle">Best fitness</dt>
                <dd class="num text-queen-300">{{ formatFitness(bestFitnessOf(featured.run, featuredHistory), 2) }}</dd>
              </div>
            </dl>
            <div>
              <p class="mb-1.5 text-[11px] text-fg-subtle">Best fitness over training</p>
              <Sparkline :values="featuredHistory.map((h) => h.best_fitness)" class="h-14 w-full" />
            </div>
            <ol class="space-y-2 border-t border-line pt-4 text-xs text-fg-muted">
              <li class="flex gap-2"><span class="num text-queen-300">1</span>11 sensor features in: danger ahead/left/right, heading, food direction.</li>
              <li class="flex gap-2"><span class="num text-queen-300">2</span>A small tanh network scores 3 moves: turn left, straight, turn right.</li>
              <li class="flex gap-2"><span class="num text-queen-300">3</span>Fitness = reward across 5 benchmark games; lexicase picks the parents.</li>
            </ol>
            <NuxtLink to="/learn/teaching-a-snake" class="link text-sm">Read how it was trained →</NuxtLink>
          </aside>
        </div>
      </section>

      <!-- Highlights -->
      <section class="mt-20 grid gap-4 md:grid-cols-3">
        <div v-for="item in highlights" :key="item.title" class="card p-6">
          <span class="flex size-10 items-center justify-center rounded-lg border border-queen-400/30 bg-queen-500/10 font-mono text-queen-300">
            {{ item.icon }}
          </span>
          <h3 class="mt-4 text-lg font-semibold">{{ item.title }}</h3>
          <p class="mt-2 text-sm leading-relaxed text-fg-muted">{{ item.body }}</p>
        </div>
      </section>

      <!-- Games + recent runs -->
      <section class="mt-20 grid gap-10 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div>
          <div class="flex items-end justify-between">
            <div>
              <p class="eyebrow">Arena</p>
              <h2 class="mt-2 text-2xl font-semibold">Games</h2>
            </div>
            <NuxtLink to="/games" class="link text-sm">All games →</NuxtLink>
          </div>
          <div class="mt-5 grid gap-5 sm:grid-cols-2">
            <GameCard v-for="game in games" :key="game.slug" :game="game" compact />
          </div>
        </div>

        <div>
          <div class="flex items-end justify-between">
            <div>
              <p class="eyebrow">Telemetry</p>
              <h2 class="mt-2 text-2xl font-semibold">Recent runs</h2>
            </div>
            <NuxtLink to="/runs" class="link text-sm">All runs →</NuxtLink>
          </div>
          <div class="card mt-5 divide-y divide-line/60">
            <NuxtLink
              v-for="run in recentRuns"
              :key="run.run_id"
              :to="`/runs/${run.run_id}`"
              class="flex items-center gap-4 px-4 py-3 transition hover:bg-raised/60"
            >
              <div class="min-w-0 flex-1">
                <p class="truncate text-sm font-medium">{{ describeRun(run).title }}</p>
                <p class="font-mono text-[11px] text-fg-subtle">{{ shortId(run.run_id) }} · {{ formatRelative(run.created_at, now) }}</p>
              </div>
              <Sparkline :values="(runsStore.histories[run.run_id] ?? []).map((h) => h.best_fitness)" class="hidden h-8 w-24 sm:block" />
              <span class="num w-14 text-right text-sm">{{ formatFitness(bestFitnessOf(run, runsStore.histories[run.run_id]), 2) }}</span>
              <StatusBadge :status="run.status" :stale="isStale(run, now)" />
            </NuxtLink>
            <p v-if="recentRuns.length === 0" class="px-4 py-8 text-center text-sm text-fg-subtle">
              {{ runsStore.error ? "Backend offline -- start apis/backend to see runs." : "No runs recorded yet." }}
            </p>
          </div>
        </div>
      </section>

      <!-- Learn -->
      <section class="mt-20">
        <div class="flex items-end justify-between">
          <div>
            <p class="eyebrow">The interactive textbook</p>
            <h2 class="mt-2 text-2xl font-semibold">Learn how it works</h2>
          </div>
          <NuxtLink to="/learn" class="link text-sm">All chapters →</NuxtLink>
        </div>
        <div class="mt-5 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
          <ChapterCard
            v-for="(chapter, i) in learnChapters.slice(0, 4)"
            :key="chapter.slug"
            :chapter="chapter"
            :number="i + 1"
          />
        </div>
      </section>
    </div>
  </main>
</template>

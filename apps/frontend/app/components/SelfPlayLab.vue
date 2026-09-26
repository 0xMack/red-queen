<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import { probeDevice } from "~/inference/device"
import { cellOf, CheckersEngine, squareOf, STATIC_STRATEGIES, wasmStrategy, type CheckersPosition } from "~/utils/checkersEngine"

// Checkers by self-play, live (docs/design/0010 Phase 4): a position-value network learning from games against
// itself in the reader's browser (the Rust TD(λ) loop, WebAssembly, in a worker). The curve is its points per game,
// searching 3 plies, against material search at the same depth and against a random mover. Below it, the same
// Checkers stage the game page uses, with the network you just trained as one of the players: pick "You" for a seat
// and play it.

const DEPTH = 3
const BUDGET = 30_000
const EVAL_EVERY = 2_500

const config = reactive({ lambda: 0.7, seed: 0 })
const lab = useSelfPlayLab()
const live = ref<boolean | null>(null)
onMounted(async () => {
  live.value = (await probeDevice()).wasm
  session.load()
})

function train() {
  lab.start({ params: `lambda=${config.lambda}`, seed: config.seed, budget: BUDGET, evalEvery: EVAL_EVERY, depth: DEPTH })
}
const started = computed(() => lab.status.value !== "idle")

const chart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  const p = lab.points.value
  return {
    x: p.map((q) => q.games),
    series: [
      { key: "material", label: `vs material search, ${DEPTH} plies each`, color: "#ff5c7a", values: p.map((q) => q.material), width: 2.5 },
      { key: "random", label: "vs a random mover", color: "#60a5fa", values: p.map((q) => q.random), width: 1.5 },
      { key: "even", label: "even (0.5)", color: "#6b7489", values: p.map(() => 0.5), width: 1, dashed: true },
    ],
  }
})

// The trained network joins the stage's players as soon as there is one; its id stays the same as it improves, so a
// seat holding it keeps holding "your network".
const strategies = computed(() => {
  const n = lab.network.value
  if (!n) return STATIC_STRATEGIES
  const trained = wasmStrategy({
    id: "trained",
    kind: "evaluator",
    label: `Your network (${n.games.toLocaleString()} games)`,
    description: `The network trained above by self-play, after ${n.games.toLocaleString()} games against itself, searching ${DEPTH} plies.`,
    brain: { kind: "layered", weights: n.weights, layerSizes: n.layerSizes },
    depth: DEPTH,
  })
  return [trained, ...STATIC_STRATEGIES]
})
const session = useVersusSession<CheckersPosition>(() => CheckersEngine.create(), strategies, {
  defaultSeats: ["trained", "material-1"],
  autoRestartMs: 1800,
})
const { path, starts, targets, seats } = session
const flipped = computed(() => seats.value[0] === "human" && seats.value[1] !== "human")
const asSquares = (cells: number[]) => cells.map(squareOf)
const { board, trail, turn, banner, pieceCounts: scores } = useCheckersBoardView(session)
</script>

<template>
  <div class="not-prose my-6" data-self-play-lab>
    <div class="card p-4 sm:p-5">
      <p v-if="live === false" class="text-sm text-fg-muted">
        Training live needs WebAssembly, which this browser can't run. The Checkers stage below still works: the self-play network on the
        <NuxtLink to="/games/checkers" class="link">leaderboard</NuxtLink> is one of its players there.
      </p>
      <template v-else>
        <div class="flex flex-wrap items-center gap-2">
          <button v-if="lab.status.value === 'running'" class="btn-primary btn-sm" @click="lab.pause">Pause</button>
          <button v-else-if="lab.status.value === 'paused'" class="btn-primary btn-sm" @click="lab.resume">Resume</button>
          <button class="btn-sm" :class="lab.status.value === 'running' || lab.status.value === 'paused' ? 'btn-ghost' : 'btn-primary'" :disabled="live !== true" @click="train">
            {{ started ? "Train again" : "Train by self-play" }}
          </button>
          <label class="ml-auto flex items-center gap-2 text-sm">
            <span class="text-fg-muted">λ</span>
            <select v-model.number="config.lambda" class="rounded-md border border-line bg-surface px-2 py-1 text-xs" aria-label="lambda">
              <option :value="0">0 (one-step TD)</option>
              <option :value="0.7">0.7</option>
              <option :value="1">1 (Monte-Carlo)</option>
            </select>
          </label>
          <label class="flex items-center gap-2 text-sm">
            <span class="text-fg-muted">seed</span>
            <select v-model.number="config.seed" class="rounded-md border border-line bg-surface px-2 py-1 text-xs" aria-label="random seed">
              <option v-for="s in 5" :key="s" :value="s - 1">{{ s - 1 }}</option>
            </select>
          </label>
        </div>
        <dl class="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
          <div><dt class="text-fg-subtle">games against itself</dt><dd class="num text-fg">{{ lab.games.value.toLocaleString() }} / {{ BUDGET.toLocaleString() }}</dd></div>
          <div><dt class="text-fg-subtle">plies per game</dt><dd class="num text-fg">{{ lab.meanPlies.value === null ? "--" : lab.meanPlies.value.toFixed(0) }}</dd></div>
          <div><dt class="text-fg-subtle">draws</dt><dd class="num text-fg">{{ lab.drawRate.value === null ? "--" : `${(lab.drawRate.value * 100).toFixed(0)}%` }}</dd></div>
          <div><dt class="text-fg-subtle">TD loss</dt><dd class="num text-fg">{{ lab.loss.value === null ? "--" : lab.loss.value.toFixed(4) }}</dd></div>
        </dl>
        <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
          <span v-for="s in chart.series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }}</span>
        </div>
        <LineChart v-if="lab.points.value.length > 1" class="mt-1" :x="chart.x" :series="chart.series" :height="170" x-label="self-play games" :format="(v: number) => v.toFixed(2)" />
        <p v-else class="mt-2 rounded-lg border border-dashed border-line p-6 text-center text-xs text-fg-subtle">
          Points per game (win 1, draw ½), 20 games against each opponent, every {{ EVAL_EVERY.toLocaleString() }} self-play games.
        </p>
        <p v-if="lab.error.value" class="mt-2 text-xs text-queen-300">{{ lab.error.value }}</p>
      </template>
    </div>

    <VersusStage class="mt-4" :session="session" :scores="scores" :player-colors="['#ef3b5d', '#e9ebf1']" :arena-defaults="['trained', 'material-3']" compact>
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
      <template #arena-note>Pick <em>Your network</em> as A to measure it against anything else.</template>
    </VersusStage>
  </div>
</template>

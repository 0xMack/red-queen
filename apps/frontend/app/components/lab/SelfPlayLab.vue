<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import { cellOf, CheckersEngine, squareOf, STATIC_STRATEGIES, wasmStrategy, type CheckersPosition } from "~/utils/checkersEngine"
import * as F from "~/data/math/self-play"

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
const live = useWasmSupport()
onMounted(() => session.load())

// The targets it learns from, on top (docs/design/0012), with λ linked to its select.
const scope = useOrProvideTermScope()
watch(
  () => config.lambda,
  (lambda) => (scope.values.value = { lambda }),
  { immediate: true },
)

function train() {
  lab.start({ params: `lambda=${config.lambda}`, seed: config.seed, budget: BUDGET, evalEvery: EVAL_EVERY, depth: DEPTH })
}

const chart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  const p = lab.points.value
  return {
    x: p.map((q) => q.games),
    series: [
      { key: "material", label: `vs material search, ${DEPTH} plies each`, color: palette.queen400, values: p.map((q) => q.material), width: 2.5 },
      { key: "random", label: "vs a random mover", color: palette.signal400, values: p.map((q) => q.random), width: 1.5 },
      { key: "even", label: "even (0.5)", color: palette.fgSubtle, values: p.map(() => 0.5), width: 1, dashed: true },
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
  <div class="not-prose" data-self-play-lab>
    <LabFrame :live="live" title="Checkers by self-play · TD(λ)" split="none" class="mb-4">
      <template #recorded>
        <p class="text-sm text-fg-muted">
          Training live needs WebAssembly, which this browser can't run. The Checkers stage below still works: the self-play network on the
          <NuxtLink to="/games/checkers" class="link">leaderboard</NuxtLink> is one of its players there.
        </p>
      </template>

      <template #formula>
        <MathFormula :formula="F.lambdaReturn" bare />
      </template>

      <LabControls
        :status="lab.status.value"
        :disabled="live !== true"
        train-label="Train by self-play"
        :hint="false"
        @train="train"
        @pause="lab.pause"
        @resume="lab.resume"
      >
        <div class="flex flex-wrap gap-x-6 gap-y-2.5">
          <UiSelect
            v-bind="scope.target('lambda')"
            v-model="config.lambda"
            class="w-48"
            label="λ"
            :options="[{ value: 0, label: '0 (one-step TD)' }, { value: 0.7, label: '0.7' }, { value: 1, label: '1 (Monte-Carlo)' }]"
          />
          <UiSelect v-model="config.seed" class="w-24" label="seed" :options="Array.from({ length: 5 }, (_, i) => i)" />
        </div>
      </LabControls>
      <UiStats
        class="mt-4"
        :cols="4"
        :items="[
          { label: 'games vs itself', value: `${lab.games.value.toLocaleString()} / ${BUDGET.toLocaleString()}` },
          { label: 'plies per game', value: lab.meanPlies.value === null ? '--' : lab.meanPlies.value.toFixed(0) },
          { label: 'draws', value: lab.drawRate.value === null ? '--' : `${(lab.drawRate.value * 100).toFixed(0)}%` },
          { label: 'TD loss', value: lab.loss.value === null ? '--' : lab.loss.value.toFixed(4) },
        ]"
      />
      <LabCurve
        class="mt-4"
        :x="chart.x"
        :series="chart.series"
        :show="lab.points.value.length > 1"
        x-label="self-play games"
        :format="(v) => v.toFixed(2)"
        :empty="`Points per game (win 1, draw ½), 20 games against each opponent, every ${EVAL_EVERY.toLocaleString()} self-play games.`"
      />
      <LabStatus :status="lab.status.value" :error="lab.error.value" />
    </LabFrame>

    <VersusStage :session="session" :scores="scores" :player-colors="[palette.queen500, palette.fg]" :arena-defaults="['trained', 'material-3']" compact>
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

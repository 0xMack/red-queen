<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import type { Candidates } from "~/types/versus"

// How each bot valued the move it chose, ply by ply -- a chart per seat that scores its moves, side by
// side, each on its own scale (a network's tanh output and a material count aren't comparable, so they
// aren't plotted together). A steady climb is a plan working; a sudden drop is where it walked into
// something. Game-agnostic: it only needs the session's decisions. Each chart sits in a fixed-height box
// so the panel doesn't change size as points arrive.
const props = defineProps<{ decisions: Candidates[]; players: readonly string[]; colors?: string[] }>()

const WIN = 500 // |score| beyond this is "won/lost outright" (a sentinel, not a magnitude) -- clamped for the plot
const CHART_HEIGHT = 120

const traces = computed(() =>
  props.players.map((name, player) => {
    const own = props.decisions.filter((d) => d.player === player && d.chosen !== null)
    const raw = own.map((d) => d.scores[d.chosen!]!)
    const finite = raw.filter((v) => Math.abs(v) < WIN)
    const cap = (finite.length ? Math.max(...finite.map(Math.abs)) : 1) * 1.25 || 1
    return {
      name,
      player,
      ready: own.length >= 2,
      x: own.map((d) => d.ply + 1),
      series: [
        {
          key: `p${player}`,
          label: `${name}'s valuation of its move`,
          color: props.colors?.[player] ?? (player === 0 ? "#ef3b5d" : "#cdd3e0"),
          values: raw.map((v) => (Math.abs(v) >= WIN ? Math.sign(v) * cap : v)),
          width: 2,
        } satisfies ChartSeries,
      ],
    }
  }),
)
</script>

<template>
  <div class="rounded-lg border border-line bg-sunken p-3">
    <p class="eyebrow">Evaluation over the game</p>
    <div class="mt-2 grid gap-4 sm:grid-cols-2">
      <div v-for="t in traces" :key="t.player">
        <p class="text-[11px] text-fg-subtle">{{ t.name }}: how it valued the move it chose</p>
        <div class="mt-1" :style="{ height: `${CHART_HEIGHT}px` }">
          <ClientOnly v-if="t.ready">
            <LineChart :x="t.x" :series="t.series" :height="CHART_HEIGHT" x-label="ply" :format="(v: number) => v.toFixed(2)" />
          </ClientOnly>
          <p v-else class="flex h-full items-center justify-center rounded-md border border-dashed border-line text-xs text-fg-subtle">
            fills in as this seat's bot plays
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

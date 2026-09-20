<script setup lang="ts">
import type { VersusSession } from "~/composables/useVersusSession"

// What the players are thinking, for any versus game, straight from the session: one panel per player, side
// by side -- its candidate moves and (if it evaluates with a network) that network lit up on the position it
// chose to leave behind -- then how each valued its moves over the game. Each panel keeps its player's
// latest decision, so nothing swaps as the turn passes (a single panel that flipped between players made
// the page jump). Which bots can feed it is up to the strategies: anything with `scores()` gets candidate
// bars, anything with a `network` and `activations()` gets the diagram.
const props = defineProps<{ session: VersusSession<any>; colors?: string[] }>()

const { players, currentPlayer, done, candidatesByPlayer, networkByPlayer, decisions } = props.session
const { seatLabel } = props.session

const DEFAULT_COLORS = ["#ef3b5d", "#e9ebf1"]
const colorOf = (player: number) => props.colors?.[player] ?? DEFAULT_COLORS[player]

const panels = computed(() =>
  players.value.map((name, player) => {
    const candidates = candidatesByPlayer.value[player] ?? null
    const active = !done.value && currentPlayer.value === player
    return {
      player,
      name,
      candidates,
      network: networkByPlayer.value[player] ?? null,
      active,
      status: !candidates ? undefined : active && candidates.chosen === null ? `weighing ${candidates.labels.length}…` : "last move",
    }
  }),
)
</script>

<template>
  <div class="space-y-4">
    <div class="grid gap-4 lg:grid-cols-2">
      <VersusPlayerPanel
        v-for="p in panels"
        :key="p.player"
        :player-name="p.name"
        :strategy-label="seatLabel(p.player)"
        :color="colorOf(p.player)"
        :active="p.active"
        :status="p.status"
      >
        <div class="grid h-full gap-3" :class="p.network ? 'grid-cols-[minmax(0,2fr)_minmax(0,3fr)]' : 'grid-cols-1'">
          <VersusCandidates :candidates="p.candidates" />
          <VersusNetwork v-if="p.network" :network="p.network" :activations="p.candidates?.activations ?? null" />
        </div>
      </VersusPlayerPanel>
    </div>
    <VersusEvalTrace :decisions="decisions" :players="players" :colors="colors" />
  </div>
</template>

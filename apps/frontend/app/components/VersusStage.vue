<script setup lang="ts">
import type { VersusSession } from "~/composables/useVersusSession"

// The whole two-player stage (docs/design/0006), for any game. Left: the players and status line, the game's
// board (a slot -- the one game-specific piece), and the playback controls underneath it (pause, speed, new
// game -- the same PlaybackControls Snake's champion uses). Right: who plays each seat, this session's record,
// the game's own live stats (the `stats` slot), and the move history. Below: per-player diagnostics, an arena,
// and (optionally) the list of players. Checkers fills the board and stats slots; the next game brings its own
// and gets everything else. The session and engine are what make it game-agnostic (useVersusSession).
// Vue reads an *absent* boolean prop as false, so the on-by-default ones need explicit defaults.
const props = withDefaults(
  defineProps<{
    session: VersusSession<any>
    /** Right of each player's name in the status line: piece counts, captured tallies, ... */
    scores?: (number | string)[]
    playerColors?: string[]
    arenaDefaults?: [string, string]
    /** One column, and no move log or player list: for a narrow slot (the run page's champion column). */
    compact?: boolean
    /** List every strategy as a card at the bottom. Off where a leaderboard already describes them. */
    showPlayers?: boolean
    /** Per-player candidate moves, networks and evaluation traces under the board (default on). */
    diagnostics?: boolean
  }>(),
  { showPlayers: true, diagnostics: true, compact: false },
)

// Destructured once so the refs are top-level and unwrap in the template.
const { loading, error, players, drawNote, seats, done, winner, currentPlayer, humanTurn, paused, speed, history, record, strategies, path } = props.session
const { seatLabel, newGame, setPaused, setSeat, arena } = props.session

const anyBot = computed(() => seats.value.some((s) => s !== "human"))

const status = computed(() => {
  if (done.value) {
    if (winner.value === null) return `Draw: ${drawNote.value}`
    return `${players.value[winner.value]} (${seatLabel(winner.value)}) wins`
  }
  const who = `${players.value[currentPlayer.value]} (${seatLabel(currentPlayer.value)})`
  if (humanTurn.value) return path.value.length ? `${who}: pick where to land` : `${who}: pick a piece`
  return paused.value ? `${who} to move -- paused` : `${who} is thinking…`
})

const hint = computed(() => {
  const bots = seats.value.filter((s) => s !== "human")
  if (bots.length === 2) return "Bot vs bot: watch them play, or pause and change a seat."
  if (bots.length === 1) return strategies.value.find((s) => s.id === bots[0])?.description ?? ""
  return "Pass-and-play: both seats are human."
})

function onSeat(player: 0 | 1, seat: string) {
  setSeat(player, seat, true)
}
function onNewGame() {
  newGame()
  props.session.resetRecord()
}
</script>

<template>
  <div>
    <p v-if="error" class="card border-queen-500/40 p-4 text-sm text-queen-300">Couldn't load the game core: {{ error }}</p>
    <p v-else-if="loading" class="text-sm text-fg-subtle">Loading the game core…</p>

    <div v-show="!loading && !error" class="grid gap-6" :class="compact ? '' : 'lg:grid-cols-[minmax(0,520px)_minmax(0,1fr)]'">
      <!-- The game: players, board, playback controls -->
      <div>
        <!-- Two short lines of fixed height -- the players and their counts, then one line of status that
             truncates (full text on hover) -- so a long strategy name or a changing message never
             re-wraps the header and pushes the board (and everything under it) around. -->
        <div class="mb-1 flex items-center justify-between gap-3 text-sm">
          <span class="flex min-w-0 items-center gap-2">
            <span class="size-3 shrink-0 rounded-full" :style="{ background: playerColors?.[0] ?? '#ef3b5d' }" />
            <span class="truncate">{{ players[0] }}<template v-if="scores"> · {{ scores[0] }}</template></span>
          </span>
          <span class="flex min-w-0 items-center gap-2">
            <span class="truncate"><template v-if="scores">{{ scores[1] }} · </template>{{ players[1] }}</span>
            <span class="size-3 shrink-0 rounded-full" :style="{ background: playerColors?.[1] ?? '#e9ebf1' }" />
          </span>
        </div>
        <p class="mb-2 h-5 truncate text-sm" :class="humanTurn ? 'text-fg' : 'text-fg-muted'" :title="status" aria-live="polite" data-testid="status">
          {{ status }}
        </p>
        <slot name="board" :session="session" />
        <!-- Reserved height: a hint that appears on some plies (a forced capture) must not move the page. -->
        <div class="mt-2 min-h-10">
          <slot name="hint" :session="session" />
        </div>
        <!-- Pause and speed only mean something while a bot is playing; a game with two humans just has New game. -->
        <PlaybackControls
          class="mt-1"
          :paused="paused"
          :speed="speed"
          :can-pause="anyBot"
          :show-speed="anyBot"
          @update:paused="setPaused"
          @update:speed="speed = $event"
          @new-game="onNewGame"
        />
      </div>

      <!-- The setup and the session: seats, record, the game's own stats, history -->
      <div class="space-y-5">
        <VersusSeatPicker :players="players" :seats="seats" :strategies="strategies" @seat="onSeat" />
        <p class="-mt-2 line-clamp-2 min-h-8 text-xs text-fg-subtle" :title="hint">{{ hint }}</p>

        <div class="space-y-3">
          <p class="flex flex-wrap items-baseline gap-x-2 text-xs text-fg-subtle" data-testid="record">
            <span class="eyebrow">This session</span>
            <span class="num">{{ players[0] }} {{ record.wins[0] }} · {{ players[1] }} {{ record.wins[1] }} · Draws {{ record.draws }}</span>
          </p>
          <slot name="stats" :session="session" />
        </div>
        <slot name="controls" :session="session" />

        <VersusMoveLog v-if="!compact" :history="history" :players="players" />
      </div>
    </div>

    <!-- What the bots are thinking -->
    <div v-if="diagnostics && !compact && !loading && !error" class="mt-6">
      <VersusDiagnostics :session="session" :colors="playerColors" />
    </div>

    <VersusArena
      v-if="!error"
      :class="compact ? 'mt-6' : 'mt-8'"
      :strategies="strategies"
      :run="arena"
      :disabled="loading"
      :default-a="arenaDefaults?.[0]"
      :default-b="arenaDefaults?.[1]"
    >
      <slot name="arena-note" />
    </VersusArena>

    <VersusStrategyList v-if="!compact && showPlayers" class="mt-8" :strategies="strategies">
      <slot name="strategies-note" />
    </VersusStrategyList>
  </div>
</template>

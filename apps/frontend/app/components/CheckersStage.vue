<script setup lang="ts">
import type { GameDevice } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import { cellOf, CheckersEngine, squareOf, type CheckersPosition } from "~/utils/checkersEngine"

// Checkers' stage on the shared game page, for Watch *and* Play (told apart by `mode`): the versus
// stack (VersusStage) with a Checkers board and Checkers-specific stats and diagnostics. Its players are
// the leaderboard's entrants -- baselines and trained champions alike -- so the seats, the arena and the
// leaderboard all talk about the same things.
//   watch: the entrant on stage against the strongest *other* entrant, replaying game after game;
//   play:  you (Red) against the entrant on stage.
// Props are spelled out (the shape of `StageProps` in games/types.ts): the SFC compiler can't resolve an
// imported type used as the whole props type without TypeScript installed.
const props = defineProps<{ mode: "watch" | "play"; entry: EvaluationRecord | null; entries: EvaluationRecord[]; device: GameDevice }>()
const emit = defineEmits<{ score: [score: number, live: boolean, label?: string]; exit: [] }>()

const { strategies, failed } = useCheckersEntrants(() => props.entries)

// Seats are entrant ids. With no leaderboard (backend down) the players are the static ones, keyed by kind.
const subject = props.entry?.entrant_id ?? "material-2"
// The opponent: the best other entrant -- but if the entrant on stage is a baseline (no network), the best *evolved*
// one, so the default stage has a network to draw and a trained player to compare with the strongest baseline.
const others = props.entries.filter((e) => e.entrant_id !== subject)
const rival =
  (props.entry?.entrant_kind === "baseline" ? others.find((e) => e.entrant_kind === "champion") : undefined)?.entrant_id ??
  others[0]?.entrant_id ??
  (subject === "material-1" ? "random" : "material-1")
const watching = props.mode === "watch"

const session = useVersusSession<CheckersPosition>(() => CheckersEngine.create(), strategies, {
  defaultSeats: watching ? [subject, rival] : ["human", subject],
  autoRestartMs: watching ? 1800 : null,
})
onMounted(session.load)
const { seats, path, starts, targets, history, moves, humanTurn, record, players } = session

// The human's own pieces sit at the bottom: flip when the only human is Red (Red starts at the top).
const flipped = computed(() => seats.value[0] === "human" && seats.value[1] !== "human")
const asSquares = (cells: number[]) => cells.map(squareOf)
const { board, trail, turn, banner } = useCheckersBoardView(session)
const mustCapture = computed(() => humanTurn.value && moves.value.some((m) => Math.abs(squareOf(m[1]!)[0] - squareOf(m[0]!)[0]) === 2))

// Live stats: material counts men 1 and kings 2 (the same scale the material players use).
const stats = computed(() => {
  const piece = [0, 0]
  const material = [0, 0]
  const kings = [0, 0]
  for (const c of board.value.cells) {
    const side = c.label.startsWith("red") ? 0 : 1
    const king = c.label.endsWith("king")
    piece[side]!++
    material[side]! += king ? 2 : 1
    if (king) kings[side]!++
  }
  const captures = history.value.reduce((n, ply) => n + (Math.abs(squareOf(ply.move[1]!)[0] - squareOf(ply.move[0]!)[0]) === 2 ? ply.move.length - 1 : 0), 0)
  return { piece, material, kings, captures }
})
const lead = computed(() => stats.value.material[0]! - stats.value.material[1]!)

// Play mode: report the human's points per game (win 1, draw ½) so the side leaderboard can slot "You" in.
// It's *this browser's session* record against whoever was on the other side, not a leaderboard entry.
watch(
  record,
  (r) => {
    const humanSeat = seats.value.indexOf("human")
    const games = r.wins[0] + r.wins[1] + r.draws
    if (props.mode !== "play" || humanSeat < 0 || games === 0) return
    emit("score", (r.wins[humanSeat]! + 0.5 * r.draws) / games, false, `You · ${games} game${games === 1 ? "" : "s"}`)
  },
  { deep: true },
)
</script>

<template>
  <VersusStage
    :session="session"
    :player-colors="['#ef3b5d', '#e9ebf1']"
    :scores="stats.piece"
    :arena-defaults="watching ? [subject, rival] : [subject, rival]"
    :show-players="false"
  >
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
    <template #hint>
      <p v-if="mustCapture" class="mt-2 text-xs text-life-400">A capture is available, so you must take it (mandatory captures).</p>
      <p v-else-if="humanTurn && !path.length" class="mt-2 text-xs text-fg-subtle">
        Pieces you can move are ringed in green. Multi-jumps: keep clicking each landing square.
      </p>
      <p v-if="failed.length" class="mt-2 text-xs text-queen-300">Couldn't load: {{ failed.join("; ") }}</p>
    </template>

    <template #stats>
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Move" :value="history.length" hint="plies played this game" />
        <StatTile label="Material lead" :value="lead === 0 ? 'even' : `${lead > 0 ? players[0] : players[1]} +${Math.abs(lead)}`" :tone="lead === 0 ? 'default' : 'queen'" hint="men 1, kings 2" />
        <StatTile label="Kings" :value="`${stats.kings[0]} · ${stats.kings[1]}`" :hint="`${players[0]} · ${players[1]}`" />
        <StatTile label="Captures" :value="stats.captures" hint="pieces taken so far" />
      </div>
    </template>
    <template #arena-note>
      Any two entrants on the leaderboard -- the same measurement <code class="chip">jobs/evaluate_versus.py</code> ranks them by.
    </template>
  </VersusStage>
</template>

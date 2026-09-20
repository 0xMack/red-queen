<script setup lang="ts">
import { usePieceMotion } from "~/composables/usePieceMotion"

// The Checkers board (docs/design/0006). Display-only by default (the Games card, the Learn chapter); given
// `starts` / `targets` it becomes the play surface -- clickable pieces that can move, the squares they can land
// on. Pieces glide (usePieceMotion diffs each position by piece id and replays the move, hop by hop, when
// the engine supplies ids -- utils/checkersEngine.ts does), captured ones shrink away with a burst, a crowned
// one flashes, and the side to move glows around the board. `flipped` turns it 180° so a human's own pieces
// sit at the bottom whichever colour they are; coordinates follow the flip.
//
// Deliberately not a `GridBoard` variant: pieces are captured and crowned rather than glide-and-grow, and a
// human builds a multi-jump one landing at a time.
type Square = [number, number]
interface BoardPiece {
  id?: number | string
  x: number
  y: number
  label: string
}
const props = defineProps<{
  state: { width: number; height: number; cells: BoardPiece[] }
  flipped?: boolean
  selected?: Square | null
  starts?: Square[]
  targets?: Square[]
  /** The last move's path (from, ...landings): drawn as a fading trail and drives the piece motion. */
  trail?: Square[]
  /** Whose turn it is (0 = Red, 1 = Black): the board glows in their colour. null when nobody's. */
  turn?: 0 | 1 | null
  /** A result to lay over the board when the game ends. */
  banner?: { title: string; sub?: string; tone?: "red" | "black" | "draw" } | null
}>()
const emit = defineEmits<{ square: [x: number, y: number] }>()

const S = 40 // a square
const M = 18 // margin for the coordinates
const size = computed(() => props.state.width * S + 2 * M)

// Logical <-> screen: a flip turns the board 180 degrees, which is its own inverse.
const col = (x: number) => (props.flipped ? props.state.width - 1 - x : x)
const row = (y: number) => (props.flipped ? props.state.height - 1 - y : y)
/** Logical square -> pixel centre. */
const cx = (x: number) => M + col(x) * S + S / 2
const cy = (y: number) => M + row(y) * S + S / 2

const clickable = (screenCol: number, screenRow: number) => {
  const [x, y] = [col(screenCol), row(screenRow)]
  return at(props.starts, x, y) || at(props.targets, x, y)
}
const at = (list: Square[] | undefined, x: number, y: number) => !!list?.some(([a, b]) => a === x && b === y)
const sameSquare = (a: Square | null | undefined, x: number, y: number) => !!a && a[0] === x && a[1] === y
const interactive = computed(() => !!(props.starts?.length || props.targets?.length))

const { shown, bursts, busy } = usePieceMotion(
  () => props.state.cells.map((c) => ({ id: c.id ?? `${c.x},${c.y}`, x: c.x, y: c.y, label: c.label })),
  () => props.trail ?? [],
)

// A target that captures (two files from the piece about to move) is marked differently from a plain step.
const isCapture = (x: number, y: number) => !!props.selected && Math.abs(x - props.selected[0]) === 2
const trailPoints = computed(() => (props.trail ?? []).map(([x, y]) => `${cx(x)},${cy(y)}`).join(" "))

const files = computed(() => Array.from({ length: props.state.width }, (_, i) => "abcdefgh"[props.flipped ? props.state.width - 1 - i : i]))
const ranks = computed(() => Array.from({ length: props.state.height }, (_, i) => (props.flipped ? props.state.height - i : i + 1)))

// The result waits for the last move to finish playing: a banner over a board that is still moving reads as
// a game that "keeps going" after it ended.
const showBanner = computed(() => (busy.value ? null : props.banner))

const TURN = { 0: "#ef3b5d", 1: "#e9ebf1" } as const
const glow = computed(() => (props.turn == null ? "transparent" : TURN[props.turn]))
</script>

<template>
  <div class="board relative" :style="{ '--glow': glow }">
    <svg :viewBox="`0 0 ${size} ${size}`" class="block h-auto w-full select-none">
      <defs>
        <radialGradient id="piece-red" cx="38%" cy="32%" r="75%">
          <stop offset="0%" stop-color="#ff7b93" />
          <stop offset="55%" stop-color="#ef3b5d" />
          <stop offset="100%" stop-color="#b0163c" />
        </radialGradient>
        <radialGradient id="piece-black" cx="38%" cy="32%" r="75%">
          <stop offset="0%" stop-color="#ffffff" />
          <stop offset="60%" stop-color="#dfe3ee" />
          <stop offset="100%" stop-color="#a9b0c4" />
        </radialGradient>
        <linearGradient id="sq-dark" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#151926" />
          <stop offset="100%" stop-color="#0f121c" />
        </linearGradient>
        <linearGradient id="sq-light" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#252c3f" />
          <stop offset="100%" stop-color="#1e2435" />
        </linearGradient>
      </defs>

      <!-- Frame + coordinates -->
      <rect x="0" y="0" :width="size" :height="size" rx="12" fill="#0b0e17" />
      <g class="coords" font-size="8.5">
        <text v-for="(f, i) in files" :key="`f${i}`" :x="M + i * S + S / 2" :y="size - 5" text-anchor="middle">{{ f }}</text>
        <text v-for="(r, i) in ranks" :key="`r${i}`" :x="7" :y="M + i * S + S / 2 + 3" text-anchor="middle">{{ r }}</text>
      </g>

      <!-- Squares -->
      <g>
        <template v-for="y in state.height" :key="y">
          <rect
            v-for="x in state.width"
            :key="x"
            :x="M + (x - 1) * S"
            :y="M + (y - 1) * S"
            :width="S"
            :height="S"
            :fill="(x + y) % 2 === 0 ? 'url(#sq-light)' : 'url(#sq-dark)'"
          />
        </template>
      </g>

      <!-- The last move: both squares tinted, and its path -->
      <g v-if="trail?.length">
        <rect
          v-for="([x, y], i) in [trail[0]!, trail.at(-1)!]"
          :key="i"
          :x="cx(x) - S / 2"
          :y="cy(y) - S / 2"
          :width="S"
          :height="S"
          fill="#f5b84a"
          :opacity="i === 0 ? 0.1 : 0.2"
        />
        <polyline :points="trailPoints" fill="none" stroke="#f5b84a" stroke-opacity="0.45" stroke-width="2" stroke-dasharray="3 4" stroke-linecap="round" stroke-linejoin="round" />
      </g>

      <!-- Where the picked piece can go -->
      <g>
        <template v-for="([x, y], i) in targets" :key="`t${i}`">
          <circle v-if="isCapture(x, y)" class="target capture" :cx="cx(x)" :cy="cy(y)" r="11" fill="none" stroke="#f5b84a" stroke-width="2.2" />
          <circle v-else class="target" :cx="cx(x)" :cy="cy(y)" r="6.5" fill="#4ade80" />
        </template>
      </g>

      <!-- Capture bursts (under the pieces, over the squares) -->
      <circle v-for="b in bursts" :key="b.key" class="burst" :cx="cx(b.x)" :cy="cy(b.y)" r="8" fill="none" stroke="#ffd54a" stroke-width="2.5" />

      <!-- Pieces: keyed by id, so the same piece glides from square to square -->
      <CheckersPiece
        v-for="(p, i) in shown"
        :key="p.key"
        :x="cx(p.x)"
        :y="cy(p.y)"
        :label="p.label"
        :index="i"
        :selected="sameSquare(selected, p.x, p.y)"
        :movable="!selected && at(starts, p.x, p.y)"
        :hopping="p.hopping"
        :leaving="p.leaving"
        :crowned="p.crowned"
      />

      <!-- Click targets, over everything, only when the board is a play surface. -->
      <template v-if="interactive">
        <template v-for="y in state.height" :key="`c${y}`">
          <rect
            v-for="x in state.width"
            :key="`c${x}`"
            :x="M + (x - 1) * S"
            :y="M + (y - 1) * S"
            :width="S"
            :height="S"
            fill="transparent"
            :class="clickable(x - 1, y - 1) ? 'cursor-pointer' : ''"
            @click="emit('square', col(x - 1), row(y - 1))"
          />
        </template>
      </template>
    </svg>

    <!-- The result, once the last move has finished playing -->
    <BoardResult v-if="showBanner" :title="showBanner.title" :sub="showBanner.sub" :tone="showBanner.tone" />
  </div>
</template>

<style scoped>
.board {
  border-radius: 12px;
  box-shadow:
    0 0 0 1px rgb(255 255 255 / 0.06),
    0 0 0 1px var(--glow),
    0 0 26px -8px var(--glow);
  transition: box-shadow 400ms ease;
}
.coords text {
  fill: #5d667c;
  font-family: ui-monospace, monospace;
}
.target {
  animation: target 1.4s ease-in-out infinite;
  transform-box: fill-box;
  transform-origin: center;
  cursor: pointer;
}
.target.capture {
  animation-duration: 1s;
}
.burst {
  transform-box: fill-box;
  transform-origin: center;
  animation: burst 560ms ease-out both;
}
@keyframes target {
  0%,
  100% {
    opacity: 0.55;
    transform: scale(0.9);
  }
  50% {
    opacity: 1;
    transform: scale(1.12);
  }
}
@keyframes burst {
  from {
    opacity: 0.95;
    transform: scale(0.5);
  }
  to {
    opacity: 0;
    transform: scale(3);
  }
}
@media (prefers-reduced-motion: reduce) {
  .board,
  .target,
  .burst {
    transition: none;
    animation: none;
  }
}
</style>

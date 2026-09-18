<script setup lang="ts">
import type { GridCell, RenderState } from "~/types/games"

// tickMs: how often the state changes, so the glide between cells lasts about one tick at any speed.
const props = withDefaults(defineProps<{ state: RenderState; tickMs?: number }>(), { tickMs: 110 })
const glide = computed(() => `transform ${Math.min(props.tickMs, 300) * 0.9}ms linear`)

const cellSize = 28

const bodyCells = computed(() => props.state.cells.filter((c) => c.label !== "food"))
const foodCell = computed(() => props.state.cells.find((c) => c.label === "food"))

// render_state()'s cells are always ordered [segment nearest the head, ..., tail, head] (see
// games/snake.py's _cell_labels()) -- the segment right behind the head is always bodyCells[0]
// (or the head itself, on a fresh 1-cell-body edge case that never actually occurs for Snake).
// Used only to orient the head's eyes; purely cosmetic, never affects gameplay.
const heading = computed((): { dx: number; dy: number } => {
  const cells = bodyCells.value
  const head = cells.at(-1)
  // cells[0], not cells.at(-2): with the head stored *last*, at(-2) is the tail, which made the
  // eyes point away from the tail instead of along the direction of travel.
  const behindHead = cells.length > 1 ? cells[0] : head
  if (!head || !behindHead) return { dx: 1, dy: 0 }
  const dx = Math.sign(head.x - behindHead.x)
  const dy = Math.sign(head.y - behindHead.y)
  return dx === 0 && dy === 0 ? { dx: 1, dy: 0 } : { dx, dy }
})

// Eye centers as fractions of a cell, offset toward whichever edge the snake is heading, and
// spread perpendicular to travel so they read as a pair of eyes rather than one dot.
const eyeOffsets = computed(() => {
  const { dx, dy } = heading.value
  const forward = 0.5 + 0.22 * dx
  const forwardY = 0.5 + 0.22 * dy
  const spread = dx !== 0 ? { x: 0, y: 0.2 } : { x: 0.2, y: 0 }
  return [
    { x: forward + spread.x, y: forwardY + spread.y },
    { x: forward - spread.x, y: forwardY - spread.y },
  ]
})

// Body fades toward the tail (index 0 is the segment nearest the head) -- reads direction at a
// glance even in a still screenshot.
function opacityFor(index: number): number {
  const n = bodyCells.value.length
  return n <= 2 ? 1 : 1 - 0.55 * (index / (n - 1))
}

function fillFor(cell: GridCell): string {
  return cell.label === "head" ? "#86efac" : "#22c55e"
}

// Unique per instance -- several boards can be on one page (e.g. game cards).
const uid = useId()
const checkerId = `board-checker-${uid}`
const glowId = `food-glow-${uid}`
</script>

<template>
  <svg
    :viewBox="`0 0 ${props.state.width * cellSize} ${props.state.height * cellSize}`"
    class="block h-auto w-full rounded-xl border border-line bg-sunken shadow-[0_20px_60px_-20px_rgb(0_0_0/0.8)]"
  >
    <defs>
      <pattern :id="checkerId" :width="cellSize * 2" :height="cellSize * 2" patternUnits="userSpaceOnUse">
        <rect :width="cellSize * 2" :height="cellSize * 2" fill="#0c1310" />
        <rect :width="cellSize" :height="cellSize" fill="#101a15" />
        <rect :x="cellSize" :y="cellSize" :width="cellSize" :height="cellSize" fill="#101a15" />
      </pattern>
      <filter :id="glowId" x="-100%" y="-100%" width="300%" height="300%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>

    <rect :width="props.state.width * cellSize" :height="props.state.height * cellSize" :fill="`url(#${checkerId})`" />

    <g
      v-for="(cell, index) in bodyCells"
      :key="index"
      :style="{ transform: `translate(${cell.x * cellSize}px, ${cell.y * cellSize}px)`, transition: glide }"
    >
      <rect
        :x="1"
        :y="1"
        :width="cellSize - 2"
        :height="cellSize - 2"
        :fill="fillFor(cell)"
        :fill-opacity="cell.label === 'head' ? 1 : opacityFor(index)"
        :stroke="cell.label === 'head' ? '#dcfce7' : 'none'"
        :stroke-width="cell.label === 'head' ? 1 : 0"
        rx="7"
      />
      <template v-if="cell.label === 'head'">
        <circle
          v-for="(eye, i) in eyeOffsets"
          :key="i"
          :cx="eye.x * cellSize"
          :cy="eye.y * cellSize"
          r="2.6"
          fill="white"
        />
        <circle
          v-for="(eye, i) in eyeOffsets"
          :key="`pupil-${i}`"
          :cx="eye.x * cellSize + heading.dx * 0.8"
          :cy="eye.y * cellSize + heading.dy * 0.8"
          r="1.2"
          fill="#052e16"
        />
      </template>
    </g>

    <circle
      v-if="foodCell"
      :cx="foodCell.x * cellSize + cellSize / 2"
      :cy="foodCell.y * cellSize + cellSize / 2"
      :r="cellSize * 0.32"
      fill="#ff5c7a"
      :filter="`url(#${glowId})`"
      class="food-pulse"
    />

    <text
      v-if="!props.state.alive"
      :x="(props.state.width * cellSize) / 2"
      :y="(props.state.height * cellSize) / 2"
      text-anchor="middle"
      dominant-baseline="middle"
      class="fill-queen-300 font-display text-2xl font-bold"
    >
      game over
    </text>
  </svg>
</template>

<style scoped>
.food-pulse {
  transform-box: fill-box;
  transform-origin: center;
  animation: food-pulse 1s ease-in-out infinite;
}

@keyframes food-pulse {
  0%,
  100% {
    transform: scale(0.85);
  }
  50% {
    transform: scale(1.1);
  }
}
</style>

<script setup lang="ts">
// One Checkers piece, in board units (0,0 = the centre of the a1 square's cell in the *logical* board; the
// board handles flipping and passes the pixel position). It owns how it looks (a domed, lit disc; a crown for
// a king) and how it moves: position is a CSS transform, so when the position changes the browser glides
// it -- the hop path is the motion composable's job, this just eases each step. `hopping` lifts it,
// `leaving` shrinks it away, `crowned` flashes, and a piece mounting (a new game) pops in. All of it is off
// under prefers-reduced-motion.
const props = defineProps<{
  x: number
  y: number
  label: string
  index?: number
  selected?: boolean
  movable?: boolean
  hopping?: boolean
  leaving?: boolean
  crowned?: boolean
}>()

const red = computed(() => props.label.startsWith("red"))
const king = computed(() => props.label.endsWith("king"))
</script>

<template>
  <g
    class="piece"
    :class="{ leaving, movable, 'is-king': king }"
    :style="{ transform: `translate(${x}px, ${y}px)`, '--i': index ?? 0 }"
  >
    <g class="body" :class="{ hopping, selected, crowned }">
      <ellipse class="shadow" cx="0" cy="13" rx="13" ry="4" />
      <circle r="15.5" :fill="red ? 'url(#piece-red)' : 'url(#piece-black)'" :stroke="red ? '#7a1230' : '#8b93a8'" stroke-width="1.2" />
      <circle r="10.5" fill="none" :stroke="red ? '#b81f47' : '#9aa3b8'" stroke-width="1.2" opacity="0.7" />
      <ellipse class="gloss" cx="-4" cy="-6" rx="7" ry="3.4" />
      <g v-if="king" class="crown">
        <path d="M-7.5 4 L-8.5 -4 L-3.8 0 L0 -6.5 L3.8 0 L8.5 -4 L7.5 4 Z" :fill="red ? '#fff3c4' : '#c81e45'" :stroke="red ? '#e8b923' : '#7a1230'" stroke-width="0.8" stroke-linejoin="round" />
        <circle cx="0" cy="-7.2" r="1.3" fill="#ffd54a" />
      </g>
      <circle v-if="crowned" class="crown-flash" r="15" fill="none" stroke="#ffd54a" stroke-width="2" />
      <circle v-if="movable && !selected" class="beacon" r="18.5" fill="none" stroke="#4ade80" stroke-width="1.6" />
      <circle v-if="selected" r="19" fill="none" stroke="#f5b84a" stroke-width="2" />
    </g>
  </g>
</template>

<style scoped>
.piece {
  transition:
    transform 230ms cubic-bezier(0.3, 0.7, 0.25, 1),
    opacity 320ms ease;
  /* A new piece (a new game) pops in, in a ripple from the first piece to the last. `backwards`, not `both`: it
     holds the invisible start frame during the stagger delay, but once finished it must let go -- `both` keeps the
     final keyframe's opacity: 1 applied forever, which beats `.leaving { opacity: 0 }` and stops a captured piece fading. */
  animation: piece-in 420ms cubic-bezier(0.2, 0.9, 0.3, 1.2) backwards;
  animation-delay: calc(var(--i) * 18ms);
}
.piece.leaving {
  opacity: 0;
  pointer-events: none;
}
.body {
  transform-box: fill-box;
  transform-origin: center;
  transition: transform 230ms cubic-bezier(0.3, 0.7, 0.25, 1);
}
.piece.leaving .body {
  transform: scale(0.4) rotate(40deg);
  transition: transform 320ms ease-in;
}
.body.hopping {
  transform: translateY(-4px) scale(1.12);
}
.body.selected {
  transform: translateY(-3px) scale(1.1);
}
.shadow {
  fill: rgb(0 0 0 / 0.45);
  transition: transform 230ms ease;
}
.body.hopping .shadow,
.body.selected .shadow {
  transform: translateY(3px) scale(0.85);
}
.gloss {
  fill: rgb(255 255 255 / 0.22);
}
.beacon {
  animation: beacon 1.6s ease-in-out infinite;
}
.crown-flash {
  transform-box: fill-box;
  transform-origin: center;
  animation: crown-flash 900ms ease-out both;
}
.is-king .crown {
  filter: drop-shadow(0 0 2.5px rgb(255 213 74 / 0.7));
}

@keyframes piece-in {
  from {
    opacity: 0;
    scale: 0.4;
  }
  to {
    opacity: 1;
    scale: 1;
  }
}
@keyframes beacon {
  0%,
  100% {
    opacity: 0.25;
    transform: scale(0.96);
  }
  50% {
    opacity: 0.85;
    transform: scale(1.04);
  }
}
@keyframes crown-flash {
  from {
    opacity: 1;
    transform: scale(0.8);
  }
  to {
    opacity: 0;
    transform: scale(2.4);
  }
}

@media (prefers-reduced-motion: reduce) {
  .piece,
  .body,
  .shadow,
  .beacon,
  .crown-flash {
    transition: none;
    animation: none;
  }
}
</style>

<script setup lang="ts">
import type { BanditPull } from "~/composables/useBanditRun"

// Every pull of a game in order (docs/design/0011): one mark per pull, coloured by machine, tall for a win (or a
// payout above the machine's scenario average), short for a loss. Exploring reads as a patchwork of colours, exploiting
// as a long run of one -- and a strategy that settled too early as a run of the wrong one. In a lamp scenario, a dot
// under each mark says which lamp was lit.
const props = withDefaults(defineProps<{ pulls: BanditPull[]; budget: number; binary: boolean; lamps?: boolean; drift?: number | null }>(), {
  lamps: false,
  drift: null,
})
const W = 600
const step = computed(() => W / Math.max(1, props.budget))
const high = (r: number) => (props.binary ? r >= 1 : r > (props.pulls.reduce((s, p) => s + p.reward, 0) / Math.max(1, props.pulls.length)))
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${lamps ? 44 : 36}`" class="block h-auto w-full" preserveAspectRatio="none" role="img" aria-label="Every pull in order, coloured by machine">
    <rect :width="W" height="30" :fill="palette.sunken" rx="3" />
    <line v-if="drift" :x1="drift * step" :x2="drift * step" y1="0" y2="30" :stroke="palette.queen400" stroke-dasharray="2 2" />
    <rect
      v-for="(p, i) in pulls"
      :key="i"
      :x="i * step + 0.15 * step"
      :y="high(p.reward) ? 3 : 19"
      :width="Math.max(1, step * 0.7)"
      :height="high(p.reward) ? 24 : 8"
      :fill="armColor(p.arm)"
      :opacity="high(p.reward) ? 1 : 0.55"
      rx="0.8"
    />
    <template v-if="lamps">
      <circle v-for="(p, i) in pulls" :key="`l${i}`" :cx="i * step + step / 2" cy="39" r="1.6" :fill="p.lamp === 0 ? palette.queen400 : palette.signal400" />
    </template>
  </svg>
</template>

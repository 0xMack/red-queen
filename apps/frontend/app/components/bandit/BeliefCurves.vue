<script setup lang="ts">
// Thompson sampling's beliefs, drawn (docs/design/0011): for a win/lose machine, a Beta(1 + wins, 1 + losses) curve
// of where its win rate could be -- wide when it has barely been tried, a narrow peak once it has. Each pull draws one
// plausible value from every curve and pulls the highest, so a wide curve still gets tried sometimes: that's all the
// exploring it does. `truth` (after the game) marks where each machine really was.
const props = defineProps<{ wins: number[]; pulls: number[]; truth?: number[] | null }>()

const W = 480
const H = 150
const N = 120
// log Beta(a, b) for positive integers: log((a-1)! (b-1)! / (a+b-1)!)
const logFactorial = (n: number) => {
  let s = 0
  for (let i = 2; i <= n; i++) s += Math.log(i)
  return s
}
const curves = computed(() => {
  const raw = props.wins.map((w, arm) => {
    const a = 1 + w
    const b = 1 + props.pulls[arm]! - w
    const logB = logFactorial(a - 1) + logFactorial(b - 1) - logFactorial(a + b - 1)
    return Array.from({ length: N + 1 }, (_, i) => {
      const x = Math.min(1 - 1e-9, Math.max(1e-9, i / N))
      return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB)
    })
  })
  const top = Math.max(1, ...raw.flat())
  return raw.map((ys, arm) => ({
    arm,
    // rounded: SVG attributes must match between server and client (Math.exp/log can differ in the last digit)
    d: ys.map((y, i) => `${i === 0 ? "M" : "L"}${((i / N) * W).toFixed(1)},${(H - 4 - (y / top) * (H - 12)).toFixed(1)}`).join(""),
  }))
})
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H + 16}`" class="block h-auto w-full" role="img" aria-label="Each machine's belief about its win rate">
    <line :x1="0" :x2="W" :y1="H - 4" :y2="H - 4" :stroke="palette.lineStrong" />
    <text v-for="t in [0, 0.25, 0.5, 0.75, 1]" :key="t" :x="t * W" :y="H + 10" :text-anchor="t === 0 ? 'start' : t === 1 ? 'end' : 'middle'" class="num fill-fg-subtle text-[9px]">
      {{ t === 0 ? "never wins" : t === 1 ? "always wins" : `${t * 100}%` }}
    </text>
    <path v-for="c in curves" :key="c.arm" :d="c.d" fill="none" :stroke="armColor(c.arm)" stroke-width="1.8" stroke-linejoin="round" />
    <template v-if="truth">
      <line v-for="(t, arm) in truth" :key="`t${arm}`" :x1="t * W" :x2="t * W" :y1="4" :y2="H - 4" :stroke="armColor(arm)" stroke-dasharray="2 3" stroke-width="1" />
    </template>
  </svg>
</template>

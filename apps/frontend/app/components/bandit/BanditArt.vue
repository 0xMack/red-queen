<script setup lang="ts">
// A still of the bandit floor (docs/design/0011) for the game card and the chapter cover: five cabinets, one paying
// out, each with the belief bar a strategy keeps -- and the one row of values that is the whole table. Drawn at a
// fixed 400x250 viewBox, like ChapterArt.
withDefaults(defineProps<{ table?: boolean }>(), { table: true })
const MACHINES = [
  { reel: "–", estimate: 0.3, pulls: 4 },
  { reel: "WIN", estimate: 0.72, pulls: 31, win: true },
  { reel: "–", estimate: 0.45, pulls: 9 },
  { reel: "–", estimate: 0.2, pulls: 3 },
  { reel: "–", estimate: 0.55, pulls: 12 },
]
const X = (i: number) => 30 + i * 70
</script>

<template>
  <svg viewBox="0 0 400 250" class="block size-full" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    <rect width="400" height="250" :fill="palette.sunken" />
    <defs>
      <radialGradient id="bandit-art-glow" cx="0.5" cy="0.4" r="0.6">
        <stop offset="0" :stop-color="palette.queen500" stop-opacity="0.16" />
        <stop offset="1" :stop-color="palette.queen500" stop-opacity="0" />
      </radialGradient>
    </defs>
    <rect width="400" height="250" fill="url(#bandit-art-glow)" />
    <g v-for="(m, i) in MACHINES" :key="i">
      <rect :x="X(i)" y="40" width="60" height="118" rx="7" :fill="palette.surface" :stroke="m.win ? armColor(i) : palette.line" :stroke-width="m.win ? 1.5 : 1" />
      <text :x="X(i) + 10" y="62" class="font-display" font-size="18" :fill="palette.fg">{{ armName(i) }}</text>
      <circle :cx="X(i) + 50" :cy="56" r="3.5" :fill="armColor(i)" :opacity="m.win ? 1 : 0.35" />
      <rect :x="X(i) + 7" y="70" width="46" height="30" rx="4" :fill="palette.sunken" :stroke="palette.line" />
      <text :x="X(i) + 30" y="90" text-anchor="middle" font-size="11" font-family="Geist Mono, monospace" :fill="m.win ? armColor(i) : palette.fgSubtle">{{ m.reel }}</text>
      <line :x1="X(i) + 64" :x2="X(i) + 64" y1="104" y2="80" :stroke="palette.lineStrong" stroke-width="2" stroke-linecap="round" />
      <circle :cx="X(i) + 64" cy="77" r="4" :fill="armColor(i)" />
      <text :x="X(i) + 8" y="118" font-size="8" font-family="Geist Mono, monospace" :fill="palette.fgSubtle">{{ m.pulls }}×</text>
      <rect :x="X(i) + 7" y="128" width="46" height="4" rx="2" :fill="palette.sunken" />
      <rect :x="X(i) + 7" y="128" :width="46 * m.estimate" height="4" rx="2" :fill="armColor(i)" />
      <rect :x="X(i) + 7 + 46 * m.estimate - 1" y="124" width="1.5" height="12" :fill="palette.gold300" :opacity="i === 1 ? 1 : 0" />
    </g>
    <g v-if="table">
      <text x="30" y="190" font-size="8" font-family="Geist Mono, monospace" letter-spacing="1" :fill="palette.fgSubtle">THE TABLE · ONE ROW</text>
      <g v-for="(m, i) in MACHINES" :key="`t${i}`">
        <rect :x="X(i)" y="198" width="60" height="26" rx="4" :fill="palette.surface" :stroke="i === 1 ? palette.fgMuted : palette.line" />
        <text :x="X(i) + 30" y="215" text-anchor="middle" font-size="11" font-family="Geist Mono, monospace" :fill="palette.fg">{{ m.estimate.toFixed(2) }}</text>
      </g>
    </g>
  </svg>
</template>

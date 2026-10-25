<script setup lang="ts">
// A still of Copy Cat (docs/design/0022) for the game card: on the left the image a generator delivers -- coarse pixels,
// off-colour, on a background -- and on the right the design behind it, as layers pulled apart: a circle whole behind
// the square that hides part of it. Drawn at a fixed 400x250 viewBox, like BanditArt.
const CELL = 10
// a circle partly behind a square, sampled on a coarse grid with a little colour wobble (deterministic)
const pixels = computed(() => {
  const out: { x: number; y: number; fill: string; opacity: number }[] = []
  for (let gy = 0; gy < 15; gy++) {
    for (let gx = 0; gx < 15; gx++) {
      const x = gx + 0.5
      const y = gy + 0.5
      const wobble = 0.82 + 0.18 * Math.abs(Math.sin(gx * 12.9898 + gy * 78.233))
      const inSquare = x > 6 && x < 13 && y > 6 && y < 13
      const inCircle = (x - 6) ** 2 + (y - 6) ** 2 < 25
      if (inSquare) out.push({ x: gx, y: gy, fill: palette.signal400, opacity: wobble })
      else if (inCircle) out.push({ x: gx, y: gy, fill: palette.queen400, opacity: wobble })
      else out.push({ x: gx, y: gy, fill: palette.fgSubtle, opacity: 0.12 + 0.08 * wobble })
    }
  }
  return out
})
</script>

<template>
  <svg viewBox="0 0 400 250" class="block size-full" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    <rect width="400" height="250" :fill="palette.sunken" />
    <g transform="translate(36 50)">
      <rect v-for="(p, i) in pixels" :key="i" :x="p.x * CELL" :y="p.y * CELL" :width="CELL - 0.6" :height="CELL - 0.6" :fill="p.fill" :opacity="p.opacity" />
    </g>
    <path d="M198 125 h22 m-7 -7 l7 7 l-7 7" :stroke="palette.fgMuted" stroke-width="1.6" fill="none" stroke-linecap="round" />
    <g transform="translate(300 128) skewX(-14) scale(1 0.62)" :stroke="palette.lineStrong" stroke-width="1">
      <rect x="-62" y="-40" width="124" height="124" rx="4" fill="none" stroke-dasharray="3 4" transform="translate(-18 -18)" />
      <circle cx="-18" cy="-18" r="50" :fill="palette.queen400" fill-opacity="0.92" transform="translate(-14 -14)" />
      <rect x="2" y="2" width="70" height="70" :fill="palette.signal400" fill-opacity="0.95" transform="translate(16 16)" />
    </g>
    <text x="36" y="226" class="font-mono" font-size="11" :fill="palette.fgSubtle">the image</text>
    <text x="372" y="226" text-anchor="end" class="font-mono" font-size="11" :fill="palette.fgSubtle">the design, in layers</text>
  </svg>
</template>

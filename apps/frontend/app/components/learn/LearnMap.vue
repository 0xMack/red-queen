<script setup lang="ts">
import { chapterBySlug, chapterHref, learnChapters, learnPaths, pathsThrough, type LearnChapter, type LearnPathId } from "~/data/learnChapters"
import { palette } from "~/utils/palette"

// The Learn section as a transit map: every path is a coloured line, every chapter a station, and a chapter on two
// paths an interchange. A dashed spur is what a path assumes from another. Hovering a station shows the chapter;
// `highlight` (a path id, from the cards below) dims the other lines.
const props = defineProps<{ highlight?: LearnPathId | null }>()
const emit = defineEmits<{ hoverPath: [LearnPathId | null] }>()

// Hand-placed, (column, row): the map is drawn for legibility, like a real one -- not generated. Every chapter on a
// path needs a place here (a missing one warns in development and isn't drawn).
const PLACES: Record<string, [number, number]> = {
  "genetic-algorithms": [0, 0],
  "selection-strategies": [1, 0],
  "genome-representations": [2, 0],
  neuroevolution: [3, 0],
  neat: [4, 0],
  "teaching-a-snake": [5, 0],
  "multi-armed-bandits": [1, 1],
  "q-learning": [2, 1],
  autodiff: [3, 2],
  dqn: [4, 1],
  "policy-gradients": [5, 1],
  "multi-agent-games": [6, 1],
  "self-play": [7, 1],
  transformers: [5, 2],
  "real-time-architecture": [0, 2],
}
// Stations whose label sits above rather than below (the top row, clear of the lines under it).
const LABEL_ABOVE = new Set(["genetic-algorithms", "selection-strategies", "genome-representations", "neuroevolution", "neat", "teaching-a-snake"])

const COL = 150
const ROW = 104
const PAD_X = 80
const PAD_Y = 52
const width = PAD_X * 2 + COL * 7
const height = PAD_Y * 2 + ROW * 2

function at(slug: string): { x: number; y: number } | null {
  const place = PLACES[slug]
  return place ? { x: PAD_X + place[0] * COL, y: PAD_Y + place[1] * ROW } : null
}

if (import.meta.dev) {
  for (const p of learnPaths) for (const slug of p.chapters) if (!PLACES[slug]) console.warn(`LearnMap: no place for "${slug}"`)
}

// Transit-map segments: a 45° diagonal for the vertical change and straight along for the rest. Going down, the
// straight part comes first, so the line leaves a station level and clears the label hanging under it.
function segment(a: { x: number; y: number }, b: { x: number; y: number }): string {
  const dy = b.y - a.y
  const dx = b.x - a.x
  const diag = Math.min(Math.abs(dy), Math.abs(dx))
  const mid = dy > 0 ? { x: b.x - Math.sign(dx) * diag, y: a.y } : { x: a.x + Math.sign(dx) * diag, y: a.y + Math.sign(dy) * diag }
  return `M${a.x},${a.y} L${mid.x},${mid.y} L${b.x},${b.y}`
}

const lines = computed(() =>
  learnPaths.map((path) => {
    const points = path.chapters.map(at).filter((p) => !!p)
    const d = points.length === 1 ? `M${points[0]!.x - 34},${points[0]!.y} L${points[0]!.x + 34},${points[0]!.y}` : points.slice(1).map((p, i) => segment(points[i]!, p)).join(" ")
    const spurs = (path.assumes ?? []).map((slug) => {
      const from = at(slug)
      const to = at(path.chapters[0]!)
      return from && to ? segment(from, to) : ""
    })
    return { path, d, spurs }
  }),
)

const stations = computed(() =>
  learnChapters
    .filter((c) => at(c.slug))
    .map((c) => ({ chapter: c, ...at(c.slug)!, paths: pathsThrough(c), above: LABEL_ABOVE.has(c.slug) })),
)

const { isRead } = useReadChapters()
const hovered = ref<LearnChapter | null>(null)
const hoveredPaths = computed(() => (hovered.value ? pathsThrough(hovered.value) : []))
const dim = (id: LearnPathId) => !!props.highlight && props.highlight !== id

function onStation(c: LearnChapter | null) {
  hovered.value = c
  emit("hoverPath", c ? c.home : null)
}

// Tooltip placement as a fraction of the map, so it tracks the SVG as it scales.
const tip = computed(() => {
  const c = hovered.value
  const p = c && at(c.slug)
  if (!p) return null
  return { left: `${(p.x / width) * 100}%`, top: `${(p.y / height) * 100}%`, flip: p.x > width * 0.6 }
})

// Wrap a short label onto two lines at the space nearest its middle.
function labelLines(text: string): string[] {
  if (text.length <= 12 || !text.includes(" ")) return [text]
  const mid = text.length / 2
  let best = -1
  for (let i = 0; i < text.length; i++) if (text[i] === " " && (best < 0 || Math.abs(i - mid) < Math.abs(best - mid))) best = i
  return [text.slice(0, best), text.slice(best + 1)]
}
</script>

<template>
  <div class="relative">
    <div class="overflow-x-auto">
      <svg :viewBox="`0 0 ${width} ${height}`" class="block w-full min-w-[760px]" role="img" aria-label="Map of the learning paths">
        <g v-for="{ path, d, spurs } in lines" :key="path.id" class="transition-opacity duration-200" :opacity="dim(path.id) ? 0.18 : 1">
          <path v-for="(s, i) in spurs" :key="i" :d="s" fill="none" :stroke="palette[path.color]" stroke-width="3" stroke-dasharray="2 7" stroke-linecap="round" opacity="0.7" />
          <path :d="d" fill="none" :stroke="palette[path.color]" stroke-width="7" stroke-linecap="round" stroke-linejoin="round" />
        </g>

        <g
          v-for="s in stations"
          :key="s.chapter.slug"
          class="cursor-pointer"
          :opacity="highlight && !s.paths.some((p) => p.id === highlight) ? 0.3 : 1"
          @mouseenter="onStation(s.chapter)"
          @mouseleave="onStation(null)"
          @focus="onStation(s.chapter)"
          @blur="onStation(null)"
        >
          <NuxtLink :to="chapterHref(s.chapter)" :aria-label="s.chapter.title">
            <!-- interchange: a wider white capsule; a plain stop: a ring in its line's colour -->
            <circle :cx="s.x" :cy="s.y" r="22" fill="transparent" />
            <circle
              v-if="s.paths.length > 1"
              :cx="s.x"
              :cy="s.y"
              r="11"
              :fill="isRead(s.chapter.slug) ? palette.fg : palette.bg"
              :stroke="palette.fg"
              stroke-width="3.5"
            />
            <circle
              v-else
              :cx="s.x"
              :cy="s.y"
              :r="hovered === s.chapter ? 9.5 : 8"
              :fill="isRead(s.chapter.slug) ? palette[s.paths[0]!.color] : palette.bg"
              :stroke="palette[s.paths[0]!.color]"
              stroke-width="3.5"
              class="transition-all"
            />
            <text
              :x="s.x"
              :y="s.above ? s.y - 22 : s.y + 30"
              text-anchor="middle"
              class="font-sans text-[13px]"
              :fill="hovered === s.chapter ? palette.fg : palette.fgMuted"
            >
              <tspan
                v-for="(line, i) in labelLines(s.chapter.short)"
                :key="i"
                :x="s.x"
                :dy="i === 0 ? (s.above ? -(labelLines(s.chapter.short).length - 1) * 15 : 0) : 15"
              >{{ line }}</tspan>
            </text>
          </NuxtLink>
        </g>
      </svg>
    </div>

    <div
      v-if="tip && hovered"
      class="pointer-events-none absolute z-20 hidden w-72 rounded-[10px] border border-line-strong bg-raised p-4 shadow-2xl shadow-black/60 md:block"
      :style="{ left: tip.left, top: tip.top, transform: `translate(${tip.flip ? 'calc(-100% - 20px)' : '20px'}, -50%)` }"
    >
      <p class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-subtle">
        <span v-for="p in hoveredPaths" :key="p.id" class="flex items-center gap-1.5">
          <span class="size-2 rounded-full" :style="{ background: palette[p.color] }" />{{ p.title }}
        </span>
      </p>
      <p class="mt-1.5 font-display text-lg leading-tight">{{ hovered.title }}</p>
      <p class="mt-1.5 line-clamp-4 text-[13px] leading-relaxed text-fg-muted">{{ hovered.summary }}</p>
      <p class="mt-2 font-mono text-[11px] text-fg-subtle">
        {{ hovered.readMinutes }} min<template v-if="hovered.prerequisites?.length"> · builds on {{ hovered.prerequisites.map((s) => chapterBySlug(s)?.short).join(", ") }}</template>
      </p>
    </div>
  </div>
</template>

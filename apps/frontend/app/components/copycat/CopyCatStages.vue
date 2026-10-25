<script setup lang="ts">
import type { CopyCatStages } from "~/utils/copycat"

// The structured pipeline at work (docs/design/0022, "The game, as a player sees it"): the palette it settled on, the
// regions before and after cleaning, and the layers it stacked -- pulled apart in depth, so a part completed under
// another shows whole.
const props = defineProps<{ stages: CopyCatStages }>()

const explode = ref(0.6)
const hex = (rgb: readonly number[]) => `#${rgb.map((c) => c.toString(16).padStart(2, "0")).join("")}`
const colours = computed(() => props.stages.palette.slice(1).map(hex))
const clustered = computed(() => ({
  data: labelPixels(props.stages.clustered, props.stages.palette),
  width: props.stages.width,
  height: props.stages.height,
}))
const cleaned = computed(() => ({
  data: labelPixels(props.stages.labels, props.stages.palette),
  width: props.stages.width,
  height: props.stages.height,
}))
const stats = computed(() => [
  { label: "Noise σ", value: props.stages.sigma.toFixed(3) },
  { label: "Background", value: props.stages.gradient ? "gradient" : "flat" },
  { label: "Colours", value: colours.value.length },
  { label: "Shapes", value: props.stages.shapes },
  { label: "Path nodes", value: props.stages.nodes },
  { label: "Time", value: `${Math.round(props.stages.ms)} ms` },
])
const KIND: Record<string, string> = { rect: "Rect", circle: "Circle", ellipse: "Ellipse", polygon: "Polygon", path: "Bézier path" }
// bottom layer at the back: each one lifted toward the viewer by its place in the stack
const layerStyle = (i: number) => {
  const n = props.stages.layers.length
  const depth = (i - (n - 1) / 2) * 34 * explode.value
  return { transform: `rotateX(${52 * explode.value}deg) rotateZ(${-30 * explode.value}deg) translateZ(${depth}px)` }
}
</script>

<template>
  <div class="space-y-4">
    <UiStats :items="stats" :cols="6" />
    <div class="grid gap-4 md:grid-cols-[1fr_1fr_1.3fr]">
      <CopyCatImage label="Palette, merged" :pixels="clustered" note="k-means in OKLab">
        <template #empty>Tracing…</template>
      </CopyCatImage>
      <CopyCatImage label="Regions, cleaned" :pixels="cleaned" note="background: transparent" />
      <figure class="min-w-0">
        <div class="stack-frame relative aspect-square overflow-hidden rounded-[8px] border border-line bg-sunken">
          <div class="stack absolute inset-[12%]">
            <img
              v-for="(layer, i) in stages.layers"
              :key="i"
              :src="svgDataUrl(layer.svg)"
              :alt="`layer ${i + 1}`"
              class="layer absolute inset-0 size-full"
              :style="layerStyle(i)"
              draggable="false"
            />
          </div>
        </div>
        <figcaption class="mt-1.5 flex items-center justify-between gap-3 text-xs">
          <span class="font-medium text-fg">Layers, pulled apart</span>
          <label class="flex items-center gap-2 font-mono text-[11px] text-fg-subtle">
            explode
            <input v-model.number="explode" type="range" min="0" max="1" step="0.05" class="w-24 accent-queen-400" />
          </label>
        </figcaption>
      </figure>
    </div>
    <div class="flex flex-wrap items-center gap-2 text-xs">
      <span class="label">Palette</span>
      <span v-for="c in colours" :key="c" class="inline-flex items-center gap-1.5 font-mono text-[11px] text-fg-muted">
        <span class="size-4 rounded-[4px] border border-line-strong" :style="{ background: c }" />{{ c }}
      </span>
    </div>
    <ol class="flex flex-wrap gap-2 text-xs">
      <li v-for="(layer, i) in stages.layers" :key="i" class="inline-flex items-center gap-1.5 rounded-full border border-line px-2 py-0.5 text-fg-muted">
        <span class="font-mono text-[10.5px] text-fg-subtle">{{ i + 1 }}</span>
        <span class="size-2.5 rounded-full" :style="{ background: layer.fill }" />
        {{ KIND[layer.kind] ?? layer.kind }}
      </li>
    </ol>
  </div>
</template>

<style scoped>
.stack-frame {
  perspective: 900px;
}
.stack {
  transform-style: preserve-3d;
}
.layer {
  transition: transform 0.35s ease;
  filter: drop-shadow(0 6px 10px rgb(0 0 0 / 0.35));
}
</style>

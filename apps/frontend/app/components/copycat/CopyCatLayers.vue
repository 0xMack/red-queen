<script setup lang="ts">
// An SVG's shapes as layers, pulled apart in depth (docs/design/0022, "Stacked, not cut out"): the bottom one at the
// back, each lifted toward the viewer by its place in the stack, so a part completed under another shows whole.
const props = withDefaults(defineProps<{ layers: readonly string[]; label?: string }>(), { label: "Layers, pulled apart" })

const explode = ref(0.6)
const urls = computed(() => props.layers.map(svgDataUrl))
const layerStyle = (i: number) => {
  const depth = (i - (props.layers.length - 1) / 2) * Math.min(34, 220 / Math.max(props.layers.length, 1)) * explode.value
  return { transform: `rotateX(${52 * explode.value}deg) rotateZ(${-30 * explode.value}deg) translateZ(${depth}px)` }
}
</script>

<template>
  <figure class="min-w-0">
    <div class="stack-frame relative aspect-square overflow-hidden rounded-[8px] border border-line bg-sunken">
      <div class="stack absolute inset-[12%]">
        <img
          v-for="(url, i) in urls"
          :key="i"
          :src="url"
          :alt="`layer ${i + 1}`"
          class="layer absolute inset-0 size-full"
          :style="layerStyle(i)"
          draggable="false"
        />
      </div>
      <UiEmpty v-if="!layers.length" class="absolute inset-0" compact>No shapes yet</UiEmpty>
      <input
        v-model.number="explode"
        type="range"
        min="0"
        max="1"
        step="0.05"
        class="absolute right-2 bottom-2 left-2 accent-queen-400 opacity-60 transition-opacity hover:opacity-100 focus:opacity-100"
        aria-label="How far apart the layers are pulled"
      />
    </div>
    <figcaption class="mt-1.5 flex flex-wrap items-baseline justify-between gap-x-2 text-xs">
      <span class="font-medium text-fg">{{ label }}</span>
      <span class="min-w-0 truncate font-mono text-[11px] text-fg-subtle">{{ layers.length }} shapes</span>
    </figcaption>
  </figure>
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

<script setup lang="ts">
// One picture on Copy Cat's stage: an SVG (drawn crisp by the browser at any size) or a raster image (`src`, drawn as
// its pixels), over a checkerboard so transparency shows as transparency -- a leftover background is the first thing
// a viewer should see. `pixels` draws RGBA pixels (a label image) instead.
const props = defineProps<{
  label: string
  svg?: string | null
  src?: string | null
  pixels?: { data: Uint8ClampedArray; width: number; height: number } | null
  note?: string
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
watchEffect(() => {
  const el = canvas.value
  const p = props.pixels
  if (!el || !p) return
  el.width = p.width
  el.height = p.height
  el.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(p.data), p.width, p.height), 0, 0)
})
const url = computed(() => (props.svg ? svgDataUrl(props.svg) : (props.src ?? null)))
</script>

<template>
  <figure class="min-w-0">
    <div class="checker relative aspect-square overflow-hidden rounded-[8px] border border-line">
      <canvas v-if="pixels" ref="canvas" class="pixelated size-full" />
      <img v-else-if="url" :src="url" :alt="label" class="size-full object-contain" :class="src ? 'pixelated' : ''" draggable="false" />
      <UiEmpty v-else class="absolute inset-0" compact>
        <slot name="empty">Nothing yet</slot>
      </UiEmpty>
    </div>
    <figcaption class="mt-1.5 flex items-baseline justify-between gap-2 text-xs">
      <span class="truncate font-medium text-fg">{{ label }}</span>
      <span v-if="note" class="shrink-0 font-mono text-[11px] text-fg-subtle">{{ note }}</span>
    </figcaption>
  </figure>
</template>

<style scoped>
.checker {
  background-color: var(--color-raised);
  background-image:
    linear-gradient(45deg, var(--color-line) 25%, transparent 25%, transparent 75%, var(--color-line) 75%),
    linear-gradient(45deg, var(--color-line) 25%, transparent 25%, transparent 75%, var(--color-line) 75%);
  background-size: 16px 16px;
  background-position:
    0 0,
    8px 8px;
}
.pixelated {
  image-rendering: pixelated;
}
</style>

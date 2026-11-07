<script setup lang="ts">
import type { CopyCatTarget } from "~/utils/copycat"

// Which image is on Copy Cat's stage: one row under it -- a few held-out images from every set and corruption level,
// your own image once you've added one, a tile to add one, and "more", a small pop-up with every held-out image by set
// and level. Your image is read and traced on this device; nothing is uploaded.
const props = defineProps<{
  targets: CopyCatTarget[]
  /** The image on stage: a target's id, or OWN_IMAGE. */
  selected: string | null
  own: { png: string; name: string } | null
}>()
const emit = defineEmits<{ pick: [id: string]; upload: [file: File] }>()

const SEVERITIES = ["clean", "light", "medium", "heavy"] as const
/** Held-out images in the row (one fewer once yours is in it): each set in turn, its levels in this order, so the row
 *  shows the range. */
const SHOWN = 6
const LEVELS = ["medium", "heavy", "light", "clean"] as const

const sets = computed(() => [...new Set(props.targets.map((t) => t.set))])
const sample = computed(() => {
  const shown = props.own ? SHOWN - 1 : SHOWN
  const out: CopyCatTarget[] = []
  for (let round = 0; out.length < shown && round < LEVELS.length; round++) {
    sets.value.forEach((set, j) => {
      const level = LEVELS[(round + j) % LEVELS.length]
      const t = props.targets.find((x) => x.set === set && x.severity === level && !out.includes(x))
      if (t && out.length < shown) out.push(t)
    })
  }
  // the image on stage is always in the row
  const on = props.targets.find((t) => t.id === props.selected)
  if (on && !out.includes(on)) out.splice(out.length - 1, 1, on)
  return out
})

// the pop-up: hover (with a mouse) opens it after a beat, a click pins it, Escape or a click elsewhere closes it
const open = ref(false)
const pinned = ref(false)
const shownSet = ref<string | null>(null)
const set = computed(() => shownSet.value ?? props.selected?.split("/")[0] ?? sets.value[0] ?? "")
const setOptions = computed(() => sets.value.map((s) => ({ value: s, label: `${s} (${props.targets.filter((t) => t.set === s).length})` })))
const wrap = ref<HTMLElement | null>(null)
let timer: ReturnType<typeof setTimeout> | null = null
const clear = () => timer && clearTimeout(timer)
function enter(e: PointerEvent) {
  if (e.pointerType !== "mouse") return
  clear()
  timer = setTimeout(() => (open.value = true), open.value ? 0 : 250)
}
function leave(e: PointerEvent) {
  if (e.pointerType !== "mouse") return
  clear()
  if (!pinned.value) timer = setTimeout(() => (open.value = false), 200)
}
function toggle() {
  clear()
  pinned.value = !(open.value && pinned.value)
  open.value = pinned.value
}
function close() {
  open.value = false
  pinned.value = false
}
function choose(id: string) {
  emit("pick", id)
  close()
}
const onKey = (e: KeyboardEvent) => e.key === "Escape" && close()
const onDown = (e: PointerEvent) => !wrap.value?.contains(e.target as Node) && close()
watch(open, (now) => {
  if (now) {
    window.addEventListener("keydown", onKey)
    window.addEventListener("pointerdown", onDown)
  } else {
    window.removeEventListener("keydown", onKey)
    window.removeEventListener("pointerdown", onDown)
  }
})
onUnmounted(() => {
  clear()
  close()
})

const file = ref<HTMLInputElement | null>(null)
function onFile(event: Event) {
  const f = (event.target as HTMLInputElement).files?.[0]
  if (f) emit("upload", f)
  ;(event.target as HTMLInputElement).value = ""
}
const dragging = ref(false)
function onDrop(event: DragEvent) {
  dragging.value = false
  const f = event.dataTransfer?.files[0]
  if (f) emit("upload", f)
}

const tile = "aspect-square min-w-0 overflow-hidden rounded-[6px] border transition"
const ring = (on: boolean) => (on ? "border-queen-400 ring-1 ring-queen-400" : "border-line hover:border-line-strong")
const note = (t: CopyCatTarget) => `${t.set} ${t.seed} · ${t.severity}`
</script>

<template>
  <div ref="wrap" class="relative" data-testid="copycat-picker">
    <div class="grid auto-cols-fr grid-flow-col gap-1.5" role="group" aria-label="Which image">
      <button
        v-for="t in sample"
        :key="t.id"
        type="button"
        :class="[tile, ring(t.id === selected)]"
        :title="note(t)"
        @click="choose(t.id)"
      >
        <img :src="t.png" :alt="`held-out image ${note(t)}`" class="pixelated size-full" />
      </button>
      <button v-if="own" type="button" :class="[tile, ring(selected === OWN_IMAGE)]" :title="`Your image: ${own.name}`" @click="choose(OWN_IMAGE)">
        <img :src="own.png" :alt="`your image, ${own.name}`" class="pixelated size-full" />
      </button>
      <button
        type="button"
        :class="[tile, dragging ? 'border-queen-400 bg-queen-500/10' : 'border-dashed border-line-strong hover:border-queen-400']"
        class="flex items-center justify-center text-lg text-fg-subtle hover:text-fg"
        :title="own ? 'Trace another image of yours' : 'Trace your own image -- a generated logo or icon works best. Nothing is uploaded.'"
        :aria-label="own ? 'Trace another image of yours' : 'Trace your own image'"
        @click="file?.click()"
        @dragover.prevent="dragging = true"
        @dragleave="dragging = false"
        @drop.prevent="onDrop"
      >
        +
      </button>
      <input ref="file" type="file" accept="image/*" class="hidden" @change="onFile" />
      <button
        type="button"
        :class="[tile, open ? 'border-queen-400 text-fg' : 'border-line text-fg-subtle hover:border-line-strong hover:text-fg']"
        class="flex items-center justify-center"
        aria-label="Every held-out image"
        :aria-expanded="open"
        title="Every held-out image"
        @click="toggle"
        @pointerenter="enter"
        @pointerleave="leave"
      >
        <svg viewBox="0 0 16 16" class="size-4" fill="currentColor" aria-hidden="true">
          <rect x="1.5" y="1.5" width="5" height="5" rx="1" />
          <rect x="9.5" y="1.5" width="5" height="5" rx="1" />
          <rect x="1.5" y="9.5" width="5" height="5" rx="1" />
          <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
        </svg>
      </button>
    </div>

    <div
      v-if="open"
      class="absolute top-full left-0 z-30 mt-2 max-h-[22rem] w-[min(23rem,calc(100vw-2rem))] space-y-2 overflow-y-auto rounded-[10px] border border-line-strong bg-raised p-3 shadow-xl"
      role="dialog"
      aria-label="Every held-out image"
      @pointerenter="enter"
      @pointerleave="leave"
    >
      <UiSegmented v-if="sets.length > 1" :model-value="set" :options="setOptions" size="sm" aria-label="Which held-out set" @update:model-value="(s: string) => (shownSet = s)" />
      <div v-for="severity in SEVERITIES" :key="severity" class="flex items-start gap-2">
        <span class="label w-12 shrink-0 pt-2">{{ severity }}</span>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="t in targets.filter((x) => x.set === set && x.severity === severity)"
            :key="t.id"
            type="button"
            class="size-8 overflow-hidden rounded-[5px] border transition"
            :class="ring(t.id === selected)"
            :title="note(t)"
            @click="choose(t.id)"
          >
            <img :src="t.png" :alt="`held-out image ${note(t)}`" class="pixelated size-full" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.pixelated {
  image-rendering: pixelated;
}
</style>

<script setup lang="ts">
import { useId } from "vue"
import { concept, parseRef } from "~/data/explainers"
import type { ResolvedExplainer } from "~/types/explain"

// The ⓘ beside a name: hover (with a mouse) or click/tap/Enter opens a card explaining it (`ExplainerCard`); "More"
// opens the side panel. One popover is open at a time. The trigger swallows its click, so an ⓘ inside a clickable
// row (the standings, the table) never also selects the row. Renders nothing for a ref nothing explains.
const props = withDefaults(defineProps<{ subject: string; label?: string; size?: "xs" | "sm" }>(), { label: undefined, size: "sm" })

const { resolve, open: openPanel, openTip } = useExplain()
const uid = useId()
const known = computed(() => {
  const ref = parseRef(props.subject)
  return !!ref && (ref.kind === "entrant" || !!concept(props.subject))
})

const trigger = ref<HTMLElement | null>(null)
const pop = ref<HTMLElement | null>(null)
const explainer = shallowRef<ResolvedExplainer | null>(null)
const pinned = ref(false)
const isOpen = computed(() => openTip.value === uid && !!explainer.value)
const pos = ref({ left: 0, top: 0 })

const GUTTER = 16
const WIDTH = 336
const EDGE = 8

function place() {
  const el = trigger.value
  if (!el) return
  const r = el.getBoundingClientRect()
  const width = Math.min(WIDTH, window.innerWidth - 2 * GUTTER)
  const left = Math.max(GUTTER, Math.min(r.left + r.width / 2 - width / 2, window.innerWidth - width - GUTTER))
  // Below the trigger if it fits, else above; if neither, wherever there's more room -- and never off screen (the
  // card scrolls inside itself when the screen is shorter than it).
  const height = Math.min(pop.value?.offsetHeight ?? 420, window.innerHeight - 2 * EDGE)
  const below = window.innerHeight - r.bottom
  const above = below < height + 12 && r.top > below
  const top = above ? r.top - 8 - height : r.bottom + 8
  pos.value = { left, top: Math.max(EDGE, Math.min(top, window.innerHeight - height - EDGE)) }
}

function show(pin: boolean) {
  explainer.value = resolve(props.subject)
  if (!explainer.value) return
  pinned.value = pin || pinned.value
  openTip.value = uid
  place()
  nextTick(place) // again, now that the card's height is known
}
function hide() {
  if (openTip.value === uid) openTip.value = null
  pinned.value = false
}

// Hover (mouse only): open after a beat, close after leaving both the trigger and the card.
let timer: ReturnType<typeof setTimeout> | null = null
const clear = () => timer && clearTimeout(timer)
function enter(e: PointerEvent) {
  if (e.pointerType !== "mouse") return
  clear()
  timer = setTimeout(() => show(false), isOpen.value ? 0 : 280)
}
function leave(e: PointerEvent) {
  if (e.pointerType !== "mouse") return
  clear()
  if (!pinned.value) timer = setTimeout(hide, 180)
}
function toggle() {
  clear()
  if (isOpen.value && pinned.value) hide()
  else show(true)
}
function more() {
  hide()
  openPanel(props.subject)
}

// While open: follow the trigger, close on Escape or a click elsewhere.
function onKey(e: KeyboardEvent) {
  if (e.key === "Escape" && isOpen.value) {
    hide()
    trigger.value?.focus()
  }
}
function onDown(e: PointerEvent) {
  const t = e.target as Node
  if (isOpen.value && !pop.value?.contains(t) && !trigger.value?.contains(t)) hide()
}
watch(isOpen, (now) => {
  if (now) {
    window.addEventListener("scroll", place, { passive: true, capture: true })
    window.addEventListener("resize", place)
    window.addEventListener("keydown", onKey)
    window.addEventListener("pointerdown", onDown)
  } else {
    window.removeEventListener("scroll", place, { capture: true })
    window.removeEventListener("resize", place)
    window.removeEventListener("keydown", onKey)
    window.removeEventListener("pointerdown", onDown)
    pinned.value = false
  }
})
onUnmounted(() => {
  clear()
  hide()
})
</script>

<template>
  <span v-if="known" class="inline-flex align-middle">
    <button
      ref="trigger"
      type="button"
      class="info-trigger inline-flex shrink-0 items-center justify-center rounded-full border font-display leading-none italic transition"
      :class="[size === 'xs' ? 'size-3.5 text-[10px]' : 'size-4 text-[11.5px]', isOpen ? 'is-open' : '']"
      :aria-label="`What is ${label ?? subject}?`"
      :aria-expanded="isOpen"
      :aria-controls="isOpen ? `${uid}-pop` : undefined"
      @click.stop.prevent="toggle"
      @pointerenter="enter"
      @pointerleave="leave"
    >
      i
    </button>
    <Teleport to="body">
      <Transition name="tip">
        <div
          v-if="isOpen && explainer"
          :id="`${uid}-pop`"
          ref="pop"
          role="dialog"
          :aria-label="explainer.title"
          class="tip card fixed z-[70] max-h-[calc(100vh-16px)] overflow-y-auto p-4 shadow-2xl shadow-black/60"
          :style="{ left: `${pos.left}px`, top: `${pos.top}px`, width: `min(${WIDTH}px, calc(100vw - ${2 * GUTTER}px))` }"
          @pointerenter="enter"
          @pointerleave="leave"
        >
          <ExplainerCard :explainer="explainer" @more="more" @navigate="hide" />
        </div>
      </Transition>
    </Teleport>
  </span>
</template>

<style scoped>
.info-trigger {
  border-color: var(--color-line-strong);
  color: var(--color-fg-subtle);
  font-weight: 600;
}
.info-trigger:hover,
.info-trigger.is-open {
  border-color: var(--color-queen-300);
  color: var(--color-queen-200);
  background: color-mix(in srgb, var(--color-queen-500) 14%, transparent);
}
.tip {
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--color-queen-500) 5%, transparent), transparent 30%),
    var(--color-surface);
  border-color: var(--color-line-strong);
}
.tip-enter-active,
.tip-leave-active {
  transition:
    opacity 0.14s ease,
    scale 0.14s ease;
}
.tip-enter-from,
.tip-leave-to {
  opacity: 0;
  scale: 0.97;
}
</style>

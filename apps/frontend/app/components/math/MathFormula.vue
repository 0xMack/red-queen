<script setup lang="ts">
import { mathSymbol } from "~/data/math/symbols"
import { compile, compileWorked, termsOf, type BoundValue, type Formula, type TermDef } from "~/utils/math/expr"
import { renderTex } from "~/utils/math/katex"
import "katex/dist/katex.min.css"

// A formula whose terms are part of the page (docs/design/0012). Typeset by KaTeX from an expression tree; every term
// is hoverable, focusable and clickable (click pins it), coloured by its term scope, and linked to everything else in
// the scope that names it -- a slider, a table cell, a chart mark. With values bound (a lab's last update), a worked
// line underneath repeats the formula with the real numbers, term for term; or "numbers" swaps them in place.
// A term's card says what it is, its current value, and -- when it has other forms -- lets the reader switch.
const props = withDefaults(
  defineProps<{
    formula: Formula
    /** Initial form per term id. */
    forms?: Record<string, string>
    /** Values by term id; defaults to the scope's. */
    values?: Record<string, BoundValue | undefined>
    /** How bound values show: a worked line under the formula, or the numbers in place of the symbols. */
    valueMode?: "worked" | "numbers"
    /** A line under the formula: what it says. */
    caption?: string
    /** Inline in a sentence, not a display block. */
    inline?: boolean
  }>(),
  { forms: () => ({}), values: undefined, valueMode: "worked", caption: undefined, inline: false },
)

const scope = useOrProvideTermScope()
const terms = computed<TermDef[]>(() => termsOf(props.formula.body))
scope.register(terms.value)

const forms = ref<Record<string, string>>({ ...props.forms })
watch(() => props.forms, (f) => (forms.value = { ...f }))
const mode = ref(props.valueMode)

const values = computed(() => props.values ?? scope.values.value)
// A worked instance needs its result: until then (a lab between games, an untried machine) the formula stays symbolic
// rather than half-filled. A formula without a result is instantiated by any value.
const hasValues = computed(() =>
  props.formula.result ? values.value[props.formula.result] !== undefined : terms.value.some((t) => values.value[t.id] !== undefined),
)
const html = computed(() =>
  renderTex(compile(props.formula.body, { forms: forms.value, values: values.value, bind: hasValues.value && mode.value === "numbers" }), !props.inline),
)
const workedHtml = computed(() => {
  if (!hasValues.value || mode.value !== "worked") return null
  const t = compileWorked(props.formula, { forms: forms.value, values: values.value })
  return t ? renderTex(t, !props.inline) : null
})

// --- Linking: event delegation over the rendered terms, and colouring them from the scope ---------------------------
const root = ref<HTMLElement | null>(null)
const local = ref<{ id: string; el: HTMLElement } | null>(null)

function termAt(e: Event): HTMLElement | null {
  const el = (e.target as Element | null)?.closest?.("[data-term]") as HTMLElement | null
  return el && root.value?.contains(el) ? el : null
}
function onOver(e: Event) {
  const el = termAt(e)
  scope.hover(el?.dataset.term ?? null)
  if (el) local.value = { id: el.dataset.term!, el }
}
function onLeave() {
  scope.hover(null)
}
function onClick(e: Event) {
  const el = termAt(e)
  if (!el) return
  local.value = { id: el.dataset.term!, el }
  scope.pin(el.dataset.term!)
}
function onKey(e: KeyboardEvent) {
  if (e.key === "Escape") scope.pin(null)
  else if (e.key === "Enter" || e.key === " ") {
    const el = termAt(e)
    if (el) {
      e.preventDefault()
      onClick(e)
    }
  }
}

function paint() {
  const lit = scope.lit.value
  // Only a real focus dims the rest; a flash (the lab pointing at what just happened) just lights its term.
  root.value?.classList.toggle("math-has-focus", !!scope.active.value)
  root.value?.querySelectorAll<HTMLElement>("[data-term]").forEach((el) => {
    const id = el.dataset.term!
    el.style.setProperty("--term-color", scope.color(id))
    el.classList.toggle("term-on", lit.has(id))
    el.tabIndex = 0
    el.setAttribute("role", "button")
    el.setAttribute("aria-label", nameOf(id))
  })
}
watch([() => scope.lit.value, html, workedHtml], () => nextTick(paint))
onMounted(paint)

// --- The term card ----------------------------------------------------------------------------------------------------
const def = (id: string) => terms.value.find((t) => t.id === id) ?? scope.def(id)
function nameOf(id: string): string {
  const d = def(id)
  return d?.name ?? mathSymbol(d?.symbol)?.name ?? id
}

const cardId = computed(() => {
  const active = scope.active.value
  return active && local.value?.id === active && def(active) ? active : null
})
const card = computed(() => {
  const id = cardId.value
  if (!id) return null
  const d = def(id)!
  const sym = mathSymbol(d.symbol)
  const v = values.value[id]
  return {
    id,
    glyph: renderTex(compile(d.body), false),
    name: d.name ?? sym?.name ?? id,
    meaning: d.meaning ?? sym?.meaning,
    range: sym?.range,
    value: v === undefined ? null : typeof v === "number" ? (d.format ? d.format(v) : Number.isInteger(v) ? String(v) : v.toFixed(3)) : v,
    forms: d.expandable && d.forms ? [{ name: "default", label: "as written" }, ...Object.entries(d.forms).map(([name, f]) => ({ name, label: f.label }))] : [],
    form: forms.value[id] ?? "default",
    symbol: d.symbol,
    pinned: scope.pinned.value === id,
  }
})

const pos = ref({ left: 0, top: 0 })
// Anchored to the term's element -- found again after every re-render (a form switch or a new value replaces the markup),
// below it if the card fits there, else above.
function placeCard() {
  const id = cardId.value
  if (!id) return
  let el = local.value?.el
  if (!el?.isConnected) el = root.value?.querySelector<HTMLElement>(`[data-term="${id}"]`) ?? undefined
  if (!el) return
  local.value = { id, el }
  // Horizontally at the term; vertically clear of the whole figure, so the worked line under it stays readable.
  const r = el.getBoundingClientRect()
  const box = root.value!.getBoundingClientRect()
  const width = Math.min(300, window.innerWidth - 32)
  const height = cardEl.value?.offsetHeight ?? 180
  const below = box.bottom + 8 + height < window.innerHeight
  pos.value = {
    left: Math.max(16, Math.min(r.left + r.width / 2 - width / 2, window.innerWidth - width - 16)),
    top: below ? box.bottom + 8 : Math.max(8, box.top - 8 - height),
  }
}
const cardEl = ref<HTMLElement | null>(null)
watch([cardId, html, workedHtml], () => nextTick(() => nextTick(placeCard)))
function setForm(id: string, name: string) {
  forms.value = { ...forms.value, [id]: name }
}
const { open: openExplain } = useExplain()

// A click anywhere outside the formula and its card releases a pin this formula made.
function onDocClick(e: MouseEvent) {
  const t = e.target as Node
  if (cardId.value && scope.pinned.value === cardId.value && !root.value?.contains(t) && !(t as Element).closest?.("[data-term-card]")) scope.pin(null)
}
onMounted(() => document.addEventListener("click", onDocClick))
onUnmounted(() => document.removeEventListener("click", onDocClick))
</script>

<template>
  <figure
    ref="root"
    class="math-formula not-prose"
    :class="inline ? 'inline-block align-middle' : 'my-6'"
    :aria-label="formula.title"
    @mouseover="onOver"
    @mouseleave="onLeave"
    @focusin="onOver"
    @focusout="onLeave"
    @click="onClick"
    @keydown="onKey"
  >
    <div class="overflow-x-auto overflow-y-hidden text-fg" :class="inline ? '' : 'px-1 text-[1.12rem]'" v-html="html" />
    <div v-if="workedHtml" class="math-worked overflow-x-auto overflow-y-hidden text-fg-muted" v-html="workedHtml" />
    <figcaption v-if="(!inline && caption) || hasValues" class="mt-1 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs text-fg-subtle">
      <span v-if="caption">{{ caption }}</span>
      <UiSegmented
        v-if="hasValues && formula.worked"
        v-model="mode"
        size="sm"
        :options="[
          { value: 'worked', label: 'Symbols + values' },
          { value: 'numbers', label: 'Numbers in place' },
        ]"
      />
    </figcaption>

    <Teleport to="body">
      <div
        v-if="card"
        ref="cardEl"
        class="fixed z-50 w-[300px] max-w-[calc(100vw-32px)] rounded-[10px] border border-line-strong bg-raised p-4 text-left shadow-2xl shadow-black/60"
        :style="{ left: `${pos.left}px`, top: `${pos.top}px`, borderTopColor: scope.color(card.id) }"
        :class="card.pinned ? '' : 'pointer-events-none'"
        role="dialog"
        data-term-card
        :aria-label="card.name"
      >
        <p class="flex items-baseline gap-3">
          <span class="text-lg" :style="{ color: scope.color(card.id) }" v-html="card.glyph" />
          <span class="font-display text-lg leading-tight">{{ card.name }}</span>
        </p>
        <p v-if="card.meaning" class="mt-1.5 text-[13px] leading-relaxed text-fg-muted">{{ card.meaning }}</p>
        <p v-if="card.value !== null || card.range" class="mt-2 flex flex-wrap gap-x-4 font-mono text-[11px] text-fg-subtle">
          <span v-if="card.value !== null">now <span class="text-fg">{{ card.value }}</span></span>
          <span v-if="card.range">range {{ card.range }}</span>
        </p>
        <div v-if="card.forms.length" class="mt-3 flex flex-wrap gap-1.5">
          <button
            v-for="f in card.forms"
            :key="f.name"
            class="chip transition"
            :class="f.name === card.form ? 'text-fg ring-1 ring-fg-subtle' : 'hover:text-fg'"
            @click.stop="setForm(card.id, f.name)"
          >
            {{ f.label }}
          </button>
        </div>
        <p class="mt-3 flex items-center justify-between text-[11px] text-fg-subtle">
          <span>{{ card.pinned ? "Pinned · click it again or Esc to release" : "Click to pin" }}</span>
          <button v-if="card.symbol && card.pinned" class="link" @click.stop="openExplain(`symbol:${card.symbol}`)">More</button>
        </p>
      </div>
    </Teleport>
  </figure>
</template>

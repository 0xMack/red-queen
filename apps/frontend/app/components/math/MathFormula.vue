<script setup lang="ts">
import { mathSymbol } from "~/data/math/symbols"
import {
  checkValues,
  compile,
  compileFormula,
  computeValues,
  formatPlain,
  termsOfFormula,
  type BoundValue,
  type Formula,
  type TermDef,
} from "~/utils/math/expr"
import { renderTex } from "~/utils/math/katex"
import "katex/dist/katex.min.css"

// A formula whose terms are part of the page (docs/design/0012). Typeset by KaTeX from an expression tree; every term
// is hoverable, focusable and clickable (click pins it), coloured by its term scope, and linked to everything else in
// the scope that names it -- a slider, a table cell, a chart mark. With inputs bound (a lab's last update), the formula
// computes its derived terms and a second row repeats it with the real numbers, aligned on the relation like a line of
// working: `Q_{n+1} ← Q_n + …` over `= 0.40 + … = 0.52`. A term's card says what it is, its current value, what it's
// part of, and -- when it has other forms -- lets the reader switch.
const props = withDefaults(
  defineProps<{
    formula: Formula
    /** Initial form per term id. */
    forms?: Record<string, string>
    /** Input values by term id; defaults to the scope's. The formula computes the rest. */
    values?: Record<string, BoundValue | undefined>
    /** A line under the formula: what it says. */
    caption?: string
    /** No panel around it (inside a lab, which is already one). */
    bare?: boolean
  }>(),
  { forms: () => ({}), values: undefined, caption: undefined, bare: false },
)

const scope = useOrProvideTermScope()
const terms = computed<TermDef[]>(() => termsOfFormula(props.formula))
scope.register(terms.value)

const forms = ref<Record<string, string>>({ ...props.forms })
watch(() => props.forms, (f) => (forms.value = { ...f }))

// Inputs plus everything the formula works out from them (the error, the bonus, the result).
const values = computed(() => computeValues(props.formula, props.values ?? scope.values.value, forms.value))
// A worked row needs its result: until then (a lab between games, an untried machine) the formula stays symbolic
// rather than half-filled.
const worked = computed(() => !!props.formula.worked && !!props.formula.result && values.value[props.formula.result] !== undefined)
const html = computed(() =>
  renderTex(compileFormula(props.formula, { forms: forms.value, values: values.value, formats: scope.formats.value }, worked.value), true),
)

// The formula checking the algorithm, in development: what it computed against what the real code reported.
if (import.meta.dev) {
  watch([values, () => scope.expected.value], ([v, expected]) => {
    for (const problem of checkValues(v, expected)) console.warn(`formula ${props.formula.id} disagrees with the algorithm -- ${problem}`)
  })
}

// --- Hit-testing: by glyph, not by box -------------------------------------------------------------------------------
// A term's box is everything inside it -- a compound term's box covers its parts, and KaTeX's boxes include invisible
// struts and whole fraction stacks -- so "the element under the pointer" is often the wrong term. Instead: the glyphs
// (text, rules, the radical's svg) of the rendered formula, each belonging to its innermost term; the pointer picks the
// smallest glyph under it (or the nearest within a few pixels). Pointing at a compound term's own glyphs -- its minus
// sign, its brackets -- picks the compound; pointing at a letter inside it picks the letter.
const root = ref<HTMLElement | null>(null)
interface Glyph {
  el: Element
  term: string
}
let glyphs: Glyph[] | null = null

function collectGlyphs(): Glyph[] {
  const out: Glyph[] = []
  const html = root.value?.querySelector(".katex-html")
  if (!html) return out
  const walker = document.createTreeWalker(html, NodeFilter.SHOW_ELEMENT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const el = node as Element
    const glyph =
      el.tagName.toLowerCase() === "svg" ||
      el.classList.contains("frac-line") ||
      [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.textContent!.trim() !== "")
    if (!glyph) continue
    const term = el.closest("[data-term]")
    if (term && root.value!.contains(term)) out.push({ el, term: (term as HTMLElement).dataset.term! })
  }
  return out
}

const SLOP = 4
function termAtPoint(x: number, y: number): string | null {
  glyphs ??= collectGlyphs()
  let best: { term: string; area: number } | null = null
  let near: { term: string; d: number } | null = null
  for (const g of glyphs) {
    for (const r of g.el.getClientRects()) {
      if (r.width === 0 && r.height === 0) continue
      const inside = x >= r.left - 1 && x <= r.right + 1 && y >= r.top - 1 && y <= r.bottom + 1
      if (inside) {
        const area = r.width * r.height
        if (!best || area < best.area) best = { term: g.term, area }
      } else {
        const d = Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom))
        if (d <= SLOP && (!near || d < near.d)) near = { term: g.term, d }
      }
    }
  }
  return best?.term ?? near?.term ?? null
}

const local = ref<{ id: string; el: HTMLElement } | null>(null)
function elementFor(id: string): HTMLElement | null {
  return root.value?.querySelector<HTMLElement>(`[data-term="${id}"]`) ?? null
}

// No throttling needed: browsers already deliver pointer moves at most once per frame.
function onPointerMove(e: PointerEvent) {
  if (e.pointerType === "touch") return // touch has no hover: a tap pins
  const id = termAtPoint(e.clientX, e.clientY)
  root.value?.toggleAttribute("data-hit", !!id)
  if (id !== scope.hovered.value) scope.hover(id)
  if (id) local.value = { id, el: (document.elementFromPoint(e.clientX, e.clientY)?.closest(`[data-term="${id}"]`) as HTMLElement) ?? elementFor(id)! }
}
function onPointerLeave() {
  root.value?.removeAttribute("data-hit")
  scope.hover(null)
}
function onClick(e: MouseEvent) {
  const id = e.detail === 0 ? ((e.target as Element).closest?.("[data-term]") as HTMLElement | null)?.dataset.term : termAtPoint(e.clientX, e.clientY)
  if (!id) return
  const under = e.detail === 0 ? (e.target as Element) : document.elementFromPoint(e.clientX, e.clientY)
  local.value = { id, el: (under?.closest(`[data-term="${id}"]`) as HTMLElement | null) ?? elementFor(id)! }
  scope.pin(id)
}
function onFocusIn(e: FocusEvent) {
  const el = (e.target as Element).closest?.("[data-term]") as HTMLElement | null
  if (!el) return
  local.value = { id: el.dataset.term!, el }
  scope.hover(el.dataset.term!)
}
function onKey(e: KeyboardEvent) {
  if (e.key === "Escape") scope.pin(null)
  else if (e.key === "Enter" || e.key === " ") {
    const el = (e.target as Element).closest?.("[data-term]") as HTMLElement | null
    if (el) {
      e.preventDefault()
      local.value = { id: el.dataset.term!, el }
      scope.pin(el.dataset.term!)
    }
  }
}

// --- Painting, and keeping keyboard focus across re-renders ----------------------------------------------------------
// Every new value re-renders the formula's markup; a term focused from the keyboard would otherwise vanish with it.
// Before the swap, remember which term (and which occurrence of it) had focus; after, focus the same one again.
let refocus: { id: string; index: number } | null = null
watch(
  html,
  () => {
    glyphs = null
    const active = document.activeElement as HTMLElement | null
    if (active && root.value?.contains(active) && active.dataset.term) {
      const same = [...root.value.querySelectorAll<HTMLElement>(`[data-term="${active.dataset.term}"]`)]
      refocus = { id: active.dataset.term, index: Math.max(0, same.indexOf(active)) }
    }
  },
  { flush: "pre" },
)

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
  if (refocus) {
    const same = root.value?.querySelectorAll<HTMLElement>(`[data-term="${refocus.id}"]`)
    same?.[Math.min(refocus.index, same.length - 1)]?.focus({ preventScroll: true })
    refocus = null
  }
}
watch([() => scope.lit.value, html], () => nextTick(paint))
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
  // What it's part of *here*: the enclosing term of this occurrence (the same Q_n can stand alone and sit inside a
  // bracket), read from the element that was pointed at.
  const el = local.value?.el
  const parentEl = el?.isConnected ? (el.parentElement?.closest("[data-term]") as HTMLElement | null) : null
  const parent = parentEl && root.value?.contains(parentEl) ? parentEl.dataset.term! : null
  return {
    id,
    glyph: renderTex(compile(d.body), false),
    name: d.name ?? sym?.name ?? id,
    meaning: d.meaning ?? sym?.meaning,
    range: sym?.range,
    value: v === undefined ? null : formatPlain(v, d, { formats: scope.formats.value }),
    forms: d.expandable && d.forms ? [{ name: "default", label: "as written" }, ...Object.entries(d.forms).map(([name, f]) => ({ name, label: f.label }))] : [],
    form: forms.value[id] ?? "default",
    symbol: d.symbol,
    parent: parent ? { id: parent, name: nameOf(parent) } : null,
    pinned: scope.pinned.value === id,
  }
})

const pos = ref({ left: 0, top: 0 })
const cardEl = ref<HTMLElement | null>(null)
// Anchored to the term's element -- found again after every re-render (a form switch or a new value replaces the
// markup) -- horizontally at the term, vertically clear of the whole formula so both rows stay readable.
function placeCard() {
  const id = cardId.value
  if (!id) return
  let el: HTMLElement | null | undefined = local.value?.el
  if (!el?.isConnected) el = elementFor(id)
  if (!el) return
  local.value = { id, el }
  const r = el.getBoundingClientRect()
  const box = root.value!.getBoundingClientRect()
  const width = Math.min(320, window.innerWidth - 32)
  const height = cardEl.value?.offsetHeight ?? 180
  const below = box.bottom + 8 + height < window.innerHeight
  pos.value = {
    left: Math.max(16, Math.min(r.left + r.width / 2 - width / 2, window.innerWidth - width - 16)),
    top: below ? box.bottom + 8 : Math.max(8, box.top - 8 - height),
  }
}
watch([cardId, html], () => nextTick(() => nextTick(placeCard)))
function setForm(id: string, name: string) {
  forms.value = { ...forms.value, [id]: name }
}
/** Step out: pin the term this one is part of (the bracket around r, the bonus around c). */
function pinParent(id: string) {
  const inner = local.value?.el
  const el = (inner?.isConnected ? (inner.parentElement?.closest(`[data-term="${id}"]`) as HTMLElement | null) : null) ?? elementFor(id)
  if (!el) return
  local.value = { id, el }
  scope.pin(id)
}
const { open: openExplain } = useExplain()

// A click anywhere outside the formula and its card releases a pin this formula made.
function onDocClick(e: MouseEvent) {
  const t = e.target as Node
  if (cardId.value && scope.pinned.value === cardId.value && !root.value?.contains(t) && !(t as Element).closest?.("[data-term-card]")) scope.pin(null)
}
onMounted(() => document.addEventListener("click", onDocClick))
onUnmounted(() => {
  document.removeEventListener("click", onDocClick)
})
</script>

<template>
  <figure
    ref="root"
    class="math-formula not-prose"
    :class="bare ? '' : 'math-panel my-7'"
    :aria-label="formula.title"
    @pointermove="onPointerMove"
    @pointerleave="onPointerLeave"
    @focusin="onFocusIn"
    @focusout="onPointerLeave"
    @click="onClick"
    @keydown="onKey"
  >
    <div class="overflow-x-auto overflow-y-hidden px-1 text-[1.15rem] text-fg" v-html="html" />
    <figcaption v-if="caption" class="math-caption">{{ caption }}</figcaption>

    <Teleport to="body">
      <div
        v-if="card"
        ref="cardEl"
        class="math-card fixed z-50 w-[320px] max-w-[calc(100vw-32px)]"
        :style="{ left: `${pos.left}px`, top: `${pos.top}px`, '--term-color': scope.color(card.id) }"
        :class="card.pinned ? '' : 'pointer-events-none'"
        role="dialog"
        data-term-card
        :aria-label="card.name"
      >
        <div class="flex items-center gap-3">
          <span class="math-card-glyph" v-html="card.glyph" />
          <div class="min-w-0">
            <p class="font-display text-[1.15rem] leading-tight">{{ card.name }}</p>
            <p v-if="card.value !== null || card.range" class="mt-0.5 flex flex-wrap items-baseline gap-x-3 font-mono text-[11px] text-fg-subtle">
              <span v-if="card.value !== null">now <span class="math-card-value">{{ card.value }}</span></span>
              <span v-if="card.range">{{ card.range }}</span>
            </p>
          </div>
        </div>
        <p v-if="card.meaning" class="mt-3 text-[13px] leading-relaxed text-fg-muted">{{ card.meaning }}</p>
        <div v-if="card.forms.length" class="mt-3">
          <p class="label mb-1.5">Show it</p>
          <UiSegmented
            :model-value="card.form"
            :options="card.forms.map((f) => ({ value: f.name, label: f.label }))"
            aria-label="Form"
            @update:model-value="(v) => setForm(card!.id, v as string)"
          />
        </div>
        <p v-if="card.parent" class="mt-3 text-[12px] text-fg-subtle">
          Part of
          <button v-if="card.pinned" class="link" @click.stop="pinParent(card.parent.id)">{{ card.parent.name }}</button>
          <span v-else class="text-fg-muted">{{ card.parent.name }}</span>
        </p>
        <p class="mt-3 flex items-center justify-between border-t border-line pt-2.5 text-[11px] text-fg-subtle">
          <span>{{ card.pinned ? "Pinned · Esc or click it to release" : "Click the term to pin" }}</span>
          <button v-if="card.symbol && card.pinned" class="link" @click.stop="openExplain(`symbol:${card.symbol}`)">More →</button>
        </p>
      </div>
    </Teleport>
  </figure>
</template>

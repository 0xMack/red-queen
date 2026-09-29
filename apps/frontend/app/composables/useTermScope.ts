import type { InjectionKey } from "vue"
import { termsOf, type BoundValue, type TermDef } from "~/utils/math/expr"
import { palette, type PaletteColor } from "~/utils/palette"

// A term scope (docs/design/0012): one figure's shared focus. Everything inside it that names a term -- a formula's
// terms, an inline <MathTerm>, a slider, a table cell, an SVG mark -- takes part: pointing at any of them focuses the
// term, and every other place that term appears lights up. Colours are assigned per scope, in the order terms are
// registered (the formula's reading order), so α is the same colour in the formula, on its slider and in the chart.
// Outside a scope, `useTermScope()` returns an inert one: components can always call `target()`.

const COLORS: PaletteColor[] = ["gold400", "signal400", "life400", "violet400", "teal400", "orange400", "pink400", "queen300"]

export interface TermScope {
  /** The focused term: the pinned one, else the hovered one. */
  active: ComputedRef<string | null>
  /** Everything that lights up with it: the term and the terms it's built from (the bonus lights c, t and N too). */
  lit: ComputedRef<Set<string>>
  hovered: Ref<string | null>
  pinned: Ref<string | null>
  /** Values bound to terms right now (the lab writes them): shown on cards and in worked lines. */
  values: Ref<Record<string, BoundValue | undefined>>
  register: (defs: TermDef[] | string[]) => void
  color: (id: string) => string
  /** What a term is, for its card: the first definition registered under the id. */
  def: (id: string) => TermDef | undefined
  hover: (id: string | null) => void
  /** Light a term for a moment without focusing it: the lab saying "this just happened" (the branch a pull took). */
  flash: (id: string, ms?: number) => void
  pin: (id: string | null) => void
  /** Bindings that make any element (HTML or SVG) take part as `ids`: `v-bind="scope.target('alpha')"`. */
  target: (ids: string | string[], opts?: { interactive?: boolean }) => Record<string, unknown>
  inert: boolean
}

const KEY: InjectionKey<TermScope> = Symbol("term-scope")

function createScope(inert: boolean): TermScope {
  const hovered = ref<string | null>(null)
  const pinned = ref<string | null>(null)
  const values = ref<Record<string, BoundValue | undefined>>({})
  // Plain, not reactive: `target()` registers during render, and a colour never changes once assigned.
  const order: string[] = []
  const defs = new Map<string, TermDef>()
  const active = computed(() => pinned.value ?? hovered.value)
  const flashed = ref<string | null>(null)
  let flashTimer: ReturnType<typeof setTimeout> | null = null
  const lit = computed(() => {
    const id = active.value ?? flashed.value
    if (!id) return new Set<string>()
    const d = defs.get(id)
    return new Set([id, ...(d ? termsOf(d.body).map((t) => t.id) : [])])
  })

  function register(list: TermDef[] | string[]) {
    const ids = list.map((d) => (typeof d === "string" ? d : d.id))
    for (const d of list) if (typeof d !== "string" && !defs.has(d.id)) defs.set(d.id, d)
    for (const id of ids) if (!order.includes(id)) order.push(id)
  }
  function color(id: string): string {
    const i = order.indexOf(id)
    return palette[COLORS[(i < 0 ? 0 : i) % COLORS.length]!]
  }
  const hover = (id: string | null) => {
    if (!inert) hovered.value = id
  }
  const flash = (id: string, ms = 700) => {
    if (inert) return
    flashed.value = id
    if (flashTimer) clearTimeout(flashTimer)
    flashTimer = setTimeout(() => (flashed.value = null), ms)
  }
  const pin = (id: string | null) => {
    if (!inert) pinned.value = pinned.value === id ? null : id
  }

  function target(ids: string | string[], opts: { interactive?: boolean } = {}) {
    if (inert) return {}
    const list = Array.isArray(ids) ? ids : [ids]
    register(list)
    const litId = list.find((id) => lit.value.has(id))
    const on = litId !== undefined
    return {
      "data-term": list[0],
      class: ["term-target", { "term-on": on, "term-dim": !!active.value && !on }],
      style: { "--term-color": color(litId ?? list[0]!) },
      onMouseenter: () => hover(list[0]!),
      onMouseleave: () => hover(null),
      ...(opts.interactive === false ? {} : { onFocusin: () => hover(list[0]!), onFocusout: () => hover(null) }),
    }
  }

  return { active, lit, hovered, pinned, values, register, color, def: (id) => defs.get(id), hover, flash, pin, target, inert }
}

/** Opens a scope for this component's subtree. */
export function provideTermScope(): TermScope {
  const scope = createScope(false)
  provide(KEY, scope)
  return scope
}

/** The nearest scope, or an inert one. */
export function useTermScope(): TermScope {
  return inject(KEY, null) ?? createScope(true)
}

/** The nearest scope, or a new one of its own: a formula standing alone still links its own terms. */
export function useOrProvideTermScope(): TermScope {
  return inject(KEY, null) ?? provideTermScope()
}

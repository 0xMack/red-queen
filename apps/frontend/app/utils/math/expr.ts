// Formulas as expression trees (docs/design/0012). A formula is written with the builders below; `compile` turns it
// into LaTeX for KaTeX, wrapping every *term* -- a symbol or a named sub-expression -- in `\htmlData{term=<id>}`, so
// the rendered formula is addressable: hover, focus, colour and link any term to the rest of the page.
//
// Three things make a formula more than a string:
// - **forms**: a term can be shown several ways (δ, or the bracket it abbreviates; 1/n, or α). The page picks a
//   default, the reader can switch where the term is `expandable`.
// - **values**: bind a term id to a number and it renders as that number, still carrying its term id -- so a worked
//   row with real values links to everything the symbolic one does.
// - **evaluation**: operators are nodes (`ops`, `mul`, `frac`, `sqrt`, `fn`), so a formula can compute its own derived
//   terms from its inputs (the error from r and Q, the score from Q and the bonus) -- a lab supplies only the inputs --
//   and bracket a negative number that follows an operator. With the value the real algorithm reported, the formula
//   checks itself: `checkValues`.
// Compiling is a pure function of (formula, view), so the server and the client produce the same markup.

export type Op = "+" | "-"

export type Expr =
  | { k: "tex"; tex: string }
  | { k: "seq"; parts: Expr[] }
  | { k: "ops"; first: Expr; rest: { op: Op; e: Expr }[] }
  | { k: "mul"; parts: Expr[] }
  | { k: "fn"; name: string; tex: string; arg: Expr }
  | { k: "pow"; base: Expr; exp: Expr }
  | { k: "term"; term: TermDef }
  | { k: "frac"; num: Expr; den: Expr; size: "t" | "d" | "" }
  | { k: "sqrt"; body: Expr }
  | { k: "paren"; body: Expr; open: string; close: string }
  | { k: "cases"; rows: { value: Expr; when: Expr }[] }

export interface TermForm {
  label: string
  expr: Expr
}

export interface TermDef {
  id: string
  body: Expr
  /** The symbol registry entry this term is an instance of (`data/math/symbols.ts`): its card's name and meaning. */
  symbol?: string
  /** What this term is, here -- overrides the symbol's name ("the estimate before the pull"). */
  name?: string
  meaning?: string
  /** Other ways to show the term, by name; `body` is the form called "default". */
  forms?: Record<string, TermForm>
  /** Whether the reader may switch forms (else only the page's view picks one). */
  expandable?: boolean
  /** How a bound value is written, as a fixed-width number (preferred: it doesn't move as the value changes). */
  num?: NumFormat
  /** Or as any string -- for values that aren't numbers in a range. Takes precedence over `num`. */
  format?: (value: number) => string
}

/** A fixed-width number: always `decimals` places, and padded (with invisible digits, and an invisible minus when it
 *  could be negative) to the width of the largest value it can take -- so a worked row doesn't shift when 9.50 becomes
 *  10.00 or 0.40 becomes −0.40. TeX's digits are all one width, so the padding is exact. */
export interface NumFormat {
  decimals?: number
  /** The largest magnitude it can take: sets how many integer digits to reserve. */
  max?: number
  /** It can be negative: reserve a minus sign (and, after an operator, the brackets a negative number needs). */
  signed?: boolean
}

type Part = Expr | string

const toExpr = (p: Part): Expr => (typeof p === "string" ? { k: "tex", tex: p } : p)

export const tex = (s: string): Expr => ({ k: "tex", tex: s })
/** Juxtaposed pieces of notation, not arithmetic (`a_t =`, `\arg\max_a`): not evaluable. */
export const seq = (...parts: Part[]): Expr => ({ k: "seq", parts: parts.map(toExpr) })
/** A sum and difference: `ops(a, "+", b, "-", c)`. */
export function ops(first: Part, ...rest: (Part | Op)[]): Expr {
  const out: { op: Op; e: Expr }[] = []
  for (let i = 0; i < rest.length; i += 2) out.push({ op: rest[i] as Op, e: toExpr(rest[i + 1] as Part) })
  return { k: "ops", first: toExpr(first), rest: out }
}
export const add = (...parts: Part[]): Expr => ops(parts[0]!, ...parts.slice(1).flatMap((p) => ["+" as Op, p]))
export const sub = (a: Part, b: Part): Expr => ops(a, "-", b)
/** A product: juxtaposed as symbols (γ max Q), with a · once values are bound (0.9 · 10.00). */
export const mul = (...parts: Part[]): Expr => ({ k: "mul", parts: parts.map(toExpr) })
export const frac = (num: Part, den: Part, size: "t" | "d" | "" = ""): Expr => ({ k: "frac", num: toExpr(num), den: toExpr(den), size })
export const sqrt = (body: Part): Expr => ({ k: "sqrt", body: toExpr(body) })
export const ln = (arg: Part): Expr => ({ k: "fn", name: "ln", tex: "\\ln", arg: toExpr(arg) })
export const tanh = (arg: Part): Expr => ({ k: "fn", name: "tanh", tex: "\\tanh", arg: toExpr(arg) })
export const exp = (arg: Part): Expr => ({ k: "fn", name: "exp", tex: "\\exp", arg: toExpr(arg) })
/** A power: `pow(e, "2")` is e². The base is bracketed when it's compound. */
export const pow = (base: Part, exponent: Part): Expr => ({ k: "pow", base: toExpr(base), exp: toExpr(exponent) })
export const paren = (body: Part, open = "(", close = ")"): Expr => ({ k: "paren", body: toExpr(body), open, close })
export const bracket = (body: Part): Expr => paren(body, "[", "]")
export const cases = (...rows: [Part, Part][]): Expr => ({ k: "cases", rows: rows.map(([value, when]) => ({ value: toExpr(value), when: toExpr(when) })) })

export function term(
  id: string,
  body: Part,
  opts: Omit<TermDef, "id" | "body" | "forms"> & { forms?: Record<string, { label: string; expr: Part }> } = {},
): Expr {
  const forms = opts.forms && Object.fromEntries(Object.entries(opts.forms).map(([name, f]) => [name, { label: f.label, expr: toExpr(f.expr) }]))
  return { k: "term", term: { ...opts, id, body: toExpr(body), forms } }
}

export interface Formula {
  id: string
  /** What the formula says, in words -- its accessible label and the card's heading. */
  title: string
  /** The left-hand side and the relation (`Q_{n+1}`, `\leftarrow`). With them, the worked row aligns under the
   *  relation -- a derivation: `Q_{n+1} ← Q_n + …` over `= 0.40 + … = 0.52`. */
  lhs?: Expr
  rel?: string
  /** The right-hand side (or the whole formula, without `lhs`). */
  body: Expr
  /** The worked instance: the right-hand side with values bound, ending `= result`. */
  worked?: Expr
  /** The worked row's own left-hand side, when it isn't just "=" under the formula's (UCB: `score(C) =`). */
  workedLhs?: Expr
  /** The term the worked row ends on (`… = 0.52`): evaluated from `worked`, unless supplied. */
  result?: string
  /** Terms the worked row shows as plain numbers, not opened up into their parts -- when their parts are another
   *  formula's business (the chain rule shows `e` and `a` as values; the forward pass works them out). */
  atoms?: string[]
}

/** Every part of a formula, for finding and registering its terms. */
const partsOf = (f: Formula): Expr[] => [f.lhs, f.body, f.workedLhs, f.worked].filter((e): e is Expr => !!e)

export const formula = (f: Formula): Formula => f

export type BoundValue = number | string

export interface FormulaView {
  /** A form name per term id ("default" is the term's body). */
  forms?: Record<string, string>
  /** Values by term id; a bound term renders as its value. */
  values?: Record<string, BoundValue | undefined>
  /** Bind values (the worked row) or not (symbols). */
  bind?: boolean
  /** Number formats by term id, for this instance -- the lab knows ranges the formula can't (this game's budget, its
   *  largest payout). Merged over each term's own `num`. */
  formats?: Record<string, NumFormat>
  /** Terms shown as their bound value even where they could open up (see `Formula.atoms`). */
  atoms?: string[]
}

// Compiling state that isn't part of the view: whether the expression being compiled follows an operator (a negative
// number there needs brackets), and whether any number shown was rounded (the result is then "≈", not "=").
interface Ctx {
  view: FormulaView
  operand: boolean
  rounded: { any: boolean }
}

const warned = new Set<string>()

function spec(def: TermDef, view: FormulaView): NumFormat {
  return { ...def.num, ...view.formats?.[def.id] }
}

/** A value as plain text, in its term's format (what a card shows). */
export function formatPlain(v: BoundValue, def: TermDef, view: FormulaView = {}): string {
  if (typeof v === "string") return v
  if (def.format) return def.format(v)
  const s = spec(def, view)
  const decimals = s.decimals ?? (Number.isInteger(v) ? 0 : 2)
  return (v < 0 ? "−" : "") + Math.abs(v).toFixed(decimals)
}

/** A value as LaTeX, padded to its format's full width (see `NumFormat`); bracketed when negative after an operator. */
function formatValue(v: BoundValue, def: TermDef, ctx: Ctx): string {
  if (typeof v === "string") return v
  if (def.format) return def.format(v)
  const s = spec(def, ctx.view)
  const decimals = s.decimals ?? (Number.isInteger(v) ? 0 : 2)
  const digits = Math.abs(v).toFixed(decimals)
  if (Math.abs(Math.abs(v) - Number(digits)) > 1e-9) ctx.rounded.any = true
  let pad = ""
  if (s.max !== undefined) {
    const room = String(Math.floor(Math.abs(s.max))).length
    const used = digits.split(".")[0]!.length
    if (used > room && import.meta.dev && !warned.has(def.id)) {
      warned.add(def.id)
      console.warn(`formula term "${def.id}": ${v} is outside its declared range (max ${s.max}) -- the formula will shift`)
    }
    if (room > used) pad = `\\phantom{${"0".repeat(room - used)}}`
  }
  // `{-}`: an ordinary minus (a sign, not a binary operator), so it and its invisible stand-in are the same width.
  const signed = s.signed || v < 0
  const sign = v < 0 ? "{-}" : signed ? "\\phantom{{-}}" : ""
  // After an operator a negative number is bracketed -- `1 − (−0.50)`, never `1 − −0.50` -- and a number that *could*
  // be negative reserves the brackets' width, so it doesn't move when it changes sign.
  if (ctx.operand && signed) return v < 0 ? `(${pad}${sign}${digits})` : `\\phantom{(}${pad}${sign}${digits}\\phantom{)}`
  return `${pad}${sign}${digits}`
}

/** An argument or a base that must be bracketed: a sum, a product, or a signed number once bound. */
function needsParens(e: Expr, view: FormulaView): boolean {
  if (isSum(e, view)) return true
  if (e.k === "mul") return true
  if (e.k === "term") {
    const bound = view.bind ? view.values?.[e.term.id] : undefined
    if (typeof bound === "number") return bound < 0
    return needsParens(shownForm(e.term, view), view)
  }
  return false
}

/** A sum or difference as shown (a term showing one counts): it needs brackets as a factor. */
function isSum(e: Expr, view: FormulaView): boolean {
  if (e.k === "ops") return e.rest.length > 0
  if (e.k === "term") {
    const bound = view.bind ? view.values?.[e.term.id] : undefined
    if (bound !== undefined && (!hasTerms(shownForm(e.term, view)) || view.atoms?.includes(e.term.id))) return false
    return isSum(shownForm(e.term, view), view)
  }
  return false
}

function shownForm(def: TermDef, view: FormulaView): Expr {
  const name = view.forms?.[def.id]
  return name && def.forms?.[name] ? def.forms[name]!.expr : def.body
}

function compileIn(e: Expr, ctx: Ctx): string {
  const plain = { ...ctx, operand: false }
  const c = (x: Expr) => compileIn(x, plain)
  switch (e.k) {
    case "tex":
      return e.tex
    case "seq":
      return e.parts.map(c).join(" ")
    case "ops":
      return [compileIn(e.first, ctx), ...e.rest.map((r) => `${r.op === "+" ? "+" : "-"} ${compileIn(r.e, { ...ctx, operand: true })}`)].join(" ")
    case "mul":
      // Precedence: a sum or difference that is a factor gets brackets -- (1 − λ)·V, never 1 − λ V.
      return e.parts
        .map((p, i) => {
          const out = compileIn(p, i === 0 ? ctx : { ...ctx, operand: true })
          return isSum(p, ctx.view) ? `\\left( ${out} \\right)` : out
        })
        .join(ctx.view.bind ? " \\cdot " : " \\, ")
    case "fn":
      // `\ln t` for a single symbol or number, `\tanh(x w + b)` for anything compound or negative.
      return needsParens(e.arg, ctx.view) ? `${e.tex}\\left( ${c(e.arg)} \\right)` : `${e.tex} ${c(e.arg)}`
    case "pow": {
      const base = c(e.base)
      return `${needsParens(e.base, ctx.view) || e.base.k === "frac" ? `\\left( ${base} \\right)` : `{${base}}`}^{${c(e.exp)}}`
    }
    case "frac":
      return `\\${e.size}frac{${c(e.num)}}{${c(e.den)}}`
    case "sqrt":
      return `\\sqrt{${c(e.body)}}`
    case "paren":
      return `\\left${e.open === "[" ? "[" : e.open === "{" ? "\\{" : e.open} ${c(e.body)} \\right${e.close === "]" ? "]" : e.close === "}" ? "\\}" : e.close}`
    case "cases":
      return `\\begin{cases} ${e.rows.map((r) => `${c(r.value)} & ${c(r.when)}`).join(" \\\\ ")} \\end{cases}`
    case "term": {
      const def = e.term
      const shown = shownForm(def, ctx.view)
      // A bound term shows its value -- unless what it shows is built from other terms *and can be worked out from
      // them* (1/n, r − Q): then the value goes one level down, into those terms, and the structure stays readable
      // (1/5, 1 − 0.40). A form that can't be worked out here (Q_n as an average of payouts the page doesn't have)
      // shows the number instead of a half-filled expression.
      const bound = ctx.view.bind ? ctx.view.values?.[def.id] : undefined
      const opensUp =
        !ctx.view.atoms?.includes(def.id) && hasTerms(shown) && evaluate(shown, ctx.view.values ?? {}, ctx.view.forms) !== undefined
      const inner = bound !== undefined && !opensUp ? formatValue(bound, def, ctx) : compileIn(shown, ctx)
      return `\\htmlData{term=${def.id}}{${inner}}`
    }
  }
}

/** The LaTeX for an expression under a view. Every term is `\htmlData{term=id}{…}`. */
export function compile(e: Expr, view: FormulaView = {}): string {
  return compileIn(e, { view, operand: false, rounded: { any: false } })
}

// --- Evaluation ------------------------------------------------------------------------------------------------------

/** The value of an expression from its inputs, as the page shows it (the chosen form of each term), or undefined
 *  where it isn't arithmetic (notation, an unbound symbol, a sum over data the page doesn't have). */
export function evaluate(e: Expr, inputs: Record<string, BoundValue | undefined>, forms: Record<string, string> = {}): number | undefined {
  const ev = (x: Expr) => evaluate(x, inputs, forms)
  switch (e.k) {
    case "tex": {
      const n = Number(e.tex.trim())
      return e.tex.trim() !== "" && Number.isFinite(n) ? n : undefined
    }
    case "seq":
      return e.parts.length === 1 ? ev(e.parts[0]!) : undefined
    case "ops": {
      let total = ev(e.first)
      for (const r of e.rest) {
        const v = ev(r.e)
        if (total === undefined || v === undefined) return undefined
        total = r.op === "+" ? total + v : total - v
      }
      return total
    }
    case "mul": {
      let total: number | undefined = 1
      for (const p of e.parts) {
        const v = ev(p)
        if (total === undefined || v === undefined) return undefined
        total *= v
      }
      return total
    }
    case "fn": {
      const v = ev(e.arg)
      if (v === undefined) return undefined
      return e.name === "ln" ? Math.log(v) : e.name === "tanh" ? Math.tanh(v) : e.name === "exp" ? Math.exp(v) : undefined
    }
    case "pow": {
      const b = ev(e.base)
      const x = ev(e.exp)
      return b === undefined || x === undefined ? undefined : b ** x
    }
    case "frac": {
      const a = ev(e.num)
      const b = ev(e.den)
      return a === undefined || b === undefined ? undefined : a / b
    }
    case "sqrt": {
      const v = ev(e.body)
      return v === undefined ? undefined : Math.sqrt(v)
    }
    case "paren":
      return ev(e.body)
    case "cases":
      return undefined
    case "term": {
      const given = inputs[e.term.id]
      if (typeof given === "number") return given
      if (given !== undefined) return undefined
      const name = forms[e.term.id]
      return ev(name && e.term.forms?.[name] ? e.term.forms[name]!.expr : e.term.body)
    }
  }
}

/** Every term's value: the inputs, plus whatever the formula can compute from them (the error, the bonus, the
 *  result). Inputs win -- a lab can always supply a term the formula could compute. */
export function computeValues(
  f: Formula,
  inputs: Record<string, BoundValue | undefined>,
  forms: Record<string, string> = {},
): Record<string, BoundValue | undefined> {
  const out: Record<string, BoundValue | undefined> = { ...inputs }
  for (const t of termsOfFormula(f)) {
    if (out[t.id] !== undefined) continue
    const v = evaluate({ k: "term", term: t }, inputs, forms)
    if (v !== undefined && Number.isFinite(v)) out[t.id] = v
  }
  if (f.result && out[f.result] === undefined && f.worked) {
    const v = evaluate(f.worked, inputs, forms)
    if (v !== undefined && Number.isFinite(v)) out[f.result] = v
  }
  return out
}

/** The formula checking the algorithm: each expected value (what the real code reported) against what the formula
 *  computed from the same inputs. Returns the mismatches, e.g. `["q-new: formula 0.52, reported 0.51"]`. */
export function checkValues(computed: Record<string, BoundValue | undefined>, expected: Record<string, number>, tolerance = 1e-9): string[] {
  const out: string[] = []
  for (const [id, want] of Object.entries(expected)) {
    const got = computed[id]
    if (typeof got !== "number") continue
    if (Math.abs(got - want) > tolerance * Math.max(1, Math.abs(want))) out.push(`${id}: formula ${got}, reported ${want}`)
  }
  return out
}

// --- The whole formula -----------------------------------------------------------------------------------------------

/** A whole formula: symbolic, and -- when `worked` is true and it has a worked instance -- a second row with the
 *  values bound, aligned on the relation. The worked row's cells are wrapped in `\htmlClass{math-worked}` so the page
 *  can set them back a step. The result is "≈" when a number shown was rounded, so the row still reads true. */
export function compileFormula(f: Formula, view: FormulaView, worked: boolean): string {
  const symbolic = { ...view, bind: false }
  const rel = f.rel ?? "="
  if (!worked || !f.worked) return f.lhs ? `${compile(f.lhs, symbolic)} ${rel} ${compile(f.body, symbolic)}` : compile(f.body, symbolic)

  const ctx: Ctx = { view: { ...view, bind: true, atoms: f.atoms }, operand: false, rounded: { any: false } }
  let rhs = compileIn(f.worked, ctx)
  if (f.result) {
    const def = termsOfFormula(f).find((t) => t.id === f.result)
    const v = view.values?.[f.result]
    if (v !== undefined && def) {
      const shown = formatValue(v, def, { ...ctx, operand: false })
      // "≈" only for a number: a choice (argmax → "straight") is exact however rounded the values it chose between.
      rhs += ` ${ctx.rounded.any && typeof v === "number" ? "\\approx" : "="} \\htmlData{term=${def.id}}{${shown}}`
    }
  }
  const cls = (t: string) => `\\htmlClass{math-worked}{${t}}`
  const top = f.lhs ? `${compile(f.lhs, symbolic)} &${rel} ${compile(f.body, symbolic)}` : `& ${compile(f.body, symbolic)}`
  // The worked row takes no width in the layout (`\mathrlap`): the block is sized and centred by the symbolic row alone,
  // which never changes, so a new value can't move the formula -- the numbers only extend to the right of the relation.
  const lhs = f.workedLhs ? cls(compileIn(f.workedLhs, { ...ctx, operand: false })) : ""
  return `\\begin{aligned} ${top} \\\\[0.35em] ${lhs} &\\mathrlap{${cls(`= ${rhs}`)}} \\end{aligned}`
}

// --- Walking the tree ------------------------------------------------------------------------------------------------

function childrenOf(e: Expr): Expr[] {
  switch (e.k) {
    case "seq":
    case "mul":
      return e.parts
    case "ops":
      return [e.first, ...e.rest.map((r) => r.e)]
    case "fn":
      return [e.arg]
    case "pow":
      return [e.base, e.exp]
    case "frac":
      return [e.num, e.den]
    case "sqrt":
    case "paren":
      return [e.body]
    case "cases":
      return e.rows.flatMap((r) => [r.value, r.when])
    case "term":
      return [e.term.body, ...Object.values(e.term.forms ?? {}).map((f) => f.expr)]
    default:
      return []
  }
}

function hasTerms(e: Expr): boolean {
  return e.k === "term" || childrenOf(e).some(hasTerms)
}

/** Every term in an expression, in reading order (the first occurrence of each id), including those inside forms. */
export function termsOf(e: Expr, out: TermDef[] = []): TermDef[] {
  if (e.k === "term" && !out.some((t) => t.id === e.term.id)) out.push(e.term)
  for (const child of childrenOf(e)) termsOf(child, out)
  return out
}

/** Every term of a formula -- both sides, and the worked row's -- in reading order. */
export function termsOfFormula(f: Formula): TermDef[] {
  const out: TermDef[] = []
  for (const part of partsOf(f)) termsOf(part, out)
  return out
}

// Formulas as expression trees (docs/design/0012). A formula is written with the builders below; `compile` turns it
// into LaTeX for KaTeX, wrapping every *term* -- a symbol or a named sub-expression -- in `\htmlData{term=<id>}`, so
// the rendered formula is addressable: hover, focus, colour and link any term to the rest of the page.
//
// Two things make a formula more than a string:
// - **forms**: a term can be shown several ways (δ, or the bracket it abbreviates; 1/n, or α). The page picks a
//   default, the reader can switch where the term is `expandable`.
// - **values**: bind a term id to a number and it renders as that number, still carrying its term id -- so a worked
//   line with real values links to everything the symbolic one does.
// Compiling is a pure function of (formula, view), so the server and the client produce the same markup.

export type Expr =
  | { k: "tex"; tex: string }
  | { k: "seq"; parts: Expr[] }
  | { k: "mul"; parts: Expr[] }
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
 *  could be negative) to the width of the largest value it can take -- so a worked line doesn't shift when 9.50 becomes
 *  10.00 or 0.40 becomes −0.40. TeX's digits are all one width, so the padding is exact. */
export interface NumFormat {
  decimals?: number
  /** The largest magnitude it can take: sets how many integer digits to reserve. */
  max?: number
  /** It can be negative: reserve a minus sign. */
  signed?: boolean
}

type Part = Expr | string

const toExpr = (p: Part): Expr => (typeof p === "string" ? { k: "tex", tex: p } : p)

export const tex = (s: string): Expr => ({ k: "tex", tex: s })
export const seq = (...parts: Part[]): Expr => ({ k: "seq", parts: parts.map(toExpr) })
/** A product: juxtaposed as symbols (γ max Q), with a · once values are bound (0.9 · 10.00). */
export const mul = (...parts: Part[]): Expr => ({ k: "mul", parts: parts.map(toExpr) })
export const frac = (num: Part, den: Part, size: "t" | "d" | "" = ""): Expr => ({ k: "frac", num: toExpr(num), den: toExpr(den), size })
export const sqrt = (body: Part): Expr => ({ k: "sqrt", body: toExpr(body) })
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
  /** The term whose bound value ends the worked row (`… = 0.52`). */
  result?: string
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
  /** Bind values (the worked / numbers view) or not (symbols). */
  bind?: boolean
  /** Number formats by term id, for this instance -- the lab knows ranges the formula can't (this game's budget, its
   *  largest payout). Merged over each term's own `num`. */
  formats?: Record<string, NumFormat>
}

const warned = new Set<string>()

/** A value as plain text, in its term's format (what a card shows). */
export function formatPlain(v: BoundValue, def: TermDef, view: FormulaView = {}): string {
  if (typeof v === "string") return v
  if (def.format) return def.format(v)
  const spec = { ...def.num, ...view.formats?.[def.id] }
  const decimals = spec.decimals ?? (Number.isInteger(v) ? 0 : 2)
  return (v < 0 ? "−" : "") + Math.abs(v).toFixed(decimals)
}

/** A value as LaTeX, padded to its format's full width (see `NumFormat`). */
function formatValue(v: BoundValue, def: TermDef, view: FormulaView): string {
  if (typeof v === "string") return v
  if (def.format) return def.format(v)
  const spec = { ...def.num, ...view.formats?.[def.id] }
  const decimals = spec.decimals ?? (Number.isInteger(v) ? 0 : 2)
  const digits = Math.abs(v).toFixed(decimals)
  let pad = ""
  if (spec.max !== undefined) {
    const room = String(Math.floor(Math.abs(spec.max))).length
    const used = digits.split(".")[0]!.length
    if (used > room && import.meta.dev && !warned.has(def.id)) {
      warned.add(def.id)
      console.warn(`formula term "${def.id}": ${v} is outside its declared range (max ${spec.max}) -- the formula will shift`)
    }
    if (room > used) pad = `\\phantom{${"0".repeat(room - used)}}`
  }
  // `{-}`: an ordinary minus (a sign, not a binary operator), so it and its invisible stand-in are the same width.
  const sign = v < 0 ? "{-}" : spec.signed ? "\\phantom{{-}}" : ""
  return `${pad}${sign}${digits}`
}

/** The LaTeX for an expression under a view. Every term is `\htmlData{term=id}{…}`. */
export function compile(e: Expr, view: FormulaView = {}): string {
  const c = (x: Expr) => compile(x, view)
  switch (e.k) {
    case "tex":
      return e.tex
    case "seq":
      return e.parts.map(c).join(" ")
    case "mul":
      return e.parts.map(c).join(view.bind ? " \\cdot " : " \\, ")
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
      const formName = view.forms?.[def.id] && def.forms?.[view.forms[def.id]!] ? view.forms[def.id]! : "default"
      const shown = formName === "default" ? def.body : def.forms![formName]!.expr
      // A bound term shows its value -- unless what it shows is built from other terms (1/n, r − Q): then the value
      // goes one level down, into those terms, and the structure stays readable (1/5, 1 − 0.40).
      const bound = view.bind ? view.values?.[def.id] : undefined
      const inner = bound !== undefined && !hasTerms(shown) ? formatValue(bound, def, view) : c(shown)
      return `\\htmlData{term=${def.id}}{${inner}}`
    }
  }
}

/** A whole formula: symbolic, and -- when `worked` is true and it has a worked instance -- a second row with the
 *  values bound, aligned on the relation. The worked row's cells are wrapped in `\htmlClass{math-worked}` so the page
 *  can set them back a step. */
export function compileFormula(f: Formula, view: FormulaView, worked: boolean): string {
  const symbolic = { ...view, bind: false }
  const rel = f.rel ?? "="
  if (!worked || !f.worked) return f.lhs ? `${compile(f.lhs, symbolic)} ${rel} ${compile(f.body, symbolic)}` : compile(f.body, symbolic)

  const bound = { ...view, bind: true }
  let rhs = compile(f.worked, bound)
  if (f.result) {
    const def = termsOfFormula(f).find((t) => t.id === f.result)
    const v = view.values?.[f.result]
    if (v !== undefined && def) rhs += ` = \\htmlData{term=${def.id}}{${formatValue(v, def, bound)}}`
  }
  const cls = (tex: string) => `\\htmlClass{math-worked}{${tex}}`
  const top = f.lhs ? `${compile(f.lhs, symbolic)} &${rel} ${compile(f.body, symbolic)}` : `& ${compile(f.body, symbolic)}`
  // The worked row takes no width in the layout (`\mathrlap`): the block is sized and centred by the symbolic row alone,
  // which never changes, so a new value can't move the formula -- the numbers only extend to the right of the relation.
  const bottom = `${f.workedLhs ? cls(compile(f.workedLhs, bound)) : ""} &\\mathrlap{${cls(`= ${rhs}`)}}`
  return `\\begin{aligned} ${top} \\\\[0.35em] ${bottom} \\end{aligned}`
}

/** Every term of a formula -- both sides, and the worked row's -- in reading order. */
export function termsOfFormula(f: Formula): TermDef[] {
  const out: TermDef[] = []
  for (const part of partsOf(f)) termsOf(part, out)
  return out
}

function hasTerms(e: Expr): boolean {
  switch (e.k) {
    case "term":
      return true
    case "seq":
    case "mul":
      return e.parts.some(hasTerms)
    case "frac":
      return hasTerms(e.num) || hasTerms(e.den)
    case "sqrt":
    case "paren":
      return hasTerms(e.body)
    case "cases":
      return e.rows.some((r) => hasTerms(r.value) || hasTerms(r.when))
    default:
      return false
  }
}

/** Every term in an expression, in reading order (the first occurrence of each id), including those inside forms. */
export function termsOf(e: Expr, out: TermDef[] = []): TermDef[] {
  const add = (d: TermDef) => {
    if (!out.some((t) => t.id === d.id)) out.push(d)
  }
  switch (e.k) {
    case "seq":
    case "mul":
      e.parts.forEach((p) => termsOf(p, out))
      break
    case "frac":
      termsOf(e.num, out)
      termsOf(e.den, out)
      break
    case "sqrt":
    case "paren":
      termsOf(e.body, out)
      break
    case "cases":
      e.rows.forEach((r) => {
        termsOf(r.value, out)
        termsOf(r.when, out)
      })
      break
    case "term":
      add(e.term)
      termsOf(e.term.body, out)
      Object.values(e.term.forms ?? {}).forEach((f) => termsOf(f.expr, out))
      break
  }
  return out
}

export function findTerm(e: Expr, id: string): TermDef | undefined {
  return termsOf(e).find((t) => t.id === id)
}

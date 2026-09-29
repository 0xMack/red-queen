# 0012 — Interactive maths notation: formulas tied to the demos they describe

Status: **Proposed** (2026-09-29). Nothing built yet. The plan is a renderer spike, then one pilot chapter iterated
until it's right, then the rest.
Relates to: [0005](0005-frontend-and-api-contracts.md) (one frontend, reusable components),
[0011](0011-multi-armed-bandits.md) (the pilot chapter's game and demos), the explainer registry
(`app/data/explainers/`, `types/explain.ts`).

## Context

The Learn chapters explain each technique in prose, code (now in pseudocode, Python and Rust) and live demos. The
maths behind them, like `Q(a) ← Q(a) + α (r − Q(a))`, appears at most as inline text. The formula is the most compact
statement of what the demo is doing, and the least readable part of any textbook. The aim is to *ground* it: every
symbol is something you can point at on screen, and the demo is something you can point at in the formula.

Decided with the user (2026-09-29):

| Question | Decision |
|---|---|
| Rendering | Whatever gives the most control over the content at run time while still looking like typeset LaTeX |
| Linking | **Two-way wherever possible.** Hovering a term highlights the control and visual it drives; touching a control or visual highlights the term |
| Values | Highlighting the terms is the baseline. Showing the actual numbers too is worth exploring: e.g. the symbolic formula with a worked instance underneath, using the demo's current values. To be settled by trying it |
| Term definitions | Shared or per formula, either is fine, but look into **expanding terms or changing their form** (δ ↔ `r + γ max Q′ − Q`, 1/n ↔ α) |
| Rollout | One representative chapter that exercises the whole framework, iterated until satisfying, **then** the rest |

## What we want from a formula

1. **Typeset quality.** It should look like a LaTeX formula: real fractions, sums, sub- and superscripts, maths fonts.
2. **Addressable terms.** Every symbol or sub-expression that matters carries an id we can hover, style and focus.
3. **Cheap to change at run time.** Re-render when a value changes (every bandit pull), when the reader expands a
   term, or when they switch variants (sample average vs. constant step), with no visible cost.
4. **Server-rendered.** Chapters are server-rendered, so the formula must render in Node and hydrate without a
   mismatch (the same constraint the chapters already have for SVG maths; see the frontend README).
5. **Accessible.** Screen readers get MathML, and terms are reachable by keyboard.

## Renderer options

| Option | How terms get ids | Look | Server rendering | Run-time changes | Notes |
|---|---|---|---|---|---|
| **KaTeX** + HTML extension | `\htmlData{term=alpha}{\alpha}`, `\htmlClass{…}` → a `<span data-term>` around exactly that sub-expression | TeX fonts and spacing; the most "LaTeX-looking" | `renderToString` works in Node | Re-render the string (small formulas are fast) and swap it via `v-html` | Needs `trust` and a relaxed `strict` for the HTML extension, which is safe because we author every input. Outputs HTML plus hidden MathML for screen readers. Prior art: *equations-explained-colorfully* (Vue + KaTeX `\htmlClass`) builds almost exactly this interaction |
| **Temml** → MathML Core | `\class`, `\id`, `\data` (also behind `trust`) | Native browser maths rendering; good in Chromium/Firefox, a few gaps in Safari; depends on a maths font | Yes (a string) | Same as KaTeX | Lighter output (native `<math>`), but the look varies by browser and font |
| **MathJax** (v3/v4) | `\class`, `\cssId`, `\data` via the HTML extension | Excellent | Possible, but heavier to set up | Heaviest to re-typeset | The biggest dependency of the three for the same result |
| **Hand-written MathML as Vue components** | Each term is a component: fully reactive, events on each node | As MathML, above | Yes | Vue re-renders only what changed | Maximum control, but we'd be writing a small TeX layout engine's inputs by hand, formula by formula |
| **HTML/CSS by hand** | Any element | Poor (fractions, radicals, alignment) | Yes | Yes | Not a real option past `x = y + z` |

**Recommendation: author formulas as a small expression tree in TypeScript, compile it to LaTeX with a term id on
every node that has one, and render it with KaTeX.** The tree is where the flexibility lives, not the renderer:

- **Ids come from the tree**, so authors never write `\htmlData` by hand and never misspell a term.
- **Expanding a term or changing its form is a tree operation.** A node can carry alternative forms (`δ` ↔
  `r + γ maxₐ′ Q(s′,a′) − Q(s,a)`), and the reader picks which one to render.
- **A worked instance is the same tree with values bound.** Every number keeps its term's id, so hovering `0.1` in
  the worked line highlights α everywhere, exactly like hovering `α`.
- **The renderer is swappable.** The only thing that touches KaTeX is the compile-to-LaTeX step. If native MathML
  gets good enough, Temml can take LaTeX in the same dialect.

Phase 0 runs a short spike to confirm this before committing. Render the pilot's UCB and incremental-mean formulas
through KaTeX and through Temml, with ids on every term. Check server rendering and hydration in Nuxt, how long a
re-render takes, the added bundle and font weight, and how both look in Chromium, Firefox and Safari. KaTeX is
the default unless the spike says otherwise.

## The model

### Symbols: defined once, overridable locally

`app/data/math/symbols.ts` is a typed registry, like the explainer registry. Each symbol has:

- an id (`alpha`);
- its TeX (`\alpha`);
- a name ("step size");
- one line on what it means;
- its range and units, where they make sense;
- optionally, a link to a concept in the explainer registry.

A chapter can override the wording locally ("the learning rate" in the DQN chapter) without redefining the symbol.
One registry buys consistent notation across chapters (Q, V, G, γ, α mean one thing everywhere) and a generated
**notation page** (`/learn/notation`: every symbol, where it appears, and a link to each formula).

### Formulas: an expression tree

`app/data/math/<chapter>.ts` builds formulas with small helpers, for example:

```ts
const tdError = term("td-error", sub(add(sym("r"), mul(sym("gamma"), max("a'", q("s'", "a'")))), q("s", "a")), {
  compact: sym("delta"),            // the reader can collapse the bracket to δ, or expand δ back out
})
export const qUpdate = formula("q-update", assign(q("s", "a"), add(q("s", "a"), mul(sym("alpha"), tdError))), {
  variants: { sarsa: { replace: { "max-next": q("s'", "a'") } } },   // Q-learning ↔ SARSA: one node swapped
})
```

These are the node kinds:

- **Structural:** `sym`, `num`, `add`, `sub`, `mul`, `frac`, `pow`, `sqrt`, `sum`, `max`/`argmax`, `apply`, `assign`,
  `cases` (a piecewise function, like ε-greedy's choice rule), `group`.
- **Semantic:** `term(id, expr, forms?)`, a named sub-expression. It's what makes "the exploration bonus" or "the TD
  error" a single hoverable, expandable thing.

A formula compiles to LaTeX given a **view**:

- which form each term is shown in;
- which variant of the formula;
- whether to bind values.

Compiling is a pure function, so the server and client produce the same string, and there are no hydration
mismatches as long as bound values are rounded the same way on both sides.

### Terms as parts of the page: a shared focus

A `TermScope` is a provided and injected focus store around one figure or section. It holds the hovered or focused
terms and one pinned term (click to lock, as in the prior art). Everything inside a scope takes part:

- **`<MathFormula :formula :view :values>`** renders the formula and delegates hover, focus and click on
  `[data-term]` spans.
- **`<Term id="alpha">α</Term>`** puts a term inline in prose, typeset with the same KaTeX.
- **A `term` prop on the lab controls** (`UiRange`, `UiSelect`, `LabControls`). Hovering, focusing or dragging the
  control focuses its term; focusing the term rings the control.
- **A `v-term="'alpha'"` directive** marks any element (an SVG bar in `BanditTable`, a curve in `BeliefCurves`, a cell
  in `QTableGrid`) so visuals take part without knowing about formulas.

Highlighting is one CSS class plus a per-term colour custom property. Within a scope, terms get colours from the
palette tokens in the order they first appear, like the prior art, so the colour of α in the formula is the colour of
its slider's accent and of its region in the chart. No hex values, per the frontend rules.

### Hover cards

A term's card shows:

- its name and meaning;
- the **current value** when the scope binds one;
- its range and what changing it does;
- the other formulas it appears in;
- the action **Expand** or **Collapse** when the term has forms.

This is the explainer card's presentation (`InfoTip` popover, "More" side panel) with a new explainer kind,
`symbol`. That way `?explain=symbol:gamma` works, and a term is authored once, like every other explained thing.

### Showing values

This is to be settled by trying three presentations in the pilot and keeping what reads best:

1. **Symbolic plus a worked line.** The formula, and under it the same formula with the demo's current values, e.g.
   `Q₃ ← 0.40 + ⅕ (1 − 0.40) = 0.52` after the last pull. The two lines are aligned term for term, colour for colour.
2. **Swap in place.** A toggle (or holding a key) turns symbols into numbers in the same formula.
3. **Value chips.** The formula stays symbolic, and small value tags sit under the bound terms.

## The pilot: Multi-Armed Bandits

This chapter exercises every part of the framework, and its demos make it the cheapest place to do so:

- **Every step is discrete and inspectable.** The strategies run in WASM on the main thread (`useBanditRun`), one
  pull at a time, so after each pull there is an exact update to show as a worked line.
- **Its formulas cover every node kind and feature:**

| Section | Formula | What it exercises |
|---|---|---|
| Keeping score | `Qₙ₊₁ = Qₙ + 1/n (r − Qₙ)` | Expanding a term: `Qₙ` ↔ the sample average `(1/n) Σ rᵢ`. Worked line on every pull; the `BanditTable` cell and `BanditTape` entry linked |
| When the world changes | the same, `1/n` → `α` | A variant switch (sample average vs. constant step), with an α slider tied to its term |
| Exploring on purpose | ε-greedy as `cases`; UCB `Q(a) + c·√(ln t / N(a))` | Piecewise formulas. The **bonus term** is linked to the uncertainty whiskers in the belief view; ε and c on sliders |
| Thompson sampling | `θₐ ~ Beta(1 + wins, 1 + losses)`, `a = argmax θₐ` | Terms linked to the `BeliefCurves` curves |
| Two lamps | `Q(s, a)`: the row is picked by what's observed | Linking a subscript (`s`) to the table's rows |
| The detour | `Q(s,a) ← Q(s,a) + α (r + γ maxₐ′ Q(s′,a′) − Q(s,a))` | The full Bellman form, a γ dial tied to its term, and δ collapse/expand. This line leads straight into the Q-learning chapter |

The demos need small additions for this:

- **Knobs:** the lab currently exposes a strategy and γ. The pilot adds ε, α and c sliders, passed through the
  existing `params` string.
- **The last update:** `useBanditRun` gains a `lastUpdate` (row, arm, reward, value before and after, n) so the
  worked line reads the numbers from the real Rust update rather than recomputing them.

**Q-learning goes second.** Its lab runs in a worker at speed, so it tests the other case: values sampled from a
running process, with pause and step. After that, the rest follow.

## Phases

0. **Spike (short).** KaTeX vs. Temml on two pilot formulas, checking server rendering and hydration, re-render
   time, bundle and font weight, and the look in three browsers. Then decide.
1. **Core:** the expression tree and its compiler; `MathFormula` and `Term`; `TermScope` and `v-term`; `term` props
   on the lab controls; the symbol registry and the `symbol` explainer kind. There is no frontend test suite, so the
   compiler gets a small Node check (like `check-rl-determinism.mjs`) that every formula compiles and every term id
   exists.
2. **Pilot:** Multi-Armed Bandits, section by section, iterated with the user: the value presentations,
   expand/collapse, colours, and mobile (tap to pin replaces hover).
3. **Q-learning,** then the remaining chapters. Their formulas are fewer or simpler, but they reuse everything.
4. **Notation page** generated from the registry.

## Open questions

- **Colours:** assign per scope in order of appearance (consistent within a figure), or give each symbol a fixed
  colour site-wide (α is always gold)? Per-scope is the prior art's choice. Fixed colours would run out of distinct
  tokens quickly.
- **Expanding:** does an expanded term animate open in place, or re-render with a highlight on what changed?
- **Where terms can be touched from:** should prose terms (`<Term>`) always take part in the nearest scope, or only
  when the formula is on screen?
- **Without JavaScript and in print:** the server-rendered KaTeX output is already a complete formula. Nothing
  interactive should be needed to read it.

# 0012 — Interactive maths notation: formulas tied to the demos they describe

Status: **Pilot implemented** (2026-09-29). Phase 0 (the renderer spike) chose KaTeX. Phase 1 (the framework) and
phase 2 (the Multi-Armed Bandits pilot) are built, pending review and iteration with the user; the other chapters wait
on that. See "Implementation notes".
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

## Implementation notes

### Phase 0: the renderer spike (2026-09-29)

The same incremental-mean and UCB formulas went through KaTeX 0.18 (`\htmlData`) and Temml 0.13 (`\data` to MathML
Core), each term carrying a `data-term` id. Both were rendered during setup, so on the server too, with one hover
state lighting every element with the hovered id.

| | KaTeX | Temml |
|---|---|---|
| Term hooks | `<span data-term>` around exactly the sub-expression | `data-term` on the MathML node (`msub`, `mrow`, `mi`, …) |
| Linking across both renderers | worked | worked |
| Look (Chromium, Windows) | TeX spacing and fonts; `arg max` limits, fractions and radicals as in LaTeX | cramped with the system maths font (Cambria Math); needs a hosted Latin Modern or STIX woff2 (several hundred KB, not in the npm package) for a consistent look |
| Re-render (client, a pair of formulas) | ~0.23 ms | ~0.07 ms |
| JS, gzipped | ~76 KB (its own chunk, only on pages with formulas: 83 KB built) | ~50 KB, plus the font |
| Server render and hydration | no warnings | no warnings |

**Decision: KaTeX.** Temml is faster, but both are far below a frame. KaTeX's output is HTML and CSS with its own
fonts, so it renders the same in every browser. Temml's look depends on each browser's MathML support and on a maths
font we would have to host; that variance is exactly what this project would have to test for. KaTeX's fonts load on
demand (a chapter fetches the few faces it uses). Firefox and Safari were not available to test.

### Phase 1: the framework

- `utils/math/expr.ts` is the expression tree:
  - Builders: `seq`, `mul`, `frac`, `sqrt`, `paren`/`bracket`, `cases`, `term`.
  - `compile(expr, view)` produces LaTeX; `compileWorked` produces the worked line, ending in `= result`.
  - A `mul` is juxtaposed as symbols and gets a `·` once values are bound (`0.9 · 10.00`, not `0.9 10.00`).
  - A bound term shows its value only if what it shows has no nested terms. Otherwise the value goes one level down,
    so `1/n` becomes `1/5` and `r − Q_n` becomes `1 − 0.40`, instead of the structure collapsing to one number. The
    same rule makes an `α` form show `0.20`.
- `utils/math/katex.ts` renders memoised, with `trust` limited to `\htmlData`.
- `data/math/symbols.ts` is the notation registry. Each symbol is also a `symbol:<id>` explainer, so the side panel
  works for it.
- `useTermScope` (and the `MathScope` component) holds one figure's shared state:
  - Colours are assigned per scope in reading order, as decided.
  - `active` is the pinned term, else the hovered one.
  - `lit` is the active term plus the terms it's built from: focusing the bonus lights `c`, `t` and `N(a)`.
  - `flash` lights a term for a moment, without dimming anything, when a lab points at what just happened.
  - `target(ids)` is a spread of bindings that makes any HTML or SVG element take part.
- `MathFormula` renders a formula:
  - Hover, focus and click (click pins, Esc or clicking outside releases) are handled by delegation over the rendered
    terms.
  - The card shows name, meaning, the current value, range and forms, and "More" opens the panel. It sits below the
    whole figure, so the worked line stays visible.
  - Values: settled with the user after the first pass. There's no toggle; both are always on: the symbolic formula,
    and under it the same formula with the numbers, in one KaTeX `aligned` block aligned on the relation, like a line
    of working (`Q_{n+1} ← Q_n + 1/n (r − Q_n)` over `= 0.60 + 1/6 · (1 − 0.60) = 0.67`). A formula has an explicit
    `lhs`/`rel`/`body` for this; `workedLhs` gives the worked row its own left side (UCB: `score(C) =`). The worked
    row is set back a step (`\htmlClass{math-worked}`). Value chips were dropped.
  - At rest a formula is plain ink; hovering it underlines its smallest live terms in their colours.
  - It shows no worked line until the formula's result has a value, so it's never half-filled.
- `MathTerm` is a term inside prose.

### Phase 2: the pilot, Multi-Armed Bandits

- `useBanditRun.lastUpdate` reads each pull's effect from the Rust strategy's beliefs just before and after: the
  cell's old and new value, n, the row the pull was made in and the row it led to, the next row's best value, the
  beliefs the choice was made from, and whether the choice was greedy.
- `BanditLab`'s `math` prop is one of incremental, constant-step, epsilon, ucb or bellman. It shows the formula, a
  narration line and the knob sliders (ε, α, c; γ stays a select), each linked to its term.
  - Knobs restart the game 350 ms after the slider settles.
  - While a term is pinned, the game holds.
  - Each ε-greedy pull flashes the branch it took.
- `BanditTable` and `BanditTape` take part through the term scope:
  - estimates are `estimate`, counts are `count`, uncertainties are `bonus`;
  - the updated cell is also `q-new` and `n`;
  - the last pull on the tape is `reward`.
- Each worked line was checked against the real update pull by pull, with nothing recomputed that could be read:
  - incremental: `0.50 + ⅓·(1 − 0.50) = 0.67`;
  - constant step: `0.20 + 0.20·(0 − 0.20) = 0.16`;
  - UCB: `0.89 + 1.41·√(ln 27 / 9) = 1.74`, equal to the strategy's own reported bonus;
  - Bellman: `10 + 0.5·[0 + 0.9·10 − 10] = 9.5`.
- The chapter now has:
  - the incremental formula in the prose, with `MathTerm`s and the "as an average" expansion;
  - worked formulas in the greedy and drifting labs;
  - new ε and UCB labs;
  - Thompson's formula (symbolic only);
  - the detour's formula, prose and γ lab in one `MathScope`, so the prose formula fills in from the lab below it.
- `/dev/math` (unlinked) renders every registered formula and, with `?lab=<math>`, a lab at the top of the page. It's
  where a new formula gets checked (KaTeX throws in development), and it's usable for screenshots, since the Browser
  pane doesn't draw scrolled content while hidden.

### Open, for the iteration with the user

- **Keyboard stops:** every term is a tab stop. Nested terms make that a lot of stops.
- **Thompson and the lamp section:** the `wins`/`losses` terms aren't linked to `BeliefCurves` yet, and `Q(s, a)`'s
  `s` isn't linked to the table's rows yet.

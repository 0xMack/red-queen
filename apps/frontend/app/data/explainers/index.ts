import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"
import type { ExplainConcept, ExplainKind, ExplainLink, ExplainRef, ExplainResults, Fact, ResolvedExplainer, ResultRow } from "~/types/explain"
import { learnChapters } from "~/data/learnChapters"
import { BANDIT_RESULTS } from "~/data/banditResults"
import { ALGORITHMS } from "./algorithms"
import { METRICS } from "./metrics"
import { REPRESENTATIONS } from "./representations"
import { SCENARIOS } from "./scenarios"
import { symbolConcepts } from "~/data/math/symbols"

// The explainer registry and resolver. A concept (what ε-greedy is) is authored once; `resolveExplainer` turns a ref
// into what the card and the panel draw, adding -- when the page has leaderboard records (`ExplainContext`) -- the
// instance layer: this entrant's settings, model, provenance and results, or who wins on a scenario, or who plays
// under a representation. Everything the UI shows goes through here, so every kind renders the same way.

export interface ExplainContext {
  game: string | null
  entries: readonly EvaluationRecord[]
  interfacesById: Readonly<Record<string, InterfaceInfo>>
}

export const EMPTY_CONTEXT: ExplainContext = { game: null, entries: [], interfacesById: {} }

const CONCEPTS = new Map<string, ExplainConcept>([...ALGORITHMS, ...SCENARIOS, ...REPRESENTATIONS, ...METRICS, ...symbolConcepts()].map((c) => [`${c.kind}:${c.id}`, c]))

export function parseRef(value: string | null | undefined): ExplainRef | null {
  if (!value) return null
  const i = value.indexOf(":")
  if (i < 0) return null
  const kind = value.slice(0, i) as ExplainKind
  if (!["algorithm", "entrant", "scenario", "representation", "metric", "symbol"].includes(kind)) return null
  return { kind, id: value.slice(i + 1) }
}

export const formatRef = (ref: ExplainRef) => `${ref.kind}:${ref.id}`

export function concept(ref: string): ExplainConcept | null {
  return CONCEPTS.get(ref) ?? null
}

// --- Links into the Learn chapters ---------------------------------------------------------------------------------

/** A chapter, or a section of one (anchors are slugify(heading), learn.vue's own ids). An unknown section falls back to
 *  the chapter, loudly in development -- headings are kept in sync with `learnChapters` by hand. */
export function chapterLink(slug: string, section?: string, label?: string): ExplainLink | null {
  const chapter = learnChapters.find((c) => c.slug === slug)
  if (!chapter) return null
  if (section && chapter.sections?.includes(section)) {
    return { kind: "section", label: label ?? section, to: `${chapter.path}#${slugify(section)}` }
  }
  if (section && import.meta.dev) console.warn(`explainers: no section "${section}" in chapter ${slug}`)
  return { kind: "chapter", label: label ?? chapter.title, to: chapter.path }
}

function conceptLinks(c: ExplainConcept): ExplainLink[] {
  const links = (c.chapter ?? []).map((ch) => chapterLink(ch.slug, ch.section, ch.label)).filter((l): l is ExplainLink => !!l)
  return [...links, ...(c.links ?? [])]
}

// --- Which concept an entrant is -----------------------------------------------------------------------------------

const BANDIT_ALGORITHM: Record<string, string> = {
  random: "random",
  greedy: "greedy",
  epsilon_greedy: "epsilon-greedy",
  optimistic: "optimistic",
  ucb1: "ucb",
  thompson: "thompson",
  gradient: "gradient-bandit",
  q_table: "q-table-bandit",
}

const BASELINE_ALGORITHM: Record<string, string> = { random: "random", greedy: "snake-greedy", "first-legal": "first-legal" }

/** The algorithm concept an entrant is an instance of, or null for one nothing describes yet. */
export function algorithmIdFor(r: EvaluationRecord): string | null {
  if (r.metrics.bandit) return BANDIT_ALGORITHM[r.metrics.bandit.strategy] ?? null
  const baseline = baselineName(r)
  if (baseline) return baseline.startsWith("material") ? "material-search" : (BASELINE_ALGORITHM[baseline] ?? null)
  const algorithm = entrantShape(r)?.algorithm.toLowerCase() ?? ""
  if (algorithm === "neat") return "neat"
  if (algorithm === "neuroevolution") return "neuroevolution"
  if (algorithm === "dqn") return "dqn"
  if (algorithm === "ppo") return "ppo"
  if (algorithm === "reinforce" || algorithm === "a2c") return "policy-gradient"
  if (algorithm === "q-learning" || algorithm === "sarsa") return "q-learning"
  if (algorithm.startsWith("td(")) return "td-lambda"
  if (algorithm.startsWith("td-leaf")) return "td-leaf"
  return null
}

/** The observer part of an interface id: "snake/egocentric.v2+relative3.v1" -> "egocentric.v2". */
export const observerName = (interfaceId: string) => interfaceId.split("/")[1]?.split("+")[0] ?? interfaceId

/** The selection concept a champion's shape names ("lexicase", "tournament(k=4)"), if any. */
function selectionRef(selection: string | null | undefined): string | null {
  if (!selection) return null
  if (selection.startsWith("lexicase")) return "algorithm:lexicase"
  if (selection.startsWith("tournament")) return "algorithm:tournament"
  if (selection.startsWith("speciation")) return "algorithm:neat"
  return null
}

const SCORE_METRIC: Record<string, string> = { snake: "metric:held-out", checkers: "metric:elo", bandit: "metric:skill" }

// --- Label for a ref (the related chips) ---------------------------------------------------------------------------

export function refLabel(ref: string, ctx: ExplainContext = EMPTY_CONTEXT): string {
  const parsed = parseRef(ref)
  if (parsed?.kind === "entrant") {
    const r = ctx.entries.find((e) => e.entrant_id === parsed.id)
    return r ? entrantShortLabel(r) : parsed.id
  }
  return concept(ref)?.title ?? parsed?.id ?? ref
}

// --- Results -------------------------------------------------------------------------------------------------------

/** One column per scenario, as the strategy x scenario matrix has them (a contextual one blind and seeing the lamp). */
const SCENARIO_COLUMNS = BANDIT_SCENARIOS.flatMap((s) =>
  s.sequential
    ? [{ key: s.id, scenario: s.id, label: s.title, sub: "sees the room" }]
    : s.contexts > 1
      ? [
          { key: s.id, scenario: s.id, label: s.title, sub: "blind" },
          { key: `${s.id}:lamp.v1`, scenario: s.id, label: s.title, sub: "sees the lamp" },
        ]
      : [{ key: s.id, scenario: s.id, label: s.title, sub: `${s.arms} × ${s.budget}` }],
)

/** Skill on every scenario, from a record's scores (or a cited row), marking where it is best and where it fails. */
function skillAcrossScenarios(skill: Record<string, number>, bestPerColumn: Record<string, number>): ExplainResults {
  const rows: ResultRow[] = SCENARIO_COLUMNS.filter((c) => skill[c.key] !== undefined).map((c) => {
    const v = skill[c.key]!
    const best = bestPerColumn[c.key]
    return {
      label: c.label,
      sub: c.sub,
      value: v,
      display: v.toFixed(1),
      ref: `scenario:${c.scenario}`,
      tone: best !== undefined && v >= best - 1e-9 ? "best" : v < 5 ? "bad" : undefined,
    }
  })
  return {
    title: "Skill on every scenario",
    caption: "0 = no better than random, 100 = the best machine every pull. Gold: the best strategy there. Red: no better than random.",
    max: 100,
    rows,
  }
}

function banditSkillRows(ctx: ExplainContext): { id: string; label: string; skill: Record<string, number> }[] {
  const live = ctx.entries.filter((r) => r.metrics.bandit)
  if (live.length) {
    return live.map((r) => ({
      id: r.entrant_id,
      label: entrantShortLabel(r),
      skill: Object.fromEntries(Object.entries(r.metrics.bandit!.scenarios).map(([k, v]) => [k, v.skill])),
    }))
  }
  return BANDIT_RESULTS.map((r) => ({ id: r.id, label: r.label, skill: r.skill }))
}

function bestPerColumn(rows: { skill: Record<string, number> }[]): Record<string, number> {
  const best: Record<string, number> = {}
  for (const r of rows) for (const [k, v] of Object.entries(r.skill)) best[k] = Math.max(best[k] ?? -Infinity, v)
  return best
}

/** Ranked entrants (a subset of the leaderboard), for "on this leaderboard" lists. */
function entrantRows(ctx: ExplainContext, pick: (r: EvaluationRecord) => boolean, selfId?: string, limit = 10): ResultRow[] {
  const ranked = ctx.entries.map((r, i) => ({ r, rank: i + 1 })).filter(({ r }) => pick(r))
  return ranked.slice(0, limit).map(({ r, rank }) => ({
    label: entrantShortLabel(r),
    sub: `#${rank} · ${entrantDetail(r, { runId: false })}`,
    value: r.metrics.quality.mean,
    display: formatScore(ctx.game, r.metrics.quality.mean),
    ref: `entrant:${r.entrant_id}`,
    tone: r.entrant_id === selfId ? "self" : undefined,
  }))
}

function formatScore(game: string | null, v: number): string {
  if (game === "checkers") return v.toFixed(0)
  if (game === "bandit") return v.toFixed(1)
  return v.toFixed(2)
}

function scoreMax(ctx: ExplainContext): number {
  if (ctx.game === "checkers") return Math.max(400, ...ctx.entries.map((r) => r.metrics.quality.mean))
  if (ctx.game === "bandit") return 100
  return Math.max(1, ...ctx.entries.map((r) => r.metrics.quality.mean))
}

// --- Resolving -----------------------------------------------------------------------------------------------------

function fromConcept(c: ExplainConcept, ref: ExplainRef): ResolvedExplainer {
  return {
    ref,
    kind: ref.kind,
    eyebrow: c.paradigm ? `${c.family} · ${c.paradigm}` : c.family,
    title: c.title,
    summary: c.summary,
    visual: c.visual,
    live: c.live,
    how: c.how,
    code: c.code,
    good: c.good,
    bad: c.bad,
    facts: c.facts ?? [],
    links: conceptLinks(c),
    related: (c.related ?? []).map((r) => ({ ref: r, label: refLabel(r) })),
  }
}

/** Everything the card and the panel show for `ref`, or null when it names nothing known (or an entrant this page
 *  has no record of). */
export function resolveExplainer(ref: ExplainRef, ctx: ExplainContext = EMPTY_CONTEXT): ResolvedExplainer | null {
  if (ref.kind === "entrant") return resolveEntrant(ref, ctx)
  const c = concept(formatRef(ref))
  if (!c) return null
  const out = fromConcept(c, ref)

  if (ref.kind === "algorithm") {
    const mine = (r: EvaluationRecord) => algorithmIdFor(r) === ref.id
    const rows = entrantRows(ctx, mine)
    if (rows.length) out.results = { title: "On this leaderboard", max: scoreMax(ctx), rows }
    else if (c.family.startsWith("Bandit")) {
      // The Learn pages have no records: the chapter's cited numbers, for this strategy's hand-set entrant.
      const cited = BANDIT_STRATEGIES.find((s) => BANDIT_ALGORITHM[s.strategy] === ref.id)
      const row = cited && BANDIT_RESULTS.find((r) => r.id === cited.id)
      if (row) out.results = { ...skillAcrossScenarios(row.skill, bestPerColumn(BANDIT_RESULTS)), title: `${row.label}, on every scenario` }
    }
  }

  if (ref.kind === "scenario") {
    const rows = banditSkillRows(ctx)
    const scenario = scenarioById(ref.id)
    const key = scenario.contexts > 1 && !scenario.sequential ? `${ref.id}:lamp.v1` : ref.id
    const ranked = rows
      .filter((r) => r.skill[key] !== undefined)
      .sort((a, b) => b.skill[key]! - a.skill[key]!)
      .map((r, i, all) => ({
        label: r.label,
        value: r.skill[key]!,
        display: r.skill[key]!.toFixed(1),
        ref: ctx.entries.length ? `entrant:${r.id}` : undefined,
        tone: i === 0 ? ("best" as const) : r.skill[key]! < 5 ? ("bad" as const) : i === all.length - 1 ? ("muted" as const) : undefined,
      }))
    if (ranked.length) {
      out.results = {
        title: key.endsWith(":lamp.v1") ? "Who wins here (seeing the lamp)" : "Who wins here",
        caption: `Skill on 500 held-out games of ${scenario.title}${ctx.entries.length ? "" : " (the chapter's cited run)"}.`,
        max: 100,
        rows: ranked,
      }
      const top = ranked[0]!
      out.facts = [...out.facts.slice(0, 3), { label: "best here", value: `${top.label} · ${top.display}` }]
    }
  }

  if (ref.kind === "representation") {
    const info = ctx.interfacesById[ref.id]
    if (info) {
      out.facts = [
        { label: "inputs", value: String(info.observer.size) },
        { label: "outputs", value: String(info.action.num_outputs) },
        { label: "level", value: `L${info.observer.level} ${info.observer.level_name}` },
      ]
      out.instance = {
        title: "What it measures",
        subtitle: info.observer.description,
        facts: [{ label: "action", value: info.action.id, hint: info.action.description }],
        chips: info.observer.size <= 40 ? info.observer.feature_names : [`${info.observer.feature_names[0]} … ${info.observer.feature_names.at(-1)}`],
      }
    }
    const rows = entrantRows(ctx, (r) => r.interface === ref.id)
    if (rows.length) out.results = { title: "Entrants that see the game this way", max: scoreMax(ctx), rows }
  }
  return out
}

function resolveEntrant(ref: ExplainRef, ctx: ExplainContext): ResolvedExplainer | null {
  const r = ctx.entries.find((e) => e.entrant_id === ref.id)
  if (!r) return null
  const algorithmId = algorithmIdFor(r)
  const c = algorithmId ? concept(`algorithm:${algorithmId}`) : null
  const rank = ctx.entries.indexOf(r) + 1
  const game = ctx.game ?? r.interface.split("/")[0] ?? null
  const shape = entrantShape(r)
  const info = ctx.interfacesById[r.interface]

  const base: ResolvedExplainer = c
    ? fromConcept(c, ref)
    : { ref, kind: "entrant", eyebrow: r.entrant_kind, title: r.label, summary: r.metrics.model.description, facts: [], links: [], related: [] }
  base.kind = "entrant"
  base.eyebrow = c ? `${c.title} · ${c.family}` : base.eyebrow
  base.title = entrantShortLabel(r)

  // --- This entrant: settings, model, provenance
  const facts: Fact[] = []
  const bandit = r.metrics.bandit
  if (bandit) {
    for (const [k, v] of Object.entries(bandit.params)) {
      const p = c?.params?.[k]
      facts.push({ label: p?.name ?? k, value: String(v), hint: p?.meaning })
    }
    if (!Object.keys(bandit.params).length) facts.push({ label: "settings", value: "defaults" })
  }
  if (shape && !bandit) {
    const size = modelSize(shape)
    if (size) facts.push({ label: "model", value: size })
    if (shape.selection) facts.push({ label: "selection", value: shape.selection })
  }
  if (r.metrics.model.search_depth && r.metrics.model.search_depth > 1) facts.push({ label: "search", value: `${r.metrics.model.search_depth} plies, alpha-beta` })
  if (!bandit) facts.push({ label: "sees", value: `${observerName(r.interface)}${info ? ` · ${info.observer.size} inputs` : ""}` })
  const t = r.metrics.training
  if (t.none) facts.push({ label: "training", value: bandit ? "none: learns as it plays" : "none" })
  else {
    const amount = t.episodes != null ? `${compactNumber(t.episodes)} episodes` : t.fitness_evaluations != null ? `${compactNumber(t.fitness_evaluations)} evaluations` : null
    const time = t.active_s != null ? formatDuration(t.active_s) : null
    if (amount || time) facts.push({ label: "training", value: [amount, time].filter(Boolean).join(" · ") + (t.measured ? "" : " (est.)") })
  }
  const variant = entrantVariant(r)
  if (variant) facts.push({ label: "plays as", value: `${variant} export`, hint: "a published package variant, scored separately in case it plays differently" })

  const notes: string[] = []
  if (r.metrics.model.note) notes.push(r.metrics.model.note)
  if (bandit && r.run_id) notes.push("Its settings weren't chosen by hand: evolution searched them over many training games (a separate set from these held-out ones).")

  base.instance = { title: r.entrant_kind === "baseline" ? "This baseline" : "This entrant", subtitle: r.metrics.model.description, facts, notes }

  const score = { label: game === "checkers" ? "Elo rating" : game === "bandit" ? "skill (classic)" : "held-out score", value: `${formatScore(game, r.metrics.quality.mean)} ± ${formatScore(game, r.metrics.quality.ci95)}` }
  base.facts = [{ label: "rank", value: `#${rank} of ${ctx.entries.length}` }, score, ...facts.slice(0, 2)]

  // --- Results
  if (bandit) {
    const rows = banditSkillRows(ctx)
    const self = rows.find((x) => x.id === r.entrant_id)
    if (self) base.results = skillAcrossScenarios(self.skill, bestPerColumn(rows))
  } else if (algorithmId) {
    const rows = entrantRows(ctx, (e) => algorithmIdFor(e) === algorithmId, r.entrant_id)
    if (rows.length > 1) base.results = { title: `Every ${c?.title ?? "such"} entrant`, max: scoreMax(ctx), rows }
  }

  // --- Live: a bandit strategy can play in the panel; a game model is already playing on the stage.
  const strategy = entrantStrategy(r)
  if (strategy) base.live = { kind: "bandit", ...strategy, scenario: "classic", label: entrantShortLabel(r), pickScenario: true }
  else base.live = undefined

  // --- Links and related
  const links = [...base.links]
  if (r.run_id) links.unshift({ kind: "run", label: "Training run", to: `/runs/${r.run_id}` })
  if (bandit && r.run_id) {
    const evolved = chapterLink("multi-armed-bandits", "Letting evolution choose the settings", "How its settings were evolved")
    if (evolved && !links.some((l) => l.to === evolved.to)) links.splice(1, 0, evolved)
  }
  base.links = links
  const related = [
    ...(c ? [`algorithm:${c.id}`] : []),
    `representation:${r.interface}`,
    selectionRef(shape?.selection),
    r.metrics.model.search_depth && r.metrics.model.search_depth > 1 && algorithmId !== "material-search" ? "algorithm:material-search" : null,
    game ? SCORE_METRIC[game] : null,
    ...base.related.map((x) => x.ref),
  ].filter((x): x is string => !!x && !!(concept(x) || x.startsWith("entrant:")))
  base.related = [...new Set(related)].map((x) => ({ ref: x, label: refLabel(x, ctx) }))
  return base
}

/** Every authored concept, for a check that each resolves and each chapter section exists. */
export const ALL_CONCEPTS: readonly ExplainConcept[] = [...CONCEPTS.values()]

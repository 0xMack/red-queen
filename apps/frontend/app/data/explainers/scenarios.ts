import type { ExplainConcept } from "~/types/explain"

// The bandit's scenarios (docs/design/0011) as explainers. The rules are `utils/bandit.ts`'s BANDIT_SCENARIOS (title,
// shape, lesson, pitfall); this adds how the machines are really set up -- mirrored from `games.bandit` /
// libs/games/rust/core/src/bandit.rs -- and who to watch. Who actually wins comes from the leaderboard records at
// resolve time. Imported explicitly (not a utils/ file), so BANDIT_SCENARIOS is imported too.
import { BANDIT_SCENARIOS } from "~/utils/bandit"

interface ScenarioExtra {
  setup: string[]
  /** A strategy to watch play it in the panel (id from BANDIT_STRATEGIES' strategy/params). */
  watch: { strategy: string; params: string; label: string }
  section?: string
  related?: string[]
}

const EXTRA: Record<string, ScenarioExtra> = {
  classic: {
    setup: [
      "Five win-or-lose machines with win rates spread like a ladder: about 15%, 30%, 45%, 60% and 75% (each ±5%), shuffled.",
      "100 pulls. The best machine is clearly best, if you give every machine a fair look.",
    ],
    watch: { strategy: "optimistic", params: "", label: "optimistic" },
    section: "One situation, five choices",
    related: ["algorithm:greedy", "algorithm:epsilon-greedy"],
  },
  "close-call": {
    setup: ["One machine wins 55% of pulls, and the other four win 42–50%.", "100 pulls: far too few to tell 55% from 50% with confidence."],
    watch: { strategy: "epsilon_greedy", params: "epsilon=0.3,decay=100", label: "ε-greedy, decaying" },
    section: "Which strategy wins depends on the game",
  },
  "lucky-start": {
    setup: [
      "Payouts are numbers, not wins: machines average 4, 4.75, 5.5, 6.25 and 7, each ± 3 on any pull.",
      "The spread is bigger than the gaps, so any machine can have a great first pull.",
    ],
    watch: { strategy: "greedy", params: "", label: "greedy" },
    section: "Greedy, and why it fails",
    related: ["algorithm:greedy", "algorithm:ucb"],
  },
  jackpot: {
    setup: [
      "One machine pays 50 with a 2% chance (worth 1.0 a pull on average), and one always pays 0.8.",
      "The other three win 30–60% of pulls.",
      "The jackpot is the best machine, but most games it looks like the worst.",
    ],
    watch: { strategy: "epsilon_greedy", params: "epsilon=0.3,decay=100", label: "ε-greedy, decaying" },
    section: "Which strategy wins depends on the game",
  },
  drifting: {
    setup: [
      "Classic's ladder of five win-or-lose machines, over 200 pulls.",
      "Somewhere between pulls 50 and 70, the best machine breaks: it starts paying like the worst one.",
    ],
    watch: { strategy: "epsilon_greedy", params: "epsilon=0.1,alpha=0.2", label: "ε-greedy, constant step" },
    section: "When the world changes",
    related: ["algorithm:epsilon-greedy", "algorithm:optimistic"],
  },
  "too-many-arms": {
    setup: ["Sixteen win-or-lose machines, win rates anywhere from 5% to 80%.", "100 pulls: six per machine if you try them all evenly."],
    watch: { strategy: "ucb1", params: "", label: "UCB1" },
    section: "Which strategy wins depends on the game",
    related: ["algorithm:ucb"],
  },
  "two-lamps": {
    setup: [
      "Before each pull a lamp lights red or blue, at random.",
      "Under red the machines follow classic's ladder. Under blue each one wins 0.9 − its red rate, so the best under one lamp is the worst under the other.",
      "Averaged over both lamps every machine wins 45%. A strategy that can't see the lamp has nothing to find.",
    ],
    watch: { strategy: "epsilon_greedy", params: "epsilon=0.1", label: "ε-greedy, seeing the lamp" },
    section: "Two lamps: when the situation matters",
    related: ["representation:bandit/lamp.v1+arm.v1", "representation:bandit/none.v1+arm.v1"],
  },
  detour: {
    setup: [
      "Two rooms. The red room has classic's ladder, except that one machine, the door, pays nothing.",
      "Pulling the door takes you to the gold room, where every machine pays 3 with a 40–80% chance. Your next pull is from there, then you're back in red.",
      "The door looks like the worst machine. It is the best plan: a 0 now buys up to 2.4 next pull.",
    ],
    watch: { strategy: "q_table", params: "gamma=0.9,initial_q=10,alpha=0.5,epsilon=0", label: "Q-learning, γ 0.9" },
    section: "The detour: when a pull changes what comes next",
    related: ["algorithm:q-table-bandit"],
  },
}

export const SCENARIOS: ExplainConcept[] = BANDIT_SCENARIOS.map((s) => {
  const extra = EXTRA[s.id]!
  const observer = s.sequential || s.contexts > 1 ? "lamp.v1" : "none.v1"
  return {
    kind: "scenario",
    id: s.id,
    title: s.title,
    family: `Bandit scenario · ${s.arms} machines × ${s.budget} pulls`,
    summary: `${s.lesson} ${s.pitfall}`,
    how: extra.setup,
    visual: { kind: "scenario", id: s.id },
    live: { kind: "bandit", scenario: s.id, observer, pickStrategy: true, ...extra.watch },
    facts: [
      { label: "machines", value: String(s.arms) },
      { label: "pulls", value: String(s.budget) },
      { label: "payouts", value: s.binary ? "win / lose" : "numbers" },
      { label: "situations", value: s.sequential ? "2 rooms" : s.contexts > 1 ? "2 lamps" : "1" },
    ],
    chapter: extra.section ? [{ slug: "multi-armed-bandits", section: extra.section }] : [{ slug: "multi-armed-bandits" }],
    related: [...(extra.related ?? []), "metric:skill"],
  }
})

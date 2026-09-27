// The bandit chapter's cited results (docs/design/0011): skill per strategy per scenario, from `jobs/evaluate_bandit.py`
// (protocol bandit.skill.v1, 500 held-out games per scenario, seeds 10,000-10,499). Hard-coded on purpose, like every
// chapter's figures: the numbers a chapter's prose cites must not change under a reader because someone re-ran a job.
// The game page draws the same table live from the leaderboard records.
import type { MatrixRow } from "~/components/bandit/BanditScenarioMatrix.vue"

const row = (id: string, label: string, v: number[]): MatrixRow => ({
  id,
  label,
  skill: Object.fromEntries(
    ["classic", "close-call", "lucky-start", "jackpot", "drifting", "too-many-arms", "two-lamps", "two-lamps:lamp.v1", "detour"].map((k, i) => [k, v[i]!]),
  ),
})

export const BANDIT_RESULTS: MatrixRow[] = [
  row("evolved-classic", "ε-greedy, evolved on classic", [80.4, 17.1, 4.8, 42.0, 67.3, 66.5, -0.1, 72.2, 14.6]),
  row("evolved-mixed", "ε-greedy, evolved on 4 scenarios", [78.5, 22.8, 6.0, 35.5, 71.8, 68.3, 0.1, 71.3, 13.9]),
  row("optimistic", "Optimistic start", [77.9, 22.2, 66.1, 4.2, 49.3, 61.8, 0.4, 68.0, 10.4]),
  row("ucb-tuned", "UCB, tuned (c 0.5)", [70.8, 17.6, 64.7, 5.4, 59.4, 43.2, 0.0, 59.1, 10.5]),
  row("thompson", "Thompson sampling", [64.4, 13.0, 53.5, 4.0, 47.3, 39.5, 0.3, 50.5, 2.7]),
  row("epsilon-decay", "ε-greedy, decaying", [64.1, 20.8, 54.2, 31.2, 43.5, 55.9, 0.3, 52.7, 8.6]),
  row("epsilon", "ε-greedy (ε 0.1)", [57.1, 15.0, 43.9, 21.9, 42.8, 51.0, -0.1, 46.1, 4.5]),
  row("gradient", "Gradient bandit", [53.1, 11.8, 50.0, 24.0, 40.6, 24.8, 0.2, 33.2, 7.2]),
  row("epsilon-tracking", "ε-greedy, constant step", [51.7, 13.9, 16.7, 9.6, 55.1, 47.0, 0.1, 42.4, 2.6]),
  row("q-lookahead", "Q-learning, looking ahead", [43.9, 6.6, 10.4, 24.3, 45.6, 29.4, 0.0, 34.2, 54.8]),
  row("ucb1", "UCB1 (c √2)", [39.1, 6.5, 52.8, 3.8, 36.9, 19.0, -0.6, 30.0, 2.2]),
  row("greedy", "Greedy", [38.8, 3.6, 0.5, -6.2, 13.6, 34.9, 0.1, 35.9, -4.2]),
  row("q-table", "Q-table (Q-learning's agent)", [31.4, 4.9, 12.2, -8.0, 41.1, 23.7, -0.2, 18.5, -2.6]),
  row("random", "Random", [-0.4, -0.2, -0.1, 0.0, -0.2, 0.4, -0.3, -0.3, 0.0]),
]

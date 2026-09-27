import BanditPlay from "~/components/bandit/BanditPlay.vue"
import BanditScenarioMatrix from "~/components/bandit/BanditScenarioMatrix.vue"
import BanditWatch from "~/components/bandit/BanditWatch.vue"
import type { GameModule } from "~/games/types"

// The multi-armed bandit (docs/design/0011): single-player, and its entrants are *strategies* -- algorithms that learn
// within one game, since a bandit's machines are drawn afresh every game -- scored by skill (0 = no better than
// random, 100 = the best machine every pull) on the default scenario's held-out games (jobs/evaluate_bandit.py). Every
// strategy runs anywhere (the Rust core in WebAssembly). Play is a race: you against the entrant you pick and two
// reference strategies, on the same machines.
export const banditModule: GameModule = {
  slug: "bandit",
  score: {
    label: "Skill",
    // Math.round(v) || 0: no "-0" for a strategy that is exactly as good as random
    format: (v) => (Math.round(v * 10) / 10 || 0).toFixed(1),
    compact: (v) => String(Math.round(v) || 0),
    scaleMin: 100,
    explain: "metric:skill",
  },
  columns: [
    {
      id: "best-arm",
      header: "Best machine",
      explain: "metric:best-rate",
      title: "share of pulls that went to the best machine",
      cell: (r) => {
        const s = r.metrics.bandit?.scenarios.classic
        return s ? { text: `${Math.round(s.best_rate * 100)}%`, sub: "of pulls", tone: "muted" } : null
      },
    },
    {
      id: "regret",
      header: "Regret",
      explain: "metric:regret",
      title: "expected payout given up against always pulling the best machine, per 100-pull game",
      cell: (r) => {
        const s = r.metrics.bandit?.scenarios.classic
        return s ? { text: s.regret.toFixed(1), sub: "per game", tone: "muted" } : null
      },
    },
  ],
  defaultEntrant: "top",
  selectInPlay: "stay",
  copy: {
    watch:
      "Five machines, a hundred pulls, and no idea which pays best. Watch a strategy decide when to explore and when to cash in -- with the table it keeps -- then race it yourself.",
    play: "Find the best machine before your pulls run out. The strategies play the same machines, with the same luck, one pull for each of yours.",
    scoreNote: ({ episodes, protocol }) =>
      `Skill on ${episodes ?? 500} held-out games of the classic scenario (${protocol}): 0 = no better than random, 100 = the best machine every pull.`,
    leaderboardIntro: ({ episodes, seeds }) =>
      `Ranked by skill on ${episodes} held-out games of the classic scenario (seeds ${seeds?.[0]}–${seeds?.[1]}). Nothing is trained beforehand: every strategy starts each game knowing nothing and learns as it pulls.`,
  },
  Watch: BanditWatch,
  Play: BanditPlay,
  initialHuman: (slug) => {
    const history = loadHumanHistory(slug)
    return history.games.length ? { score: history.best, label: "You (best)", live: false } : null
  },
  sections: { pareto: false, headToHead: false, representations: true },
  Insights: BanditScenarioMatrix,
}

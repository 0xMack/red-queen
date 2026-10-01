import CheckersStage from "~/components/checkers/CheckersStage.vue"
import type { GameModule } from "~/games/types"

// Checkers (docs/design/0006, 0007, 0013): two-player, ranked by a round robin (jobs/evaluate_versus.py). Its score is
// an *Elo rating* -- Bradley-Terry fitted to every game of the round robin, Random = 0 -- played in game pairs over a
// ballot of level openings. Unlike points per game against the field it doesn't saturate at the top, but it is still
// relative to the field, and the page says so. Every entrant here (baselines and trained champions) runs anywhere:
// the Rust core in WebAssembly, no model package to fit to the device. Watch and Play are one stage (CheckersStage,
// told apart by `mode`), because both are just "two players on a board".
export const checkersModule: GameModule = {
  slug: "checkers",
  score: {
    label: "Elo rating",
    format: (v) => v.toFixed(0),
    compact: (v) => v.toFixed(0),
    scaleMin: 400,
    explain: "metric:elo",
  },
  columns: [
    {
      id: "points",
      header: "Points per game",
      title: "points per game (win 1, draw ½) against every other entrant -- what the rating is fitted to",
      explain: "metric:points",
      cell: (r) => {
        const v = r.metrics.versus
        if (!v) return null
        const points = v.points ?? (v.wins + 0.5 * v.draws) / (v.wins + v.draws + v.losses)
        return { text: points.toFixed(2), sub: `${v.wins} · ${v.draws} · ${v.losses}`, tone: "muted" }
      },
    },
    {
      id: "pairs",
      header: "Game pairs",
      title: "game pairs (one opening, both seats) scoring 2 · 1½ · 1 · ½ · 0 points",
      explain: "metric:game-pairs",
      cell: (r) => {
        const p = r.metrics.versus?.pentanomial
        return p ? { text: [...p].reverse().join(" · "), sub: `${p.reduce((a, b) => a + b, 0)} pairs`, tone: "muted" } : null
      },
    },
    {
      id: "length",
      header: "Game length",
      title: "mean plies per game (a draw is 40 moves without a capture)",
      cell: (r) => ({ text: `${Math.round(r.metrics.quality.mean_steps)} plies`, tone: "muted" }),
    },
  ],
  // The strongest player on the leaderboard, baselines included -- watching two strong players is the
  // point; whether anything *trained* has caught up is what the table below says.
  defaultEntrant: "top",
  selectInPlay: "stay",
  copy: {
    watch:
      "The strongest entrant plays the next best, live -- with what each is weighing and how it values its moves. Pick any entrant to put it on the board, or change either seat.",
    play: "You are Red against the entrant you pick on the leaderboard. Same rules, same engine: click a ringed piece, then where it lands. Your rating is what your results against these opponents would earn you.",
    scoreNote: ({ protocol }) =>
      `Elo rating (Random = 0) fitted to every game of a round robin (${protocol}), in game pairs over level openings. A 200-point gap expects about 0.76 points per game. Relative to this field: add an entrant and every rating can shift.`,
    leaderboardIntro: ({ episodes }) =>
      `Ranked by a round robin -- every pair of entrants plays the same level openings from both seats, ${episodes} games per entrant, on games no trainer saw -- and rated in Elo: the ratings under which everyone's expected score matches the score they got, so beating a much weaker field earns nothing extra. Alongside, points per game and what each cost to train and to run.`,
  },
  Watch: CheckersStage,
  Play: CheckersStage,
  sections: { pareto: true, headToHead: true, representations: true },
}

import CheckersStage from "~/components/CheckersStage.vue"
import type { GameModule } from "~/games/types"

// Checkers (docs/design/0006, 0007): two-player, ranked by a round robin (jobs/evaluate_versus.py). Its
// score is *points per game* against every other entrant -- win 1, draw ½ -- so it's relative to the
// field, and the page says so. Every entrant here (baselines and trained champions) runs anywhere: the
// Rust core in WebAssembly, no model package to fit to the device. Watch and Play are one stage
// (CheckersStage, told apart by `mode`), because both are just "two players on a board".
export const checkersModule: GameModule = {
  slug: "checkers",
  score: {
    label: "Points per game",
    format: (v) => v.toFixed(2),
    compact: (v) => v.toFixed(2),
    scaleMin: 1,
  },
  columns: [
    {
      id: "record",
      header: "Won · drawn · lost",
      title: "wins / draws / losses over every game against every other entrant",
      cell: (r) => {
        const v = r.metrics.versus
        return v ? { text: `${v.wins} · ${v.draws} · ${v.losses}`, sub: `${v.wins + v.draws + v.losses} games`, tone: "muted" } : null
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
    play: "You are Red against the entrant you pick on the leaderboard. Same rules, same engine: click a ringed piece, then where it lands.",
    scoreNote: ({ episodes, protocol }) =>
      `Points per game (win 1, draw ½) over ${episodes ?? "many"} games against every other entrant (${protocol}). Relative to this field: add an entrant and every score shifts.`,
    leaderboardIntro: ({ episodes }) =>
      `Ranked by a round robin -- every pair of entrants plays both seats on games no trainer saw, ${episodes} per entrant -- never by training fitness. Draws count half, so 0.50 is dead even; alongside it, what each cost to train and to run.`,
  },
  Watch: CheckersStage,
  Play: CheckersStage,
  sections: { pareto: true, headToHead: true, representations: true },
}

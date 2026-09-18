export interface GameEntry {
  title: string
  summary: string
  status: "available" | "coming-soon"
  playHref?: string
  runsHref?: string
}

export const games: GameEntry[] = [
  {
    title: "Snake",
    summary:
      "A grid game with a relative (left/straight/right) action space. Play it yourself, or watch a trained neuroevolution policy play -- live, as it trains, or a finished result.",
    status: "available",
    playHref: "/play/snake",
    runsHref: "/runs",
  },
  {
    title: "Checkers",
    summary:
      "The first two-player game (docs/design/0006): real rules, mandatory captures, a reusable framework for pitting any strategy against any strategy. Framework and rules are built and tested; the UI isn't wired up yet.",
    status: "coming-soon",
  },
]

export interface GameEntry {
  slug: string
  title: string
  tagline: string
  summary: string
  status: "available" | "coming-soon"
  // The game's page (/games/<slug>): watch leaderboard entrants play, play it yourself, full
  // leaderboard -- docs/design/0007. "?mode=play" opens straight into playing.
  href: string
  // A real screenshot (public/screenshots/, captured from the running app) -- or, for a game with no
  // UI yet, `art: "checkers"` renders a real position from that game's own render_state().
  image?: string
  art?: "checkers"
  facts: string[]
  techniques: string[]
}

export const games: GameEntry[] = [
  {
    slug: "snake",
    title: "Snake",
    tagline: "Single-agent · neuroevolution",
    summary:
      "A 10×10 grid with a relative action space (left / straight / right). Play it yourself, or watch an evolved neural network play -- live as it trains, or any finished champion.",
    status: "available",
    href: "/games/snake",
    image: "/screenshots/snake.png",
    facts: ["10×10 grid", "3 actions", "2 representations", "held-out leaderboard"],
    techniques: ["Neuroevolution", "Lexicase selection", "Reward shaping"],
  },
  {
    slug: "checkers",
    title: "Checkers",
    tagline: "Two-player · strategy vs. strategy",
    summary:
      "The first two-player game (docs/design/0006): real rules -- mandatory captures, multi-jump chains, kinging -- and a framework for pitting any strategy against any strategy. Rules and match framework are built and tested; the UI is next.",
    status: "coming-soon",
    href: "/games/checkers",
    art: "checkers",
    facts: ["8×8 board", "mandatory captures", "multi-jump chains", "kinging"],
    techniques: ["Match fitness", "Co-evolution (planned)"],
  },
]

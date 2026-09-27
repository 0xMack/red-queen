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
    tagline: "Single-agent · learned policies",
    summary:
      "A 10×10 grid, three moves: turn left, go straight, turn right. Watch evolved networks, Q-tables, DQNs and PPO policies play games they've never seen -- then try to beat them.",
    status: "available",
    href: "/games/snake",
    image: "/screenshots/snake.png",
    facts: ["10×10 grid", "3 actions", "5 representations", "200 held-out games"],
    techniques: ["Neuroevolution", "NEAT", "Q-learning", "DQN", "PPO"],
  },
  {
    slug: "checkers",
    title: "Checkers",
    tagline: "Two-player · strategy vs. strategy",
    summary:
      "Real rules -- mandatory captures, multi-jump chains, kinging. Search, evolved evaluators and a self-play network meet in a round robin; watch the top two play with their reasoning on screen, or take one on.",
    status: "available",
    href: "/games/checkers",
    art: "checkers",
    facts: ["8×8 board", "round-robin leaderboard", "2 players", "held-out games"],
    techniques: ["Alpha-beta search", "Neuroevolution", "Self-play TD(λ)"],
  },
]

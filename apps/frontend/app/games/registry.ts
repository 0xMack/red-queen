import { checkersModule } from "~/games/checkers"
import { snakeModule } from "~/games/snake"
import type { GameModule } from "~/games/types"

// Every game with a page, by slug. A game listed in data/games.ts but absent here shows as coming soon.
// To add a game: write `app/games/<slug>.ts` (a GameModule -- see games/types.ts), register it here, add
// its entry to data/games.ts, and produce leaderboard records for it with an evaluation job.
export const gameModules: Record<string, GameModule> = {
  [snakeModule.slug]: snakeModule,
  [checkersModule.slug]: checkersModule,
}

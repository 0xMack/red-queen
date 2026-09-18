import { readFile, readdir } from "node:fs/promises"
import { fileURLToPath } from "node:url"

// Serves libs/games' actual Python source straight from disk -- the single source of truth for
// that package, same principle as apis/backend reusing telemetry's pydantic models directly rather
// than duplicating them. usePyodideGames.ts writes these files into Pyodide's virtual filesystem
// as the `games` package, so `import games.snake` in the browser runs the exact code
// libs/games/tests exercises, not a checked-in copy that can drift.
//
// Path resolution note: import.meta.url-relative, verified working both under `nuxt dev` and a
// built `.output/server/index.mjs` run from apps/frontend (Nitro preserves the relative offset to
// libs/games/src/games through its build). Not verified from a working directory or deployment
// layout other than "this repo checkout, run from apps/frontend" -- doc 0005 scopes deployment as
// out of scope for now.
const GAMES_SRC_DIR = fileURLToPath(new URL("../../../../libs/games/src/games", import.meta.url))

export default defineEventHandler(async () => {
  const entries = await readdir(GAMES_SRC_DIR)
  const files: Record<string, string> = {}
  for (const name of entries.filter((entry) => entry.endsWith(".py"))) {
    files[name] = await readFile(`${GAMES_SRC_DIR}/${name}`, "utf-8")
  }
  return files
})

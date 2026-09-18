import { readFile, readdir } from "node:fs/promises"
import { resolve } from "node:path"

// Serves a libs/<pkg> package's actual Python source straight from disk -- generalizes what was a
// single-purpose py-games.get.ts (doc 0005 step 4) into one route both Pyodide bridges use (games
// for gameplay, evolve for loading/running a trained WeightVector -- doc 0005 step 6). Single
// source of truth: the browser always runs the current libs/ code, never a checked-in copy that
// can drift.
//
// evolve needs its *whole* package, not just neuro.py: evolve/__init__.py eagerly imports every
// submodule, and all of them are pure stdlib (verified before wiring this up) -- no numpy/micropip
// needed in Pyodide for either package.
//
// Path resolution note: resolved from process.cwd(), not import.meta.url -- Nitro's dev-mode
// bundler doesn't preserve this route's actual source-tree depth (server/api/py-source/[pkg].get.ts
// is one directory deeper than the single-package route it replaced, but an import.meta.url-relative
// path landed two directories too high, not one -- a real, confirmed-by-running discrepancy, not a
// hypothetical). process.cwd() sidesteps that: both `nuxt dev` and a built
// `.output/server/index.mjs` are run from apps/frontend (verified for both), so cwd + "../../libs"
// is reliable there. Deployment from a different working directory is still out of scope
// (docs/design/0005).
const LIBS_DIR = resolve(process.cwd(), "../../libs")

const ALLOWED_PACKAGES = new Set(["games", "evolve"])

export default defineEventHandler(async (event) => {
  const pkg = getRouterParam(event, "pkg")
  if (!pkg || !ALLOWED_PACKAGES.has(pkg)) {
    throw createError({ statusCode: 404, statusMessage: `unknown package: ${pkg}` })
  }

  const srcDir = `${LIBS_DIR}/${pkg}/src/${pkg}`
  const entries = await readdir(srcDir)
  const files: Record<string, string> = {}
  for (const name of entries.filter((entry) => entry.endsWith(".py"))) {
    files[name] = await readFile(`${srcDir}/${name}`, "utf-8")
  }
  return files
})

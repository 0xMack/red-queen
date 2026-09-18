import type { PyodideInterface } from "pyodide"

// Pinned to match the `pyodide` npm package version exactly -- the JS loader and the CDN-hosted
// runtime files (wasm binary, stdlib zip) must be the same version.
const PYODIDE_VERSION = "314.0.7"
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

let pyodidePromise: Promise<PyodideInterface> | null = null

// Loads Pyodide once per browser session (module-level singleton, not per-page-visit) and writes
// libs/games' real source (server/api/py-games.get.ts) into its virtual filesystem as the `games`
// package. Only call this client-side (e.g. from onMounted) -- it touches WebAssembly/fetch and
// has no SSR-safe path. The dynamic import keeps pyodide's browser-only code out of the SSR bundle
// entirely, rather than needing a Vite external/optimizeDeps workaround.
export function loadGamesPyodide(): Promise<PyodideInterface> {
  pyodidePromise ??= initialize()
  return pyodidePromise
}

async function initialize(): Promise<PyodideInterface> {
  const { loadPyodide } = await import("pyodide")
  const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL })

  const files = await $fetch<Record<string, string>>("/api/py-games")
  pyodide.FS.mkdirTree("/py/games")
  for (const [name, content] of Object.entries(files)) {
    pyodide.FS.writeFile(`/py/games/${name}`, content)
  }
  pyodide.runPython(`
import sys
if "/py" not in sys.path:
    sys.path.insert(0, "/py")

# Pyodide bridge glue, not part of libs/games itself -- same boundary-flattening
# apis/backend/src/backend/game_sessions.json_safe_render_state() does for the JSON boundary.
# render_state()'s "cells" is dict[(x, y): label]; a Python tuple isn't a valid JS Map key, so
# toJs() raises ConversionError on it directly. Flatten to a list of {x, y, label} dicts instead,
# which toJs(dict_converter=...) can convert cleanly at every level.
def render_state_for_js(env):
    state = dict(env.render_state())
    cells = state.get("cells")
    if isinstance(cells, dict):
        state["cells"] = [{"x": x, "y": y, "label": label} for (x, y), label in cells.items()]
    return state
`)

  return pyodide
}

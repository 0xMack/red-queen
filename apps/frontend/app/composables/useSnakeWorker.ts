// A module-level singleton, not one Worker per page visit: Pyodide's first load downloads several
// MB (WASM runtime + Python stdlib) from the CDN, and that cost was previously paid again every
// time a person navigated to /play/snake or /watch/[runId] (each page created and terminated its
// own worker). Since this is a client-side SPA, the module scope survives route navigations within
// one browser session -- reusing the same worker makes every visit after the first instant.
// snakeGame.worker.ts's own ensurePyodideReady() is what makes a second "start" message on an
// already-loaded worker skip re-loading Pyodide; this composable just avoids creating a second
// worker (and paying for a second WASM runtime) in the first place.
let worker: Worker | null = null

export function getSnakeWorker(): Worker {
  worker ??= new Worker(new URL("../workers/snakeGame.worker.ts", import.meta.url), { type: "module" })
  return worker
}

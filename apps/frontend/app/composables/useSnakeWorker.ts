// A module-level singleton, not one Worker per page visit: the worker holds the WebAssembly game core
// and any ONNX Runtime build and model it has loaded (docs/design/0009), and keeping it alive across
// route navigations (this is a client-side SPA, so module scope survives them) means a model loaded
// on one page is instant on the next. snakeGame.worker.ts memoizes its own setup, so a second "start"
// on a warm worker re-uses everything.
let worker: Worker | null = null

export function getSnakeWorker(): Worker {
  worker ??= new Worker(new URL("../workers/snakeGame.worker.ts", import.meta.url), { type: "module" })
  return worker
}

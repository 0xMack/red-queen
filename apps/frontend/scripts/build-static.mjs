// The public site's build (docs/design/0019): a single-page app reading the API's static copy, for Cloudflare Pages.
//
//   pnpm generate:static     -> .output/public (the site) and .output/ort/<version>/ (ONNX Runtime, for the data host)
//
// Defaults, each overridable from the environment (e.g. a local copy for testing):
//   NUXT_PUBLIC_API_MODE  static
//   NUXT_PUBLIC_API_BASE  https://redqueen-data.mackinations.ca/api
//   NUXT_PUBLIC_ORT_BASE  https://redqueen-data.mackinations.ca/ort/<onnxruntime-web version>
// ONNX Runtime's own files are served from the data host, not the site: the WebGPU build is over Pages' 25 MiB file
// limit. They are copied to .output/ort/<version>/ for `redqueen dev deploy` to upload. Afterwards 404.html is removed:
// without one, Pages treats the site as a single-page app and answers a deep link (/runs/<id>) with index.html.
import { spawnSync } from "node:child_process"
import { copyFileSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs"
import { join } from "node:path"

const DATA_HOST = "https://redqueen-data.mackinations.ca"
const PAGES_FILE_LIMIT = 25 * 1024 * 1024
const ORT_FILES = [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
]

const ortDist = join("node_modules", "onnxruntime-web", "dist")
const ortVersion = JSON.parse(readFileSync(join("node_modules", "onnxruntime-web", "package.json"), "utf-8")).version
const env = {
  ...process.env,
  NUXT_PUBLIC_API_MODE: process.env.NUXT_PUBLIC_API_MODE ?? "static",
  NUXT_PUBLIC_API_BASE: process.env.NUXT_PUBLIC_API_BASE ?? `${DATA_HOST}/api`,
  NUXT_PUBLIC_ORT_BASE: process.env.NUXT_PUBLIC_ORT_BASE ?? `${DATA_HOST}/ort/${ortVersion}`,
}
console.log(`static build: API ${env.NUXT_PUBLIC_API_MODE} at ${env.NUXT_PUBLIC_API_BASE}; ONNX Runtime from ${env.NUXT_PUBLIC_ORT_BASE}`)
const result = spawnSync("pnpm", ["exec", "nuxt", "generate"], { env, stdio: "inherit", shell: process.platform === "win32" })
if (result.status !== 0) process.exit(result.status ?? 1)

const site = join(".output", "public")
rmSync(join(site, "404.html"), { force: true })
// ORT's own bundle has `new URL("ort-wasm-simd-threaded*.wasm", import.meta.url)` fallbacks, so Vite emits its .wasm
// files whatever runtime.ts imports. They're never fetched here (`wasmPaths` always points at NUXT_PUBLIC_ORT_BASE),
// and one is over the Pages file limit: delete them.
for (const file of readdirSync(join(site, "_nuxt"))) {
  if (file.startsWith("ort-wasm-simd-threaded")) rmSync(join(site, "_nuxt", file))
}

const ortOut = join(".output", "ort", ortVersion)
rmSync(join(".output", "ort"), { recursive: true, force: true })
mkdirSync(ortOut, { recursive: true })
for (const file of ORT_FILES) copyFileSync(join(ortDist, file), join(ortOut, file))

// Pages refuses a deploy with any file over 25 MiB: fail here, with the name, instead.
const tooBig = []
const walk = (dir) => {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    const stat = statSync(path)
    if (stat.isDirectory()) walk(path)
    else if (stat.size > PAGES_FILE_LIMIT) tooBig.push(`${path} (${(stat.size / 1048576).toFixed(1)} MiB)`)
  }
}
walk(site)
if (tooBig.length) {
  console.error(`static build: over Cloudflare Pages' 25 MiB file limit:\n  ${tooBig.join("\n  ")}`)
  process.exit(1)
}
console.log(`static build: ${site} ready (404.html removed: single-page fallback); ONNX Runtime ${ortVersion} in ${ortOut}`)

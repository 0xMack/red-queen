// The public site's build (docs/design/0019): a single-page app reading the API's static copy, served by a
// Cloudflare Worker's static assets (wrangler.jsonc).
//
//   pnpm generate:static     -> .output/public (the site) and .output/ort/<version>/ (ONNX Runtime, for the data host)
//
// Defaults, each overridable from the environment (e.g. a local copy for testing):
//   NUXT_PUBLIC_API_MODE  static
//   NUXT_PUBLIC_API_BASE  https://redqueen-data.mackinations.ca/api
//   NUXT_PUBLIC_ORT_BASE  https://redqueen-data.mackinations.ca/ort/<onnxruntime-web version>
// ONNX Runtime's own files are served from the data host, not the site: the WebGPU build is over Cloudflare's 25 MiB
// per-asset limit. They are copied to .output/ort/<version>/ for `redqueen dev deploy` to upload. Afterwards 404.html
// is removed: deep links (/runs/<id>) get the app from wrangler.jsonc's single-page fallback, never a 404 page.
import { spawnSync } from "node:child_process"
import { copyFileSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs"
import { homedir, hostname, userInfo } from "node:os"
import { join } from "node:path"

const DATA_HOST = "https://redqueen-data.mackinations.ca"
const ASSET_LIMIT = 25 * 1024 * 1024
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
  // Always plain static files. Anywhere Nitro detects a Cloudflare build (WORKERS_CI) it would otherwise pick its
  // `cloudflare-module` preset: a server Worker, with a generated wrangler.json that a .wrangler/deploy redirect makes
  // Wrangler use instead of wrangler.jsonc -- serving something other than these files.
  NITRO_PRESET: "static",
}

// ONNX Runtime's files as the data host serves them, at ort/<version>/ (for deploy_r2.py --ort-only to upload).
function stageOrt() {
  const out = join(".output", "ort", ortVersion)
  rmSync(join(".output", "ort"), { recursive: true, force: true })
  mkdirSync(out, { recursive: true })
  for (const file of ORT_FILES) copyFileSync(join(ortDist, file), join(out, file))
  return out
}

// `--stage-ort`: only stage ONNX Runtime, so a deploy can upload a new version before the build checks for it.
if (process.argv.includes("--stage-ort")) {
  console.log(`static build: ONNX Runtime ${ortVersion} staged in ${stageOrt()}`)
  process.exit(0)
}

// A leftover .wrangler/deploy redirect (from any earlier Cloudflare-preset build) would make `wrangler deploy` ignore
// wrangler.jsonc: never let one survive into a deploy.
rmSync(join(".wrangler", "deploy"), { recursive: true, force: true })
console.log(`static build: API ${env.NUXT_PUBLIC_API_MODE} at ${env.NUXT_PUBLIC_API_BASE}; ONNX Runtime from ${env.NUXT_PUBLIC_ORT_BASE}`)
const result = spawnSync("pnpm", ["exec", "nuxt", "generate"], { env, stdio: "inherit", shell: process.platform === "win32" })
if (result.status !== 0) process.exit(result.status ?? 1)

const site = join(".output", "public")
rmSync(join(site, "404.html"), { force: true })
// ORT's own bundle has `new URL("ort-wasm-simd-threaded*.wasm", import.meta.url)` fallbacks, so Vite emits its .wasm
// files whatever runtime.ts imports. They're never fetched here (`wasmPaths` always points at NUXT_PUBLIC_ORT_BASE),
// and one is over Cloudflare's per-asset limit: delete them.
for (const file of readdirSync(join(site, "_nuxt"))) {
  if (file.startsWith("ort-wasm-simd-threaded")) rmSync(join(site, "_nuxt", file))
}

const ortOut = stageOrt()

// The site loads ONNX Runtime from the data host, at this version's path: a build whose version isn't there yet (an
// onnxruntime-web upgrade) would deploy a site whose models never load. So check the host serves every file, at the
// right size. A deploy fails without it (CI's deploy job uploads a new version first); ORT_CHECK=warn (pull requests,
// which upload nothing) only warns. Only for an https base (a local copy for testing is the tester's business).
if (env.NUXT_PUBLIC_ORT_BASE.startsWith("https://")) {
  const missing = []
  for (const file of ORT_FILES) {
    const url = `${env.NUXT_PUBLIC_ORT_BASE}/${file}`
    const size = statSync(join(ortOut, file)).size
    try {
      // identity: a compressed response (fetch asks for one by default) carries no Content-Length to compare
      const response = await fetch(url, { method: "HEAD", headers: { "accept-encoding": "identity" } })
      const served = Number(response.headers.get("content-length"))
      if (!response.ok) missing.push(`${url}: HTTP ${response.status}`)
      else if (served !== size) missing.push(`${url}: ${served} bytes served, ${size} built`)
    } catch (error) {
      missing.push(`${url}: ${error.message}`)
    }
  }
  if (missing.length) {
    const warnOnly = process.env.ORT_CHECK === "warn"
    ;(warnOnly ? console.warn : console.error)(
      `static build: the data host doesn't serve ONNX Runtime ${ortVersion}:\n  ${missing.join("\n  ")}\n` +
        (warnOnly
          ? "(warning only: the deploy after merging uploads it)"
          : "Upload it first (from the repo root, after this build): " +
            "uv run --script jobs/publisher/scripts/deploy_r2.py --ort-only --ort apps/frontend/.output/ort"),
    )
    if (!warnOnly) process.exit(1)
  } else {
    console.log(`static build: the data host serves ONNX Runtime ${ortVersion}`)
  }
}

// Cloudflare refuses a deploy with any asset over 25 MiB: fail here, with the name, instead.
const tooBig = []
const walk = (dir) => {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    const stat = statSync(path)
    if (stat.isDirectory()) walk(path)
    else if (stat.size > ASSET_LIMIT) tooBig.push(`${path} (${(stat.size / 1048576).toFixed(1)} MiB)`)
  }
}
walk(site)
if (tooBig.length) {
  console.error(`static build: over Cloudflare's 25 MiB per-asset limit:\n  ${tooBig.join("\n  ")}`)
  process.exit(1)
}

// Nothing about this machine goes public (docs/design/0019): fail on its home directory, username or hostname in any
// file the site or the data host will serve -- e.g. a WASM build that kept its source paths (the WASM builds remap
// them; this catches one that didn't). Read from the machine, so nothing personal is written here.
const identifiers = new Set()
for (const path of [homedir(), process.cwd()]) {
  identifiers.add(path)
  identifiers.add(path.replaceAll("\\", "/"))
  identifiers.add(path.replaceAll("\\", "\\\\"))
}
// The bare username and hostname protect the machine that publishes. On CI they're the runner's (`runner`, an ordinary
// word in bundled code), so there only the paths are checked.
if (!process.env.CI) {
  for (const word of [userInfo().username, hostname()]) {
    if (word.length >= 4 && !["user", "root", "localhost"].includes(word.toLowerCase())) identifiers.add(word)
  }
}
const needles = [...identifiers].map((s) => Buffer.from(s.toLowerCase()))
const leaks = []
const scan = (dir) => {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) scan(path)
    else {
      const data = Buffer.from(readFileSync(path).toString("latin1").toLowerCase(), "latin1")
      for (const needle of needles) if (data.includes(needle)) leaks.push(`${path}: ${needle.toString()}`)
    }
  }
}
scan(site)
scan(join(".output", "ort"))
if (leaks.length) {
  console.error(`static build: local identifiers in published files:\n  ${leaks.slice(0, 20).join("\n  ")}`)
  process.exit(1)
}
console.log(`static build: ${site} ready (404.html removed: single-page fallback); ONNX Runtime ${ortVersion} in ${ortOut}`)

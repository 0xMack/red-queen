import init, { CopyCatGame, CopyCatTrace } from "~/wasm/games/games.js"
import gamesWasmUrl from "~/wasm/games/games_bg.wasm?url"

// Copy Cat in the browser (docs/design/0022): the game core's structured pipeline compiled to WebAssembly
// (rust/core/src/copycat_trace.rs -- the same code the leaderboard's `structured` entrant runs), on the main thread:
// at the sizes traced here it takes a few hundred milliseconds at most. Nothing is uploaded anywhere.

/** One held-out image, as `GET /games/copycat/targets` lists it. */
export interface CopyCatTarget {
  id: string
  seed: number
  severity: "clean" | "light" | "medium" | "heavy"
  levels: Record<string, unknown>
  width: number
  height: number
  /** A PNG data URL: the corrupted image an entrant is given. */
  png: string
}

/** Per-image scores (`arena.copycat.score`). */
export interface CopyCatImageScores {
  id: string
  severity: string
  fidelity: number
  psnr_black: number
  psnr_white: number
  alpha_iou: number
  layer_recovery: number
  shape_ratio: number
  node_ratio: number
  palette_ratio: number
  smoothness_ratio: number
  shapes: number
  nodes: number
  palette: number
  bytes: number
  refused: string | null
  image_us?: number
}

/** One held-out image with the clean SVG behind it and every entrant's SVG for it (`GET /games/copycat/targets/{seed}`). */
export interface CopyCatTargetDetail extends CopyCatTarget {
  clean: string
  protocol: string
  outputs: Record<string, { svg: string; scores: CopyCatImageScores }>
}

/** What the pipeline made at each stage, as plain data (the WASM objects are freed straight away). */
export interface CopyCatStages {
  width: number
  height: number
  sigma: number
  gradient: boolean
  /** Per label, its fill (label 0 is the background, which is left transparent). */
  palette: [number, number, number][]
  clustered: Uint16Array
  labels: Uint16Array
  svg: string
  /** Per shape, bottom first: its kind and the SVG of it alone. */
  layers: { kind: string; fill: string; svg: string }[]
  shapes: number
  nodes: number
  ms: number
}

let ready: Promise<unknown> | null = null
export function loadCopyCat(): Promise<unknown> {
  ready ??= init({ module_or_path: gamesWasmUrl })
  return ready
}

/** An SVG string as something an `<img>` shows (the browser draws it crisp at any size). */
export function svgDataUrl(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

const hex = (rgb: readonly number[]) => `#${rgb.map((c) => c.toString(16).padStart(2, "0")).join("")}`

/** Runs the pipeline on opaque RGBA pixels. Call `loadCopyCat()` first. */
export function traceStages(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): CopyCatStages {
  const start = performance.now()
  const trace = new CopyCatTrace(new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength), width, height)
  const ms = performance.now() - start
  try {
    const flat = trace.palette()
    const palette: [number, number, number][] = []
    for (let i = 0; i + 2 < flat.length; i += 3) palette.push([flat[i]!, flat[i + 1]!, flat[i + 2]!])
    const document = trace.document()
    try {
      const svg = document.canonical()
      const layers = Array.from({ length: document.shapes }, (_, i) => {
        const layer = document.layer(i)
        return { kind: document.kind(i), fill: layer.match(/fill="(#[0-9a-f]{6})"/)?.[1] ?? hex([128, 128, 128]), svg: layer }
      })
      return {
        width,
        height,
        sigma: trace.sigma,
        gradient: trace.gradient,
        palette,
        clustered: trace.clustered(),
        labels: trace.labels(),
        svg,
        layers,
        shapes: document.shapes,
        nodes: document.nodes,
        ms,
      }
    } finally {
      document.free()
    }
  } finally {
    trace.free()
  }
}

/** Counts for an SVG in the subset, or the reason it isn't one. Call `loadCopyCat()` first. */
export function describeSvg(svg: string): { shapes: number; nodes: number; palette: number } | { refused: string } {
  try {
    const document = new CopyCatGame(svg)
    try {
      return { shapes: document.shapes, nodes: document.nodes, palette: document.palette }
    } finally {
      document.free()
    }
  } catch (e) {
    return { refused: e instanceof Error ? e.message : String(e) }
  }
}

/** A label image as RGBA pixels: each label in its palette colour, the background transparent. */
export function labelPixels(labels: Uint16Array, palette: readonly (readonly number[])[]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(labels.length * 4)
  labels.forEach((label, i) => {
    if (label === 0) return
    const rgb = palette[label] ?? [128, 128, 128]
    out.set([rgb[0]!, rgb[1]!, rgb[2]!, 255], i * 4)
  })
  return out
}

/** The largest side a dropped-in image is scaled to before tracing: the held-out images are 128 px. */
export const TRACE_MAX_SIDE = 192

/** A picked or dropped image file as opaque RGBA pixels, scaled down to `maxSide` and flattened onto white. */
export async function imageFilePixels(file: Blob, maxSide = TRACE_MAX_SIDE): Promise<ImageData> {
  const bitmap = await createImageBitmap(file)
  try {
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height))
    const width = Math.max(1, Math.round(bitmap.width * scale))
    const height = Math.max(1, Math.round(bitmap.height * scale))
    const canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext("2d", { willReadFrequently: true })!
    context.fillStyle = "#ffffff" // a transparent input is read as on white, the way it would be shown
    context.fillRect(0, 0, width, height)
    context.imageSmoothingQuality = "high"
    context.drawImage(bitmap, 0, 0, width, height)
    return context.getImageData(0, 0, width, height)
  } finally {
    bitmap.close()
  }
}

/** A PNG data URL's pixels, opaque RGBA. */
export async function dataUrlPixels(url: string): Promise<ImageData> {
  const blob = await (await fetch(url)).blob()
  return imageFilePixels(blob, Number.POSITIVE_INFINITY)
}

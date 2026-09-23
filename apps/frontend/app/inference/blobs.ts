// Fetch content-addressed package blobs (docs/design/0009 Decision 5), verified against their sha256
// and kept in the Cache API forever: a blob's name *is* its content, so a cached copy can never be
// stale, and a returning visitor downloads nothing. Works on the main thread and in workers.

import type { Blob as PackageBlob } from "~/types/modelpack"

const CACHE_NAME = "redqueen-model-blobs-v1"
// Ask the browser not to evict big models under storage pressure (it may still say no).
const PERSIST_ABOVE_BYTES = 100 * 1024 * 1024

export type Progress = (loadedBytes: number) => void

// Cache keys are the hash, not the URL, so the same blob is reused across hosts (dev backend, CDN).
function cacheKey(sha: string): string {
  return `${self.location.origin}/__model-blobs/${sha}`
}

async function openCache(): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(CACHE_NAME)
  } catch {
    return null // storage blocked (private mode, disabled site data): just download every time
  }
}

async function sha256Hex(data: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data)
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("")
}

async function readWithProgress(response: Response, expected: number, onProgress?: Progress): Promise<Uint8Array<ArrayBuffer>> {
  if (!response.body || !onProgress) return new Uint8Array(await response.arrayBuffer())
  const out = new Uint8Array(expected)
  const reader = response.body.getReader()
  let offset = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (offset + value.length > expected) throw new Error(`blob is larger than its manifest says (${expected} bytes)`)
    out.set(value, offset)
    offset += value.length
    onProgress(offset)
  }
  return offset === expected ? out : out.subarray(0, offset)
}

export async function fetchBlob(baseUrl: string, blob: PackageBlob, onProgress?: Progress): Promise<Uint8Array> {
  const cache = await openCache()
  const cached = await cache?.match(cacheKey(blob.sha256))
  if (cached) {
    const data = new Uint8Array(await cached.arrayBuffer())
    onProgress?.(data.length)
    return data
  }
  const response = await fetch(`${baseUrl}/blobs/${blob.sha256}`)
  if (!response.ok) throw new Error(`blob ${blob.sha256.slice(0, 12)}: HTTP ${response.status}`)
  const data = await readWithProgress(response, blob.bytes, onProgress)
  const actual = await sha256Hex(data)
  if (actual !== blob.sha256) throw new Error(`blob ${blob.sha256.slice(0, 12)} failed its integrity check (got ${actual.slice(0, 12)})`)
  try {
    await cache?.put(cacheKey(blob.sha256), new Response(data, { headers: { "Content-Type": "application/octet-stream" } }))
  } catch {
    // Quota exceeded: still usable this time, just not cached.
  }
  return data
}

// Fetch several blobs with one combined progress figure (shards of one variant).
export async function fetchBlobs(baseUrl: string, blobs: PackageBlob[], onProgress?: (loaded: number, total: number) => void) {
  const total = blobs.reduce((sum, b) => sum + b.bytes, 0)
  if (total > PERSIST_ABOVE_BYTES) {
    await navigator.storage?.persist?.().catch(() => false)
  }
  const loaded = new Array<number>(blobs.length).fill(0)
  const report = () => onProgress?.(loaded.reduce((a, b) => a + b, 0), total)
  return Promise.all(
    blobs.map((blob, i) =>
      fetchBlob(baseUrl, blob, (n) => {
        loaded[i] = n
        report()
      }),
    ),
  )
}

export async function cachedBytes(): Promise<number> {
  const cache = await openCache()
  if (!cache) return 0
  let total = 0
  for (const request of await cache.keys()) {
    const response = await cache.match(request)
    total += Number(response?.headers.get("Content-Length") ?? 0) || (await response?.arrayBuffer())?.byteLength || 0
  }
  return total
}

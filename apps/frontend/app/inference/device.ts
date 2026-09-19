// What this browser can run (docs/design/0009 Decision 6). Plain TS, no Vue: used on the main thread
// (to explain *before* downloading anything) and safe to import from a worker.

export interface DeviceProfile {
  wasm: boolean
  wasmSimd: boolean
  // Multithreaded WASM needs SharedArrayBuffer, which needs the page cross-origin isolated
  // (COOP/COEP headers). Without it ORT still runs, single-threaded.
  threads: boolean
  logicalCores: number
  webgpu: {
    available: boolean
    reason: string | null // why not, in plain language, when unavailable
    vendor: string | null
    architecture: string | null
    features: string[]
    maxBufferSize: number
    maxStorageBufferBindingSize: number
  }
  deviceMemoryGb: number | null // navigator.deviceMemory: coarse (capped at 8), Chromium only
  storageQuotaBytes: number | null
  mobile: boolean
  browser: string
  forced: string[] // capabilities switched off by ?device=... for testing
}

// WebAssembly SIMD feature test: a module whose one function uses a v128 instruction. validate() is
// synchronous and doesn't instantiate anything.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
])

function browserName(ua: string): string {
  if (/Edg\//.test(ua)) return "Edge"
  if (/Firefox\//.test(ua)) return "Firefox"
  if (/Chrome\//.test(ua)) return "Chrome"
  if (/Safari\//.test(ua)) return "Safari"
  return "this browser"
}

function platformName(ua: string): string {
  if (/Android/.test(ua)) return "Android"
  if (/iPhone|iPad|iPod/.test(ua)) return "iOS"
  if (/Windows/.test(ua)) return "Windows"
  if (/Mac OS X/.test(ua)) return "macOS"
  if (/Linux/.test(ua)) return "Linux"
  return "this platform"
}

// `?device=nowebgpu,nowasm,nothreads` switches capabilities off, to check the unsupported paths
// without owning such a device.
function forcedOff(): string[] {
  if (typeof location === "undefined") return []
  const raw = new URLSearchParams(location.search).get("device")
  return raw ? raw.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean) : []
}

async function probeWebGpu(ua: string): Promise<DeviceProfile["webgpu"]> {
  const none = (reason: string) => ({
    available: false,
    reason,
    vendor: null,
    architecture: null,
    features: [],
    maxBufferSize: 0,
    maxStorageBufferBindingSize: 0,
  })
  const gpu = (globalThis.navigator as Navigator & { gpu?: GPU }).gpu
  if (!gpu) {
    return none(`${browserName(ua)} on ${platformName(ua)} doesn't offer WebGPU${/Firefox/.test(ua) ? " yet" : ""}`)
  }
  try {
    const adapter = await gpu.requestAdapter()
    if (!adapter) return none("WebGPU is present but no GPU adapter is available (blocklisted driver, or disabled)")
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info
    return {
      available: true,
      reason: null,
      vendor: info?.vendor || null,
      architecture: info?.architecture || null,
      features: [...adapter.features],
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    }
  } catch (e) {
    return none(`WebGPU adapter request failed: ${e instanceof Error ? e.message : String(e)}`)
  }
}

let cached: Promise<DeviceProfile> | null = null

export function probeDevice(): Promise<DeviceProfile> {
  cached ??= (async () => {
    const forced = forcedOff()
    const ua = globalThis.navigator?.userAgent ?? ""
    const wasm = typeof WebAssembly === "object" && !forced.includes("nowasm")
    let webgpu = await probeWebGpu(ua)
    if (forced.includes("nowebgpu")) {
      webgpu = { ...webgpu, available: false, reason: "WebGPU switched off for testing (?device=nowebgpu)" }
    }
    let storageQuotaBytes: number | null = null
    try {
      const estimate = await navigator.storage?.estimate()
      if (estimate?.quota !== undefined) storageQuotaBytes = estimate.quota - (estimate.usage ?? 0)
    } catch {
      // Private mode / blocked storage: unknown, not zero.
    }
    return {
      wasm,
      wasmSimd: wasm && WebAssembly.validate(SIMD_PROBE),
      threads: globalThis.crossOriginIsolated === true && !forced.includes("nothreads"),
      logicalCores: navigator.hardwareConcurrency ?? 1,
      webgpu,
      deviceMemoryGb: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null,
      storageQuotaBytes,
      mobile: /Android|iPhone|iPad|iPod|Mobile/.test(ua),
      browser: `${browserName(ua)} on ${platformName(ua)}`,
      forced,
    }
  })()
  return cached
}

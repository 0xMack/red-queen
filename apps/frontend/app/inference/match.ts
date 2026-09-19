// Pick the best variant/backend of a package this device can run -- or say, per variant, why not
// (docs/design/0009 Decision 6). Pure function of (manifest, allowed variants, profile, remembered
// failures), so the UI can explain a mismatch before anything is downloaded.

import type { DeviceProfile } from "~/inference/device"
import type { Backend, ModelManifest, Variant } from "~/types/modelpack"

// Above this, nothing downloads without an explicit click (bandwidth is the cost worth protecting).
export const CONFIRM_DOWNLOAD_BYTES = 50 * 1024 * 1024
// The WASM backend's heap is 32-bit, and also holds activations and the runtime itself.
const WASM_MAX_PEAK_BYTES = 3 * 1024 ** 3

export type ReasonCode = "no-backend" | "gpu-buffer" | "gpu-feature" | "memory" | "storage" | "failed-before"

export interface Rejection {
  variant: string
  backend: Backend | null
  code: ReasonCode
  message: string
}

export type Match =
  | { ok: true; variant: Variant; backend: Backend; needsConfirmation: boolean; rejected: Rejection[] }
  | { ok: false; rejected: Rejection[]; summary: string }

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(0)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

function checkBackend(variant: Variant, backend: Backend, profile: DeviceProfile): Rejection | null {
  const req = variant.requirements
  const reject = (code: ReasonCode, message: string): Rejection => ({ variant: variant.id, backend, code, message })
  if (backend === "webgpu") {
    if (!profile.webgpu.available) return reject("no-backend", `Needs WebGPU: ${profile.webgpu.reason}.`)
    const missing = req.webgpu_features.filter((f) => !profile.webgpu.features.includes(f))
    if (missing.length) return reject("gpu-feature", `Needs the GPU feature ${missing.join(", ")}, which this GPU doesn't offer.`)
    if (req.max_tensor_bytes > profile.webgpu.maxBufferSize) {
      return reject(
        "gpu-buffer",
        `Needs a ${formatBytes(req.max_tensor_bytes)} GPU buffer; this GPU allows ${formatBytes(profile.webgpu.maxBufferSize)}.`,
      )
    }
  } else {
    if (!profile.wasm) return reject("no-backend", "Needs WebAssembly, which is unavailable here.")
    if (req.peak_memory_bytes > WASM_MAX_PEAK_BYTES) {
      return reject("memory", `Needs ~${formatBytes(req.peak_memory_bytes)} of memory; WebAssembly tops out below that.`)
    }
  }
  if (profile.deviceMemoryGb !== null && profile.deviceMemoryGb < 8 && req.peak_memory_bytes > profile.deviceMemoryGb * 1024 ** 3 * 0.5) {
    // deviceMemory is deliberately coarse (privacy), so it only ever *rules out* a model on a small device.
    return reject("memory", `Needs ~${formatBytes(req.peak_memory_bytes)} of memory; this device reports ${profile.deviceMemoryGb} GB.`)
  }
  if (profile.storageQuotaBytes !== null && req.download_bytes > profile.storageQuotaBytes) {
    return reject(
      "storage",
      `A ${formatBytes(req.download_bytes)} download; this browser offers ${formatBytes(profile.storageQuotaBytes)} of storage.`,
    )
  }
  return null
}

export function matchVariant(
  manifest: ModelManifest,
  allowed: string[],
  profile: DeviceProfile,
  failedBefore: (variant: string, backend: Backend) => string | null = () => null,
): Match {
  const rejected: Rejection[] = []
  for (const id of allowed) {
    const variant = manifest.variants.find((v) => v.id === id)
    if (!variant) continue
    for (const backend of variant.requirements.backends) {
      const problem = checkBackend(variant, backend, profile)
      if (problem) {
        rejected.push(problem)
        continue
      }
      const failure = failedBefore(variant.id, backend)
      if (failure) {
        rejected.push({ variant: variant.id, backend, code: "failed-before", message: `Failed on this device before: ${failure}` })
        continue
      }
      return { ok: true, variant, backend, needsConfirmation: variant.requirements.download_bytes > CONFIRM_DOWNLOAD_BYTES, rejected }
    }
  }
  // One sentence for the UI: the most informative reason, not a list of every (variant, backend) pair.
  const summary = rejected.find((r) => r.code !== "no-backend")?.message ?? rejected[0]?.message ?? "No runnable variant."
  return { ok: false, rejected, summary }
}

// Runtime failures (allocation, device loss, a failed self-test) are remembered per device so a
// model that crashed once isn't retried on every visit. Only the browser's own storage -- a
// different device, or cleared site data, tries again.
const FAILURES_KEY = "redqueen.inference.failures"

type Failures = Record<string, { message: string; at: number }>

function readFailures(): Failures {
  try {
    return JSON.parse(localStorage.getItem(FAILURES_KEY) ?? "{}") as Failures
  } catch {
    return {}
  }
}

export function rememberedFailure(packageId: string) {
  const failures = readFailures()
  return (variant: string, backend: Backend) => failures[`${packageId}/${variant}/${backend}`]?.message ?? null
}

export function rememberFailure(packageId: string, variant: string, backend: Backend, message: string) {
  try {
    const failures = readFailures()
    failures[`${packageId}/${variant}/${backend}`] = { message, at: Date.now() }
    localStorage.setItem(FAILURES_KEY, JSON.stringify(failures))
  } catch {
    // Storage blocked: the failure just won't be remembered.
  }
}

export function forgetFailures() {
  try {
    localStorage.removeItem(FAILURES_KEY)
  } catch {
    // ignore
  }
}

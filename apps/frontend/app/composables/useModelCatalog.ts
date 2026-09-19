import { probeDevice, type DeviceProfile } from "~/inference/device"
import { forgetFailures, matchVariant, rememberFailure, rememberedFailure, type Match } from "~/inference/match"
import type { Catalog, CatalogEntry, ModelManifest } from "~/types/modelpack"

// Which of a game's published models (docs/design/0009) this device can run, and why not for the rest
// -- computed before anything but manifests (a few KB each) is downloaded, so the leaderboard can
// show every entrant with an honest "runs here / doesn't, because ..." next to it.

export interface Availability {
  entry: CatalogEntry
  manifest: ModelManifest | null
  match: Match | null // null while the manifest/device probe is still loading
  baseUrl: string
}

// Manifests are immutable (content-addressed), so one fetch per package per page load is enough.
const manifestCache = new Map<string, Promise<ModelManifest>>()

function fetchManifest(baseUrl: string, packageId: string): Promise<ModelManifest> {
  let manifest = manifestCache.get(packageId)
  if (!manifest) {
    manifest = $fetch<ModelManifest>(`${baseUrl}/manifests/${packageId}.json`)
    manifest.catch(() => manifestCache.delete(packageId))
    manifestCache.set(packageId, manifest)
  }
  return manifest
}

export function useModelCatalog(game: string) {
  const config = useRuntimeConfig()
  const catalog = ref<Catalog | null>(null)
  const profile = ref<DeviceProfile | null>(null)
  const manifests = ref<Record<string, ModelManifest>>({})
  const error = ref<string | null>(null)
  // Bumped when a failure is remembered/forgotten, so matches are recomputed.
  const failures = ref(0)

  const availability = computed<Record<string, Availability>>(() => {
    void failures.value
    const result: Record<string, Availability> = {}
    if (!catalog.value) return result
    for (const entry of catalog.value.entries) {
      const manifest = manifests.value[entry.package_id] ?? null
      result[entry.entrant_id] = {
        entry,
        manifest,
        baseUrl: catalog.value.base_url,
        match:
          manifest && profile.value
            ? matchVariant(manifest, entry.variants, profile.value, rememberedFailure(entry.package_id))
            : null,
      }
    }
    return result
  })

  async function load() {
    if (import.meta.server) return
    try {
      const [fetched, probed] = await Promise.all([
        $fetch<Catalog>(`/games/${game}/models`, { baseURL: config.public.apiBase }),
        probeDevice(),
      ])
      catalog.value = fetched
      profile.value = probed
      const ids = [...new Set(fetched.entries.map((e) => e.package_id))]
      const loaded = await Promise.all(ids.map((id) => fetchManifest(fetched.base_url, id).catch(() => null)))
      manifests.value = Object.fromEntries(ids.flatMap((id, i) => (loaded[i] ? [[id, loaded[i]]] : [])))
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    }
  }

  function reportFailure(packageId: string, variantId: string, backend: "webgpu" | "wasm", message: string) {
    rememberFailure(packageId, variantId, backend, message)
    failures.value += 1
  }

  function retryFailed() {
    forgetFailures()
    failures.value += 1
  }

  return { catalog, profile, availability, error, load, reportFailure, retryFailed }
}

<script setup lang="ts">
// The trained TinyLM, generating text in the visitor's browser (docs/design/0009 plan step 6): the
// published package from the `tinylm` catalog, run by ONNX Runtime in a worker (app/workers/lm.worker.ts)
// with its KV cache. Every variant this device can run is selectable -- fp32 and int8 side by side is
// the quantization tradeoff made visible -- and anything it can't run says why.
import { forgetFailures, formatBytes, matchVariant, rememberFailure, rememberedFailure, type Match } from "~/inference/match"
import type { LoadProgress } from "~/inference/runtime"
import type { Backend, ModelManifest } from "~/types/modelpack"

const models = useModelCatalog("tinylm")
onMounted(models.load)

const entry = computed(() => models.catalog.value?.entries[0] ?? null)
const available = computed(() => (entry.value ? models.availability.value[entry.value.entrant_id] ?? null : null))
const manifest = computed<ModelManifest | null>(() => available.value?.manifest ?? null)
const failures = ref(0)

// One match per variant, so each can be offered (or explained) on its own.
const perVariant = computed<{ id: string; match: Match }[]>(() => {
  void failures.value
  const m = manifest.value
  const profile = models.profile.value
  if (!m || !profile) return []
  return m.variants.map((v) => ({ id: v.id, match: matchVariant(m, [v.id], profile, rememberedFailure(m.package_id)) }))
})
const chosen = ref<string | null>(null)
watch(perVariant, (list) => {
  if (!chosen.value || !list.some((v) => v.id === chosen.value && v.match.ok)) chosen.value = list.find((v) => v.match.ok)?.id ?? null
})
const chosenMatch = computed(() => perVariant.value.find((v) => v.id === chosen.value)?.match ?? null)
const chosenVariant = computed(() => manifest.value?.variants.find((v) => v.id === chosen.value) ?? null)

const prompt = ref("Alice ")
const temperature = ref(0.8)
const maxNewTokens = ref(200)
const seed = ref(1)
const output = ref("")
const status = ref<"idle" | "loading" | "ready" | "generating" | "error">("idle")
const error = ref<string | null>(null)
const progress = ref<LoadProgress | null>(null)
const loaded = ref<{ backend: string; variantId: string; loadMs: number; selfTest: { maxAbsError: number; samples: number } | null } | null>(null)
const stats = ref<{ tokens: number; ms: number; firstMs: number | null; stepMs: number[]; recomputed: number } | null>(null)
const contextLength = computed(() => Number(manifest.value?.config.max_seq_len ?? 0))

let worker: Worker | null = null
let pendingLoadKey: string | null = null

function ensureWorker(): Worker {
  if (worker) return worker
  worker = new Worker(new URL("../workers/lm.worker.ts", import.meta.url), { type: "module" })
  worker.onmessage = (event: MessageEvent) => {
    const m = event.data
    if (m.type === "model_progress") progress.value = m.progress
    else if (m.type === "model_loaded") {
      progress.value = null
      loaded.value = m
      status.value = "ready"
    } else if (m.type === "model_error") {
      progress.value = null
      rememberFailure(m.packageId, m.variantId, m.backend as Backend, m.message)
      failures.value += 1
      models.reportFailure(m.packageId, m.variantId, m.backend, m.message)
      error.value = m.message
      status.value = "error"
    } else if (m.type === "token") {
      output.value += m.text
      const s = stats.value!
      s.tokens += 1
      s.stepMs.push(m.ms)
      if (s.firstMs === null) s.firstMs = m.ms
      if (m.recomputed) s.recomputed += 1
    } else if (m.type === "done") {
      stats.value!.ms = m.ms
      status.value = "ready"
    } else if (m.type === "error") {
      error.value = m.message
      status.value = "error"
    }
  }
  return worker
}

function load() {
  const m = chosenMatch.value
  const man = manifest.value
  if (!m?.ok || !man || !available.value) return
  const key = `${man.package_id}/${m.variant.id}/${m.backend}`
  if (key === pendingLoadKey && status.value !== "error") return
  pendingLoadKey = key
  status.value = "loading"
  error.value = null
  loaded.value = null
  const spec = { baseUrl: available.value.baseUrl, manifest: JSON.parse(JSON.stringify(man)), variantId: m.variant.id, backend: m.backend }
  ensureWorker().postMessage({ type: "load", spec })
}
watch(chosenMatch, (m) => m?.ok && !m.needsConfirmation && load())

function run() {
  if (status.value !== "ready") return
  output.value = prompt.value
  stats.value = { tokens: 0, ms: 0, firstMs: null, stepMs: [], recomputed: 0 }
  status.value = "generating"
  worker!.postMessage({ type: "generate", prompt: prompt.value, maxNewTokens: maxNewTokens.value, temperature: temperature.value, seed: seed.value })
}

function stop() {
  worker?.postMessage({ type: "stop" })
}

function retryFailed() {
  forgetFailures()
  models.retryFailed()
  failures.value += 1
  pendingLoadKey = null
  load()
}

const tokensPerSecond = computed(() => {
  const s = stats.value
  if (!s || s.stepMs.length < 2) return null
  const steady = s.stepMs.slice(1) // the first step processes the whole prompt
  return 1000 / (steady.reduce((a, b) => a + b, 0) / steady.length)
})

onUnmounted(() => worker?.terminate())
</script>

<template>
  <div class="card my-6 p-5" data-tinylm>
    <div class="flex flex-wrap items-center gap-2">
      <span class="font-display font-semibold">Run it: TinyLM in your browser</span>
      <span v-if="entry" class="chip">{{ entry.label }}</span>
    </div>
    <p v-if="models.error.value" class="mt-3 text-sm text-queen-300">Couldn't load the model catalog: {{ models.error.value }}</p>
    <p v-else-if="models.catalog.value && !entry" class="mt-3 text-sm text-fg-muted">
      No TinyLM is published yet -- <code>uv run python jobs/tinylm_run.py</code>, then
      <code>jobs/publish_models.py tinylm</code>.
    </p>
    <template v-else-if="manifest">
      <p class="mt-2 text-sm text-fg-muted">{{ manifest.description }}</p>

      <div class="mt-4 flex flex-wrap gap-2" role="radiogroup" aria-label="Model variant">
        <button
          v-for="v in perVariant"
          :key="v.id"
          class="rounded-lg border px-3 py-2 text-left text-xs transition"
          :class="[
            chosen === v.id ? 'border-queen-400/70 bg-raised' : 'border-line hover:border-line-strong',
            v.match.ok ? '' : 'cursor-not-allowed opacity-55',
          ]"
          :disabled="!v.match.ok"
          :title="v.match.ok ? '' : v.match.summary"
          :data-variant="v.id"
          @click="chosen = v.id"
        >
          <span class="font-mono font-semibold text-fg">{{ v.id }}</span>
          <span class="text-fg-subtle">
            · {{ formatBytes(manifest.variants.find((x) => x.id === v.id)!.requirements.download_bytes) }}
            · {{ ((manifest.variants.find((x) => x.id === v.id)!.parity.action_agreement ?? 0) * 100).toFixed(1) }}% same next character
          </span>
          <span v-if="v.match.ok" class="block text-fg-subtle">runs on {{ v.match.backend }}</span>
          <span v-else class="block text-queen-300">✕ {{ v.match.summary }}</span>
        </button>
      </div>

      <div v-if="perVariant.length && !perVariant.some((v) => v.match.ok)" class="mt-3 text-sm text-fg-muted">
        This device can't run any variant of this model.
        <button v-if="perVariant.some((v) => !v.match.ok && v.match.rejected.some((r) => r.code === 'failed-before'))" class="link" @click="retryFailed">
          Try again anyway
        </button>
      </div>
      <button v-else-if="chosenMatch?.ok && chosenMatch.needsConfirmation && status === 'idle'" class="btn-ghost btn-sm mt-3" @click="load">
        Download {{ formatBytes(chosenVariant!.requirements.download_bytes) }} and load
      </button>

      <div class="mt-4 grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto]">
        <input
          v-model="prompt"
          class="rounded-lg border border-line bg-sunken px-3 py-2 font-mono text-sm text-fg outline-none focus:border-line-strong"
          aria-label="Prompt"
          maxlength="200"
          @keydown.enter="run"
        />
        <div class="flex gap-2">
          <button v-if="status !== 'generating'" class="btn-primary btn-sm" :disabled="status !== 'ready'" data-generate @click="run">Generate</button>
          <button v-else class="btn-ghost btn-sm" @click="stop">Stop</button>
        </div>
      </div>
      <div class="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-fg-subtle">
        <label class="flex items-center gap-2">temperature
          <input v-model.number="temperature" type="range" min="0" max="1.5" step="0.05" class="accent-queen-500" />
          <span class="num w-8 text-fg">{{ temperature.toFixed(2) }}</span>
        </label>
        <label class="flex items-center gap-2">characters
          <input v-model.number="maxNewTokens" type="number" min="1" max="2000" class="w-20 rounded border border-line bg-sunken px-2 py-1 text-fg" />
        </label>
        <label class="flex items-center gap-2">seed
          <input v-model.number="seed" type="number" min="0" class="w-20 rounded border border-line bg-sunken px-2 py-1 text-fg" />
        </label>
      </div>

      <div class="mt-4 min-h-24 rounded-lg border border-line bg-sunken p-4 font-mono text-[13px] break-words whitespace-pre-wrap text-fg" data-output>
        <template v-if="output">{{ output }}</template>
        <span v-else-if="status === 'loading'" class="text-fg-subtle">
          <template v-if="progress?.stage === 'download'">Downloading · {{ formatBytes(progress.loaded ?? 0) }} / {{ formatBytes(progress.total ?? 0) }}</template>
          <template v-else-if="progress?.stage === 'self-test'">Checking it against the trained model…</template>
          <template v-else>Loading ONNX Runtime…</template>
        </span>
        <span v-else-if="status === 'error'" class="text-queen-300">{{ error }}</span>
        <span v-else class="text-fg-subtle">Press Generate.</span>
      </div>

      <p v-if="loaded" class="mt-3 text-xs text-fg-subtle" data-lm-runtime>
        ONNX Runtime · <span class="font-mono">{{ loaded.variantId }}</span> on <span class="font-mono">{{ loaded.backend }}</span>
        · loaded in {{ Math.round(loaded.loadMs) }} ms
        <template v-if="loaded.selfTest"> · self-test max |Δlogit| {{ loaded.selfTest.maxAbsError.toExponential(1) }}</template>
        <template v-if="stats && stats.tokens">
          · {{ stats.tokens }} characters<template v-if="tokensPerSecond"> · {{ tokensPerSecond.toFixed(0) }} chars/s</template>
          <template v-if="stats.recomputed"> · {{ stats.recomputed }} past the {{ contextLength }}-character window (recomputed)</template>
        </template>
      </p>
    </template>
    <p v-else class="mt-3 text-sm text-fg-subtle">Checking what this device can run…</p>
  </div>
</template>

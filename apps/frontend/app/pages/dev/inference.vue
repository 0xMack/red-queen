<script setup lang="ts">
// Development page for client-side inference (docs/design/0009): what this browser can run, what the
// model cache holds, and the scale-test package -- a ~340 MB randomly initialized model that exercises
// sharded downloads, the confirmation click, persistent storage and WebGPU at a realistic size.
// Not linked from the site's navigation.
import { cachedBytes } from "~/inference/blobs"
import { probeDevice, type DeviceProfile } from "~/inference/device"
import { forgetFailures, formatBytes } from "~/inference/match"

useHead({ title: "Inference (dev)" })
const profile = ref<DeviceProfile | null>(null)
const cached = ref<number | null>(null)
const persisted = ref<boolean | null>(null)

async function refresh() {
  profile.value = await probeDevice()
  cached.value = await cachedBytes()
  persisted.value = (await navigator.storage?.persisted?.()) ?? null
}
async function clearCache() {
  await caches.delete("redqueen-model-blobs-v1")
  forgetFailures()
  await refresh()
}
onMounted(refresh)
</script>

<template>
  <main class="mx-auto max-w-[1100px] px-4 py-8 sm:px-6 lg:px-8">
    <p class="eyebrow">Development</p>
    <h1 class="mt-2 text-3xl font-semibold">Client-side inference</h1>
    <p class="mt-2 max-w-3xl text-fg-muted">
      What this browser offers, what's cached, and a deliberately large model to test the big-model path.
      Add <code>?device=nowebgpu</code>, <code>nowasm</code> or <code>nothreads</code> to the URL to switch
      capabilities off.
    </p>

    <section class="card mt-6 p-5 text-sm" data-device-profile>
      <h2 class="font-display font-semibold">This device</h2>
      <dl v-if="profile" class="mt-3 grid gap-x-6 gap-y-1 sm:grid-cols-[180px_minmax(0,1fr)]">
        <dt class="text-fg-subtle">Browser</dt><dd>{{ profile.browser }}<template v-if="profile.forced.length"> · forced off: {{ profile.forced.join(", ") }}</template></dd>
        <dt class="text-fg-subtle">WebGPU</dt>
        <dd>
          <template v-if="profile.webgpu.available">
            {{ profile.webgpu.vendor ?? "unknown vendor" }} {{ profile.webgpu.architecture ?? "" }} · max buffer
            {{ formatBytes(profile.webgpu.maxBufferSize) }} · features: {{ profile.webgpu.features.join(", ") || "none" }}
          </template>
          <template v-else>unavailable -- {{ profile.webgpu.reason }}</template>
        </dd>
        <dt class="text-fg-subtle">WebAssembly</dt>
        <dd>{{ profile.wasm ? "yes" : "no" }} · SIMD {{ profile.wasmSimd ? "yes" : "no" }} · threads {{ profile.threads ? `yes (${profile.logicalCores} cores)` : "no (page isn't cross-origin isolated)" }}</dd>
        <dt class="text-fg-subtle">Memory hint</dt><dd>{{ profile.deviceMemoryGb ? `${profile.deviceMemoryGb} GB` : "not reported" }}</dd>
        <dt class="text-fg-subtle">Storage available</dt><dd>{{ profile.storageQuotaBytes !== null ? formatBytes(profile.storageQuotaBytes) : "unknown" }}</dd>
        <dt class="text-fg-subtle">Model cache</dt>
        <dd>
          {{ cached !== null ? formatBytes(cached) : "…" }} · persistent storage {{ persisted === null ? "unknown" : persisted ? "granted" : "not granted" }}
          <button class="link ml-2" @click="clearCache">clear cache and remembered failures</button>
        </dd>
      </dl>
    </section>

    <ClientOnly>
      <TinyLMPlayground catalog="scale-test" title="Scale test: a ~340 MB model in your browser" />
    </ClientOnly>
  </main>
</template>

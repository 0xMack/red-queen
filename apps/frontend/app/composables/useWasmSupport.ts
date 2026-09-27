import { probeDevice } from "~/inference/device"

// Can this browser run the WebAssembly cores? `null` while probing (and during SSR), then true/false. A live lab
// trains when it's true and falls back to a recording when it's false (`?device=nowasm` forces that).
export function useWasmSupport() {
  const live = ref<boolean | null>(null)
  onMounted(async () => {
    live.value = (await probeDevice()).wasm
  })
  return live
}

import type { CodeLang } from "~/types/code"

// The reader's preferred snippet language. One module-level ref, so choosing Rust in one CodeBlock switches every
// block on the page, and localStorage carries it to the next page and the next visit. The server always renders the
// default (pseudocode); a stored preference applies once mounted, like any other per-viewer convenience.
const KEY = "redqueen.code.lang"
const preferred = ref<CodeLang>("pseudo")
let loaded = false

export function useCodeLanguage() {
  onMounted(() => {
    if (loaded) return
    loaded = true
    try {
      const stored = localStorage.getItem(KEY)
      if (stored === "pseudo" || stored === "python" || stored === "rust") preferred.value = stored
    } catch {
      // Storage unavailable: the default stands.
    }
  })

  function setPreferred(lang: CodeLang) {
    preferred.value = lang
    try {
      localStorage.setItem(KEY, lang)
    } catch {
      // Not persisted; applies to this page view.
    }
  }

  return { preferred: readonly(preferred), setPreferred }
}

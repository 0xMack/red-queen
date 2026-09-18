import { createHighlighterCore, type HighlighterCore } from "shiki/core"
import { createJavaScriptRegexEngine } from "shiki/engine/javascript"

export type CodeLang = "python" | "typescript" | "bash" | "json"

// A fine-grained bundle (explicit langs/theme, the JS regex engine instead of the WASM oniguruma
// one) instead of shiki's full bundle -- this Learn section only ever shows a handful of
// languages, and the JS engine avoids WASM asset loading entirely (one less thing to get wrong
// under Nuxt's SSR + client bundling). Module-level singleton: loaded once per browser session,
// shared by every CodeBlock instance, same pattern as useSnakeWorker.ts.
let highlighterPromise: Promise<HighlighterCore> | null = null

export function useHighlighter(): Promise<HighlighterCore> {
  highlighterPromise ??= createHighlighterCore({
    themes: [import("shiki/themes/github-dark.mjs")],
    langs: [
      import("shiki/langs/python.mjs"),
      import("shiki/langs/typescript.mjs"),
      import("shiki/langs/bash.mjs"),
      import("shiki/langs/json.mjs"),
    ],
    engine: createJavaScriptRegexEngine(),
  })
  return highlighterPromise
}

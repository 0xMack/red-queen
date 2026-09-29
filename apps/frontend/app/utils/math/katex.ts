import katex from "katex"

// KaTeX, configured once (docs/design/0012). `trust` and the relaxed `strict` are for `\htmlData` only: every input is
// authored by us, compiled from an expression tree, never user text. Output is HTML plus hidden MathML, so screen
// readers get the maths. Renders are memoised: a live lab re-renders the same few strings over and over.

const cache = new Map<string, string>()
const LIMIT = 500

export function renderTex(tex: string, display = true): string {
  const key = `${display ? "D" : "I"}${tex}`
  const hit = cache.get(key)
  if (hit !== undefined) return hit
  const html = katex.renderToString(tex, {
    displayMode: display,
    output: "htmlAndMathml",
    trust: (context) => context.command === "\\htmlData",
    strict: (code: string) => (code === "htmlExtension" ? "ignore" : "warn"),
    throwOnError: import.meta.dev,
  })
  if (cache.size >= LIMIT) cache.delete(cache.keys().next().value!)
  cache.set(key, html)
  return html
}

// A code snippet in one or more languages (CodeBlock). The Learn chapters default to pseudocode, Python and Rust --
// the idea, then each of the project's two implementation languages -- and a snippet can have just one when that's
// all there is (a call into a library that only exists in one of them, a shell command, a JSON record).
export type CodeLang = "pseudo" | "python" | "rust" | "typescript" | "bash" | "json"

// Tab order, and the order a block falls back through when it lacks the reader's preferred language.
export const CODE_LANGS: CodeLang[] = ["pseudo", "python", "rust", "typescript", "bash", "json"]

export const CODE_LANG_LABELS: Record<CodeLang, string> = {
  pseudo: "Pseudocode",
  python: "Python",
  rust: "Rust",
  typescript: "TypeScript",
  bash: "Shell",
  json: "JSON",
}

export interface CodeVariant {
  code: string
  // The repo file this is the project's own code from (possibly trimmed). Left out, the variant is written for the
  // page -- a translation of the real code into another language, or pseudocode -- and the block says so.
  source?: string
}

export type Snippet = Partial<Record<CodeLang, string | CodeVariant>>

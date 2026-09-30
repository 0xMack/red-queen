import type { LanguageRegistration } from "shiki/core"

// A TextMate grammar for the Learn chapters' pseudocode -- shiki has none, and plain text would make pseudocode the
// one tab without highlighting. The dialect is the textbook one: `function f(x)`, `for each x in xs`, `if … then`,
// `←` for assignment, `//` comments, maths symbols as operators. Scope names are the ones utils/codeTheme.ts colours.
export const pseudocodeGrammar: LanguageRegistration = {
  name: "pseudo",
  scopeName: "source.pseudo",
  repository: {},
  patterns: [
    { name: "comment.line.double-slash.pseudo", match: "//.*$" },
    { name: "string.quoted.double.pseudo", match: '"[^"]*"' },
    {
      match: "\\b(function|procedure)\\s+([A-Za-z_][\\w]*)",
      captures: { 1: { name: "storage.type.function.pseudo" }, 2: { name: "entity.name.function.pseudo" } },
    },
    {
      name: "keyword.control.pseudo",
      match:
        "\\b(return|if|then|else|elif|for|each|in|while|repeat|until|do|end|break|continue|let|with|to|from|by|yield|loop|of|otherwise)\\b",
    },
    { name: "keyword.operator.logical.pseudo", match: "\\b(and|or|not)\\b" },
    { name: "constant.language.pseudo", match: "\\b(true|false|none|nothing|∞)\\b" },
    { name: "constant.numeric.pseudo", match: "\\b\\d+(\\.\\d+)?([eE][-+]?\\d+)?\\b" },
    { name: "meta.function-call.pseudo", match: "\\b[A-Za-z_][\\w]*(?=\\()" },
    { name: "keyword.operator.pseudo", match: "←|→|≤|≥|≠|∈|∉|×|·|Σ|∑|∏|√|∇|∂|≈|\\+|-|\\*|/|=|<|>" },
  ],
}

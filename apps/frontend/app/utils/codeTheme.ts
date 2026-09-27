import type { ThemeRegistration } from "shiki/core"

// The syntax theme for CodeBlock, drawn from the design tokens (utils/palette.ts) so code reads as part of the page:
// keywords in the queen's red, strings green, numbers gold, calls blue, types violet, comments quiet and italic.
export const inkTheme: ThemeRegistration = {
  name: "red-queen-ink",
  type: "dark",
  colors: { "editor.background": "#00000000", "editor.foreground": "#dcd5c8" },
  tokenColors: [
    { scope: ["comment", "punctuation.definition.comment"], settings: { foreground: palette.fgSubtle, fontStyle: "italic" } },
    { scope: ["keyword", "storage", "storage.type", "keyword.control", "keyword.operator.new", "keyword.operator.logical"], settings: { foreground: palette.queen300 } },
    { scope: ["string", "string.quoted", "punctuation.definition.string"], settings: { foreground: palette.life300 } },
    { scope: ["constant.numeric", "constant.language", "constant.character"], settings: { foreground: palette.gold300 } },
    { scope: ["entity.name.function", "support.function", "meta.function-call", "variable.function"], settings: { foreground: palette.signal300 } },
    { scope: ["entity.name.type", "entity.name.class", "support.type", "support.class", "entity.other.inherited-class"], settings: { foreground: palette.violet300 } },
    { scope: ["variable.parameter", "variable.language"], settings: { foreground: "#e8c9a8" } },
    { scope: ["keyword.operator", "punctuation", "meta.brace"], settings: { foreground: palette.fgMuted } },
    { scope: ["entity.name.tag", "support.type.property-name", "meta.object-literal.key"], settings: { foreground: palette.queen200 } },
  ],
}

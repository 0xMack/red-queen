// The design tokens (assets/css/main.css `@theme`) as literal values, for the places CSS classes can't reach:
// SVG presentation attributes, chart series colours, canvas. SVG attributes don't take `var(--…)`, so this is
// the one mirror of the stylesheet -- change a colour in both. Components never write a hex value themselves.

const tokens = {
  bg: "#0c0b0a",
  sunken: "#080807",
  surface: "#131211",
  raised: "#1b1a18",
  line: "#2a2825",
  lineStrong: "#3d3a35",
  fg: "#efe9df",
  fgMuted: "#b0a899",
  fgSubtle: "#7a7368",

  queen200: "#ffd0c6",
  queen300: "#ffa18f",
  queen400: "#ff6b55",
  queen500: "#ef4630",
  queen600: "#c7321f",
  queen700: "#962414",

  life300: "#bfeea0",
  life400: "#93dc6b",
  life500: "#6cbf45",
  signal300: "#b2cdff",
  signal400: "#80a9ff",
  gold300: "#f6d690",
  gold400: "#eab551",
  violet300: "#d3c2ff",
  violet400: "#b39bff",
  teal400: "#5fd0c0",
  orange400: "#f59a5b",
  pink400: "#f08bc0",
}

export type PaletteColor = keyof typeof tokens
// Typed as plain strings, so a series' colour can be any token.
export const palette: Readonly<Record<PaletteColor, string>> = tokens

/** A token at an opacity, e.g. `alpha("queen400", 0.2)` -- for fills computed per datum (heatmaps, activations). */
export function alpha(color: PaletteColor, a: number): string {
  const hex = palette[color]
  const n = Number.parseInt(hex.slice(1), 16)
  return `rgb(${(n >> 16) & 255} ${(n >> 8) & 255} ${n & 255} / ${Math.round(a * 1000) / 1000})`
}

/** Series colours in the order comparisons use them: this run, the one before, then references. */
export const seriesColors = [palette.queen400, palette.signal400, palette.life400, palette.gold400, palette.violet400, palette.teal400] as const

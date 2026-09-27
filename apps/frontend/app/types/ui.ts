// Shapes the ui/ primitives take.

/** One readout in a `UiStats` row. */
export interface UiStat {
  label: string
  value: string | number
  tone?: "default" | "queen" | "life" | "gold" | "signal"
  wide?: boolean
}

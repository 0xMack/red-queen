import type { components } from "~/types/api.gen"

// The staged plan's targets (docs/design/0022, "Where the room is"), as what a report can answer. Stage 3 isn't here: its
// limit comes from corrupting the clean SVGs, which a report doesn't record.
type Summary = components["schemas"]["ReportSummary"]

export interface CopyCatStage {
  stage: number
  title: string
  target: number
  /** The target as text, with its unit. */
  goal: string
  /** The group the progress chart draws this target over, if it is a mean fidelity. */
  group?: string
  measure: (report: Summary) => number | undefined
}

export const COPYCAT_STAGES: readonly CopyCatStage[] = [
  {
    stage: 1,
    title: "Close colours and layers, without corruption",
    target: 35.5,
    goal: "35.5 dB",
    group: "icons/train-control",
    measure: (r) => r.groups["icons/train-control"]?.mean,
  },
  {
    stage: 2,
    title: "The worst image without corruption",
    target: 20,
    goal: "20 dB",
    measure: (r) => r.groups["icons/train-control"]?.worst,
  },
  {
    stage: 4,
    title: "Layer recovery on icons",
    target: 0.75,
    goal: "0.75",
    measure: (r) => r.groups["icons/train"]?.columns.layer_recovery,
  },
]

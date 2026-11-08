import type { RunInfo } from "~/types/telemetry"

// A Copy Cat report as its run holds it (`copycat.report`, docs/design/0023), and the staged plan's targets
// (docs/design/0022, "Where the room is") as what a report can answer. Stage 3 isn't here: its limit comes from corrupting
// the clean SVGs, which a report doesn't record.
export interface ReportGroup {
  mean: number
  worst: number
  columns: Record<string, number>
  by_severity: Record<string, number>
}
export type ReportGroups = Record<string, ReportGroup>

export const reportGroups = (run: RunInfo): ReportGroups => (run.summary?.groups as ReportGroups | undefined) ?? {}

export interface CopyCatStage {
  stage: number
  title: string
  target: number
  /** The target as text, with its unit. */
  goal: string
  /** The group the progress chart draws this target over, if it is a mean fidelity. */
  group?: string
  measure: (groups: ReportGroups) => number | undefined
}

export const COPYCAT_STAGES: readonly CopyCatStage[] = [
  {
    stage: 1,
    title: "Close colours and layers, without corruption",
    target: 35.5,
    goal: "35.5 dB",
    group: "icons/train-control",
    measure: (g) => g["icons/train-control"]?.mean,
  },
  {
    stage: 2,
    title: "The worst image without corruption",
    target: 20,
    goal: "20 dB",
    measure: (g) => g["icons/train-control"]?.worst,
  },
  {
    stage: 4,
    title: "Layer recovery on icons",
    target: 0.75,
    goal: "0.75",
    measure: (g) => g["icons/train"]?.columns.layer_recovery,
  },
]

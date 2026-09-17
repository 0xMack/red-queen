// Mirrors apis/backend's response models, which are themselves libs/telemetry's pydantic models
// (telemetry.RunInfo, telemetry.GenerationStats) reused directly with no API-layer duplication.
// Keep these two in sync with libs/telemetry/src/telemetry/{registry,types}.py.

export type RunStatus = "running" | "paused" | "completed" | "failed"

export interface RunInfo {
  run_id: string
  config: Record<string, unknown>
  status: RunStatus
  created_at: number
  updated_at: number
  summary: Record<string, unknown> | null
}

export interface GenerationStats {
  run_id: string
  island_id: string | null
  generation: number
  timestamp: number
  best_fitness: number
  mean_fitness: number
  worst_fitness: number
  diversity: number
  champion_ref: string
}

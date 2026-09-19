// Mirrors libs/telemetry/src/telemetry/evaluations.py's EvaluationRecord and
// libs/games/src/games/observation.py's Interface.describe() (docs/design/0007) -- keep in sync by
// hand. `metrics` is free-form on the Python side; the shape below is what jobs/evaluate.py writes
// for protocol snake.score.v1.

export type EntrantKind = "champion" | "baseline" | "human"

export interface QualityMetrics {
  n: number
  mean: number
  ci95: number
  median: number
  min: number
  max: number
  zero_rate: number
  mean_steps: number
  train_mean: number | null
  generalization_gap: number | null
  scores?: number[] // every held-out game's score, in seed order (absent on older records)
}

export interface InferenceMetrics {
  encode_us: number
  decide_us: number
  total_us: number
  parameters: number
  artifact_bytes: number
}

export interface TrainingMetrics {
  measured: boolean
  none?: boolean // a baseline: nothing was trained
  generations?: number
  fitness_evaluations?: number | null
  episodes?: number | null
  env_steps?: number | null
  wall_s?: number | null
  paused_s?: number | null
  active_s?: number | null
  cpu_s?: number | null
  peak_rss_bytes?: number | null
  hardware?: Record<string, unknown> | null
}

export interface EvaluationRecord {
  game: string
  protocol: string
  entrant_id: string
  entrant_kind: EntrantKind
  label: string
  interface: string
  run_id: string | null
  champion_ref: string | null
  created_at: number
  metrics: {
    quality: QualityMetrics
    inference: InferenceMetrics
    training: TrainingMetrics
    model: { description: string; observer_level: number; note: string | null }
    protocol: { held_out_seeds: [number, number]; episodes: number; max_steps: number; board: Record<string, number>; metric: string }
  }
  hardware: { cpu?: string; python?: string; hardware_class?: string; engine?: string; logical_cores?: number }
}

export interface InterfaceInfo {
  id: string
  game: string
  observer: {
    id: string
    level: number
    level_name: string
    description: string
    size: number
    feature_names: string[]
  }
  action: { id: string; description: string; num_outputs: number }
}

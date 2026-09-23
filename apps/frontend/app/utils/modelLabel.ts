// One way to name a trained model everywhere (runs table, run page, leaderboards): the algorithm plus
// the size of what it produced -- "NEAT · 11 hidden · 73 conns", "Neuroevolution · 11 → 16 → 3". The
// leaderboard gets these facts from jobs/evaluate.py (`metrics.model.shape`), the runs table from the
// run's config and its latest generation's `extras`. Auto-imported (app/utils/).

export interface ModelShape {
  algorithm: string
  selection?: string | null
  hidden_nodes?: number
  connections?: number
  layer_sizes?: number[]
  // a tabular policy (docs/design/0010): its rows, how many the agent ever learned about, its actions
  table_states?: number
  visited_states?: number
  actions?: number
}

/** The size part alone: "11 hidden · 73 conns" for an evolved graph, "11 → 16 → 3" for an MLP, "table · 256 of
 * 2048 states" for a tabular policy. */
export function modelSize(shape: ModelShape): string | null {
  if (shape.hidden_nodes !== undefined && shape.connections !== undefined) {
    return `${shape.hidden_nodes} hidden · ${shape.connections} conns`
  }
  if (shape.layer_sizes?.length) return shape.layer_sizes.join(" → ")
  if (shape.table_states !== undefined) {
    return shape.visited_states !== undefined
      ? `table · ${shape.visited_states} of ${shape.table_states} states`
      : `table · ${shape.table_states} states`
  }
  return null
}

export function modelLabel(shape: ModelShape): string {
  const size = modelSize(shape)
  return size ? `${shape.algorithm} · ${size}` : shape.algorithm
}

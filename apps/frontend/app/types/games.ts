// A grid game's render state (games.rendering / render_state(), with `cells` flattened to records), as the
// session worker builds it from the WASM core's cells.

export interface GridCell {
  x: number
  y: number
  label: string
}

export interface RenderState {
  width: number
  height: number
  cells: GridCell[]
  score: number
  alive: boolean
}

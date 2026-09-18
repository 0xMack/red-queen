// Mirrors apis/backend/src/backend/schemas.py -- keep in sync if those change.

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

export interface GameSessionState {
  session_id: string
  game: string
  render_state: RenderState
  reward: number
  done: boolean
  step: number
}

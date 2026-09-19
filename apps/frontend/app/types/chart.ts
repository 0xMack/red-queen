export interface ChartSeries {
  key: string
  label: string
  color: string
  values: number[]
  width?: number
  dashed?: boolean
}

export interface ParetoPoint {
  id: string
  label: string
  x: number
  y: number
  err: number
  color: string
}

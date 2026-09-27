// A tiny Snake for drawing what an observer sees (the representation explainers' diagrams and their live panel).
// Illustration only: the real game and observers are the Rust core (libs/games/rust/core), which training and the
// leaderboard use. `features` follows features.v1's definition exactly (it is 11 yes/no facts); the rays and flood
// fills show *what* egocentric.v1/v2 measure, not their exact encoding.

export type Cell = readonly [number, number]

export interface SnakeSketch {
  w: number
  h: number
  /** Head first. */
  body: Cell[]
  /** 0 → right, 1 ↓ down, 2 ← left, 3 ↑ up (features.v1's heading order). */
  heading: number
  food: Cell
  seed: number
}

export const DIRS: Cell[] = [
  [1, 0],
  [0, 1],
  [-1, 0],
  [0, -1],
]

/** A fixed, legible position: the head mid-board facing right, a bend behind it, food up and to the right. */
export function sketchStart(): SnakeSketch {
  return {
    w: 10,
    h: 10,
    body: [
      [4, 5],
      [3, 5],
      [3, 4],
      [4, 4],
      [5, 4],
      [5, 3],
    ],
    heading: 0,
    food: [7, 2],
    seed: 7,
  }
}

const add = (a: Cell, b: Cell): Cell => [a[0] + b[0], a[1] + b[1]]
const same = (a: Cell, b: Cell) => a[0] === b[0] && a[1] === b[1]
const inside = (s: SnakeSketch, c: Cell) => c[0] >= 0 && c[1] >= 0 && c[0] < s.w && c[1] < s.h
/** Body cells that will still be there next step (the tail moves away unless the snake eats). */
const blocks = (s: SnakeSketch, c: Cell) => s.body.slice(0, -1).some((b) => same(b, c))

/** Relative moves: -1 left, 0 straight, 1 right -> absolute heading. */
export const turn = (heading: number, move: number) => (heading + move + 4) % 4

export function deadly(s: SnakeSketch, move: number): boolean {
  const next = add(s.body[0]!, DIRS[turn(s.heading, move)]!)
  return !inside(s, next) || blocks(s, next)
}

/** features.v1: danger straight/left/right, heading one-hot (→ ↓ ← ↑), food left/right/up/down. */
export function features(s: SnakeSketch): number[] {
  const [hx, hy] = s.body[0]!
  const [fx, fy] = s.food
  const heading = [0, 1, 2, 3].map((d) => (d === s.heading ? 1 : 0))
  return [deadly(s, 0), deadly(s, -1), deadly(s, 1)].map(Number).concat(heading, [fx < hx, fx > hx, fy < hy, fy > hy].map(Number))
}

export const FEATURE_NAMES = ["danger ahead", "danger left", "danger right", "heading →", "heading ↓", "heading ←", "heading ↑", "food ←", "food →", "food ↑", "food ↓"]

/** One cell's grid-flat.v1 value: empty 0, body 1, head 2, food 3. */
export function cellValue(s: SnakeSketch, c: Cell): number {
  if (same(s.body[0]!, c)) return 2
  if (s.body.some((b) => same(b, c))) return 1
  if (same(s.food, c)) return 3
  return 0
}

export interface Ray {
  name: string
  dir: Cell
  /** Where the ray stops (the last cell inside the board). */
  end: Cell
  /** Distance to the first wall / body / food along it; null when there is none. */
  wall: number
  body: number | null
  food: number | null
}

const RAYS: { name: string; f: number; r: number }[] = [
  { name: "left", f: 0, r: -1 },
  { name: "front-left", f: 1, r: -1 },
  { name: "front", f: 1, r: 0 },
  { name: "front-right", f: 1, r: 1 },
  { name: "right", f: 0, r: 1 },
  { name: "back-left", f: -1, r: -1 },
  { name: "back-right", f: -1, r: 1 },
]

/** egocentric.v1's seven rays, in the head's frame. */
export function rays(s: SnakeSketch): Ray[] {
  const f = DIRS[s.heading]!
  const r = DIRS[turn(s.heading, 1)]!
  return RAYS.map(({ name, f: a, r: b }) => {
    const dir: Cell = [a * f[0] + b * r[0], a * f[1] + b * r[1]]
    let c = s.body[0]!
    let d = 0
    let body: number | null = null
    let food: number | null = null
    for (;;) {
      const next = add(c, dir)
      d += 1
      if (!inside(s, next)) break
      c = next
      if (body === null && s.body.slice(1).some((x) => same(x, c))) body = d
      if (food === null && same(s.food, c)) food = d
    }
    return { name, dir, end: c, wall: d, body, food }
  })
}

/** Cells reachable from `from` without crossing the body (egocentric.v2's flood fill). */
export function reachable(s: SnakeSketch, from: Cell): Cell[] {
  if (!inside(s, from) || blocks(s, from)) return []
  const seen = new Set([`${from[0]},${from[1]}`])
  const queue: Cell[] = [from]
  const out: Cell[] = []
  while (queue.length) {
    const c = queue.shift()!
    out.push(c)
    for (const d of DIRS) {
      const n = add(c, d)
      const key = `${n[0]},${n[1]}`
      if (inside(s, n) && !seen.has(key) && !blocks(s, n)) {
        seen.add(key)
        queue.push(n)
      }
    }
  }
  return out
}

/** Per move (left, straight, right): the share of free cells still reachable after it, and whether the tail is. */
export function space(s: SnakeSketch): { move: number; cells: Cell[]; share: number; tail: boolean }[] {
  const free = s.w * s.h - s.body.length
  const tail = s.body.at(-1)!
  return [-1, 0, 1].map((move) => {
    const cells = reachable(s, add(s.body[0]!, DIRS[turn(s.heading, move)]!))
    const tailReachable = cells.some((c) => DIRS.some((d) => same(add(c, d), tail)))
    return { move, cells, share: cells.length / free, tail: cells.length > 0 && tailReachable }
  })
}

function nextRandom(seed: number): number {
  return (Math.imul(seed, 1103515245) + 12345) >>> 0
}

/** One step of a scripted player (safe, roomy, then toward the food): the live panel's snake. Restarts when stuck. */
export function step(s: SnakeSketch): SnakeSketch {
  const head = s.body[0]!
  const options = space(s)
    .filter((o) => !deadly(s, o.move))
    .map((o) => {
      const next = add(head, DIRS[turn(s.heading, o.move)]!)
      const dist = Math.abs(next[0] - s.food[0]) + Math.abs(next[1] - s.food[1])
      return { ...o, score: o.share * 100 + (o.tail ? 30 : 0) - dist }
    })
    .sort((a, b) => b.score - a.score)
  if (!options.length || s.body.length > 28) return sketchStart()
  const heading = turn(s.heading, options[0]!.move)
  const next = add(head, DIRS[heading]!)
  const ate = same(next, s.food)
  const body = [next, ...(ate ? s.body : s.body.slice(0, -1))]
  let food = s.food
  let seed = s.seed
  if (ate) {
    const free: Cell[] = []
    for (let y = 0; y < s.h; y++) for (let x = 0; x < s.w; x++) if (!body.some((b) => same(b, [x, y]))) free.push([x, y])
    seed = nextRandom(seed)
    food = free[seed % free.length]!
  }
  return { ...s, body, heading, food, seed }
}

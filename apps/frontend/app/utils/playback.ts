// The vocabulary every game's playback controls share, so "how fast" means the same thing on every page:
// a speed *multiplier* (a game's own timing -- Snake's tick, a versus game's delay between moves -- is its base
// interval divided by it). Auto-imported (app/utils/).

export const SPEEDS = [0.5, 1, 2, 4, 8] as const

export const speedLabel = (speed: number) => (speed === 0.5 ? "½×" : `${speed}×`)

/** Snake's base tick (ms) at 1×. */
export const SNAKE_BASE_TICK_MS = 110
/** A versus game's base pause (ms) between moves at 1× -- long enough to follow a move's animation. */
export const VERSUS_BASE_DELAY_MS = 550

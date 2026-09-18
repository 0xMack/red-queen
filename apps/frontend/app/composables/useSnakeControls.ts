// Clockwise order, must match games.snake._DIRECTIONS exactly -- translates an absolute arrow-key
// press into Snake's relative action space using a client-tracked heading (doc 0005's worked
// example). A pure function, no DOM/lifecycle coupling, so both a window-scoped listener (the
// dedicated /play page, which owns the whole page's keyboard input) and an element-scoped one
// (LiveSnakeDemo, embedded in a page that shouldn't have its arrow keys hijacked globally) can
// share the same translation logic without sharing how it's wired to events.
const HEADING_FOR_KEY: Record<string, number> = { ArrowRight: 0, ArrowDown: 1, ArrowLeft: 2, ArrowUp: 3 }

export function createHeadingTracker() {
  let headingIndex = 0
  return {
    reset() {
      headingIndex = 0
    },
    translate(key: string): number | null {
      const target = HEADING_FOR_KEY[key]
      if (target === undefined) return null
      const delta = (target - headingIndex + 4) % 4
      const action = delta === 1 ? 1 : delta === 3 ? -1 : 0 // right turn, left turn, or already-there/reversal
      headingIndex = (headingIndex + action + 4) % 4
      return action
    },
  }
}

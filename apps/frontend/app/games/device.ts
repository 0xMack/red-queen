import type { GameDevice } from "~/games/types"

/** For a game whose entrants all run on any device. */
export function noDevice(): GameDevice {
  return { load() {}, runsHere: ref({}), summary: ref(null), filterable: ref(false), context: null }
}

import type { CopyCatTargetDetail } from "~/utils/copycat"

// The image you added to Copy Cat's stage, kept for the visit: the stage is remounted for every entrant you watch, and
// your image should still be there for the next one. Never stored or uploaded.
const own = shallowRef<CopyCatTargetDetail | null>(null)
const name = ref("yours")

export function useCopyCatOwnImage() {
  return { own, name }
}

import CopyCatPlay from "~/components/copycat/CopyCatPlay.vue"
import CopyCatProgress from "~/components/copycat/CopyCatProgress.vue"
import CopyCatWatch from "~/components/copycat/CopyCatWatch.vue"
import type { GameModule, LeaderboardColumn } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"

// Copy Cat (docs/design/0022): not a game you step but an image-to-SVG task. An entrant turns each held-out image -- a
// clean SVG rendered and corrupted the way a generator would (`corrupt.v1`) -- into an SVG, scored against the clean
// SVG behind it under `copycat.clean.v1` (`arena.copycat`). Watch shows what an entrant made of each image; Play traces
// your own image with the structured pipeline, in the browser.

const columns = (r: EvaluationRecord) => r.metrics.copycat?.columns

/** A ratio to the clean SVG's: 1× is right, so the further from 1 either way, the more it's muted. */
function ratio(id: string, header: string, key: "shape_ratio" | "palette_ratio" | "smoothness_ratio", title: string): LeaderboardColumn {
  return {
    id,
    header,
    title,
    explain: "metric:structure-ratio",
    cell: (r) => {
      const v = columns(r)?.[key]
      if (v === undefined) return null
      const far = Math.max(v, 1 / v) > 2
      return { text: `${v < 10 ? v.toFixed(1) : Math.round(v)}×`, sub: "of clean", tone: far ? "warn" : "muted" }
    },
  }
}

export const copycatModule: GameModule = {
  slug: "copycat",
  score: {
    label: "Clean fidelity",
    format: (v) => `${v.toFixed(1)} dB`,
    compact: (v) => v.toFixed(1),
    scaleMin: 30,
    explain: "metric:clean-fidelity",
  },
  columns: [
    {
      id: "alpha-iou",
      header: "Alpha IoU",
      explain: "metric:alpha-iou",
      title: "how well the output's coverage matches the clean SVG's: a leftover background or a hole loses",
      cell: (r) => {
        const v = columns(r)?.alpha_iou
        return v === undefined ? null : { text: v.toFixed(2), tone: v < 0.5 ? "warn" : "muted" }
      },
    },
    {
      id: "layer-recovery",
      header: "Layers",
      explain: "metric:layer-recovery",
      title: "each clean part's best whole-shape match in the output, averaged: a crescent for a hidden circle scores low",
      cell: (r) => {
        const v = columns(r)?.layer_recovery
        return v === undefined ? null : { text: v.toFixed(2), sub: "recovered", tone: "muted" }
      },
    },
    ratio("shapes", "Shapes", "shape_ratio", "shapes, as a multiple of the clean SVG's"),
    ratio("palette", "Colours", "palette_ratio", "distinct fills, as a multiple of the clean SVG's"),
    ratio("smoothness", "Turning", "smoothness_ratio", "total turning along the outlines, as a multiple of the clean SVG's: jagged edges turn a lot"),
  ],
  defaultEntrant: "top",
  selectInPlay: "watch",
  copy: {
    watch:
      "A clean, layered SVG was rendered, then blurred, noised, compressed and set on a background, the way an image generator delivers a logo. Pick an image and see what each entrant recovered, beside the clean SVG it is scored against.",
    playLabel: "Your image",
    play: "Drop in your own image -- a generated logo or icon -- and the structured pipeline turns it into a layered SVG with a transparent background, here on your device.",
    scoreNote: ({ protocol }) =>
      `Clean fidelity on 24 held-out images (${protocol}): PSNR of the SVG's render against the clean SVG's, over black and over white, averaged.`,
    leaderboardIntro: () =>
      "Ranked by clean fidelity on the 24 held-out synthetic images, six per corruption level. Every entrant is scored against the clean SVG the image was made from -- never against the corrupted input -- so reproducing the noise loses, and so does keeping the background.",
  },
  Watch: CopyCatWatch,
  Play: CopyCatPlay,
  Insights: CopyCatProgress,
  sections: { pareto: true, headToHead: false, representations: false },
}

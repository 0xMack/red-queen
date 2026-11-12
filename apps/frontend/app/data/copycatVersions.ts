// The MDL tracer's versions (docs/design/0022, "Versions"): one per kept change, in the order they merged. The
// leaderboard entrant `baseline:structured` is always the latest; its label is "MDL tracer N", and the evaluator's
// MDL_TRACER_VERSION must equal the last version here (jobs/evaluator/tests checks it).
//
// Each score is clean fidelity (dB) on a held-out set, taken once when the change was done -- never tuned on. A set
// that didn't exist yet, or wasn't scored for that change, is absent. `pr` and `merge` are where the version shipped on
// main; `report` is its `copycat.report` run on /runs (from version 14 on: the reports began with it), whose own commit
// may carry "+dirty" -- the change measured before it was committed.
//
// Reconstructed from the merged PRs, the report records and docs/design/0022's held-out results on 2026-11-07.

export interface CopyCatVersion {
  version: number
  /** The change, in the words of the Learn case study. */
  change: string
  pr: number
  /** The merge commit on main; absent while the version's PR is open. */
  merge?: string
  report?: string
  heldout: { synthetic?: number; icons?: number; broad?: number }
  /** `copycat.clean.v2`'s score (the mean of the three sets' means), where it was run. */
  v2?: number
}

export const COPYCAT_VERSIONS: CopyCatVersion[] = [
  { version: 1, change: "The MDL tracer", pr: 70, merge: "824349b", heldout: { synthetic: 29.0 } },
  { version: 2, change: "Refine every shape", pr: 72, merge: "bcd1396", heldout: { synthetic: 30.8 } },
  { version: 3, change: "Smooth splines", pr: 73, merge: "21d4296", heldout: { synthetic: 31.38 } },
  { version: 4, change: "Cut a part in two", pr: 75, merge: "bfcc6bf", heldout: { synthetic: 31.43 } },
  { version: 5, change: "Knots by least squares", pr: 76, merge: "15e4a4e", heldout: { synthetic: 31.99 } },
  // icons was first scored on this version, unchanged, when the set arrived (#79)
  { version: 6, change: "A point must earn its keep", pr: 77, merge: "a064769", heldout: { synthetic: 32.0, icons: 18.52 } },
  { version: 7, change: "Join only what an occluder cut", pr: 80, merge: "43ebb4e", heldout: { synthetic: 31.999, icons: 18.77 } },
  { version: 8, change: "A background that reaches the corners", pr: 81, merge: "43d72bd", heldout: { synthetic: 32.003, icons: 19.41 } },
  { version: 9, change: "Paint order by fraction", pr: 82, merge: "cc3c590", heldout: { synthetic: 32.23, icons: 20.88 } },
  { version: 10, change: "A corner fit noise can't break", pr: 83, merge: "0e0038b", heldout: { synthetic: 32.23, icons: 21.37 } },
  { version: 11, change: "Light parts given back", pr: 84, merge: "93b815d", heldout: { synthetic: 32.23, icons: 22.63 } },
  { version: 12, change: "Holes", pr: 85, merge: "eb9e056", heldout: { synthetic: 32.23, icons: 23.32 } },
  { version: 13, change: "Rounded rects, tight splines", pr: 86, merge: "e071651", heldout: { synthetic: 32.469, icons: 23.51 } },
  {
    version: 14,
    change: "A pale subject restored, two colours not a gradient",
    pr: 87,
    merge: "008378a",
    report: "copycat-ae2f606",
    heldout: { synthetic: 32.469, icons: 23.512 },
  },
  { version: 15, change: "Lost pieces", pr: 89, merge: "7bbc657", report: "copycat-leftovers", heldout: { synthetic: 32.469, icons: 23.604 } },
  {
    version: 16,
    change: "Two flat colours",
    pr: 90,
    merge: "065504a",
    report: "copycat-flat-colours-gated",
    heldout: { synthetic: 32.469, icons: 23.601 },
  },
  {
    version: 17,
    change: "An outline as seen",
    pr: 91,
    merge: "8c70818",
    report: "copycat-visible-final-heldout",
    heldout: { synthetic: 32.469, icons: 23.677 },
  },
  {
    version: 18,
    change: "Pale parts",
    pr: 97,
    merge: "f65ef66",
    report: "copycat-pale-parts",
    heldout: { synthetic: 32.469, icons: 23.949, broad: 26.845 },
  },
  {
    version: 19,
    change: "A flat background judged by its own spread",
    pr: 98,
    merge: "4ae1781",
    report: "copycat-flat-judged",
    heldout: { synthetic: 32.469, icons: 23.949, broad: 26.845 },
  },
  {
    version: 20,
    change: "A subject across both bottom corners",
    pr: 99,
    merge: "0434941",
    report: "copycat-background-misfit",
    heldout: { synthetic: 32.469, icons: 23.949, broad: 26.928 },
  },
  {
    version: 21,
    change: "Restored colours",
    pr: 100,
    merge: "153bdd5",
    report: "copycat-restored-colours",
    heldout: { synthetic: 32.469, icons: 24.494, broad: 26.92 },
    v2: 27.96,
  },
  {
    version: 22,
    change: "Polish",
    pr: 102,
    merge: "bd42308",
    report: "copycat-polish",
    heldout: { synthetic: 34.666, icons: 25.093, broad: 27.952 },
    v2: 29.24,
  },
  {
    version: 23,
    change: "A point where the outline needs one",
    pr: 104,
    merge: "8dc670c",
    report: "copycat-grow-turn",
    heldout: { synthetic: 34.719, icons: 25.537, broad: 28.366 },
    v2: 29.54,
  },
  {
    version: 24,
    change: "Pale fills kept apart",
    pr: 109,
    report: "copycat-pale-fills-2",
    heldout: { synthetic: 34.719, icons: 25.538, broad: 28.366 },
  },
  {
    version: 25,
    change: "Thin features kept",
    pr: 109,
    report: "copycat-thin-features-3",
    heldout: { synthetic: 34.719, icons: 25.797, broad: 28.382 },
    v2: 29.63,
  },
  {
    version: 26,
    change: "Small enclosed parts kept",
    pr: 109,
    report: "copycat-small-enclosed-2",
    heldout: { synthetic: 34.719, icons: 25.938, broad: 28.519 },
    v2: 29.73,
  },
  {
    version: 27,
    change: "Joins undone when the pieces say it far more briefly",
    pr: 109,
    report: "copycat-joined-or-apart",
    heldout: { synthetic: 34.719, icons: 25.95, broad: 28.519 },
    v2: 29.73,
  },
  {
    version: 28,
    change: "Thin features kept on near-noiseless images",
    pr: 109,
    report: "copycat-thin-near-noiseless",
    heldout: { synthetic: 34.719, icons: 25.95, broad: 28.549 },
    v2: 29.74,
  },
  {
    version: 29,
    change: "Outlines whose knots carry their own tangents",
    pr: 0,
    report: "copycat-hermite-outlines",
    heldout: { synthetic: 35.599, icons: 27.517, broad: 29.294 },
    v2: 30.8,
  },
]

export const latestCopyCatVersion = COPYCAT_VERSIONS.at(-1)!

/** The versions that moved one set's held-out score (by 0.01 dB or more), from the first that measured it: a chart of
 *  every version would be mostly flat steps. */
export function versionsThatMoved(set: keyof CopyCatVersion["heldout"]): (CopyCatVersion & { score: number })[] {
  const out: (CopyCatVersion & { score: number })[] = []
  for (const v of COPYCAT_VERSIONS) {
    const score = v.heldout[set]
    if (score === undefined) continue
    const last = out.at(-1)
    if (!last || Math.abs(score - last.score) >= 0.01) out.push({ ...v, score })
  }
  return out
}

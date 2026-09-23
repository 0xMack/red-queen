import SnakePlayStage from "~/components/SnakePlayStage.vue"
import SnakeWatchStage from "~/components/SnakeWatchStage.vue"
import type { GameDevice, GameModule } from "~/games/types"
import type { EvaluationRecord } from "~/types/leaderboard"
import type { DeviceFit } from "~/types/modelpack"

// Snake (docs/design/0007): single-agent, scored by food eaten on held-out seeds (jobs/evaluate.py).
// Its champions are ONNX model packages, so unlike a game whose entrants run anywhere, the page needs to
// know which of them *this device* can run (docs/design/0009).

/** Which entrants this device can run: a published champion needs a capable backend; baselines and
 *  unpublished champions run the Python/WASM code. */
function snakeDevice(slug: string, entries: Readonly<Ref<EvaluationRecord[]>>): GameDevice {
  const models = useModelCatalog(slug)
  const runsHere = computed<Record<string, DeviceFit>>(() => {
    const fits: Record<string, DeviceFit> = {}
    for (const r of entries.value) {
      const available = models.availability.value[r.entrant_id]
      const match = available?.match
      if (match?.ok) fits[r.entrant_id] = { ok: true, note: `${match.variant.id} · ${match.backend}` }
      else if (match) fits[r.entrant_id] = { ok: false, note: match.summary }
      else if (!available && models.catalog.value) fits[r.entrant_id] = { ok: true, note: "Python (not a published package)" }
    }
    return fits
  })
  const summary = computed(() => {
    const p = models.profile.value
    return p ? `${p.webgpu.available ? "WebGPU" : "no WebGPU"} · ${p.wasm ? `WASM${p.threads ? " threads" : ""}` : "no WASM"}` : null
  })
  return { load: models.load, runsHere, summary, filterable: computed(() => !!models.catalog.value), context: models }
}

export const snakeModule: GameModule = {
  slug: "snake",
  score: {
    label: "Held-out score",
    format: (v) => v.toFixed(2),
    compact: (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1)),
    scaleMin: 1,
  },
  columns: [
    {
      id: "spread",
      header: "Median · max · zero",
      title: "median / best / % of games scoring zero",
      cell: (r) => ({ text: `${r.metrics.quality.median} · ${r.metrics.quality.max} · ${Math.round(r.metrics.quality.zero_rate * 100)}%`, tone: "muted" }),
    },
    {
      id: "gap",
      header: "Train gap",
      title: "mean on its own training seeds minus held-out mean -- large = overfit",
      cell: (r) => {
        const gap = r.metrics.quality.generalization_gap
        return gap == null ? null : { text: formatSigned(gap, 1), tone: gap > 2 ? "warn" : "muted" }
      },
    },
  ],
  defaultEntrant: "best-trained",
  selectInPlay: "watch",
  copy: {
    watch: "Algorithms first: pick any entrant on the leaderboard to watch it play, then see if you can beat them.",
    play: "Same board, same rules, same game -- you're steering this time.",
    scoreNote: ({ episodes, protocol }) => `Mean score over ${episodes ?? 200} held-out games each (${protocol}).`,
    leaderboardIntro: ({ episodes, seeds }) =>
      `Ranked by mean score on ${episodes} held-out games (seeds ${seeds?.[0]}–${seeds?.[1]}) -- never by training fitness -- alongside what each cost to train and to run.`,
  },
  Watch: SnakeWatchStage,
  Play: SnakePlayStage,
  device: snakeDevice,
  initialHuman: (slug) => {
    const history = loadHumanHistory(slug)
    return history.games.length ? { score: history.best, label: "You (best)", live: false } : null
  },
  sections: { pareto: true, headToHead: false, representations: true },
}

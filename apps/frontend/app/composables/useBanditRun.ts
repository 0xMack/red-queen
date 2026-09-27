import init, { BanditRun, banditEvaluate } from "~/wasm/rl/rl.js"
import rlWasmUrl from "~/wasm/rl/rl_bg.wasm?url"
import { parseBeliefs, type Beliefs, type Reveal } from "~/utils/bandit"

// The bandit (docs/design/0011) in the browser: the Rust game and strategies compiled to WebAssembly, on the main
// thread -- a pull is microseconds, so a worker would only add latency. `useBanditRun` wraps one `BanditRun` (a game,
// and the strategy playing it or "human") in reactive state for a stage or a lab to draw; `loadBandit` is the one
// module load every bandit view shares.

let ready: Promise<void> | null = null
export function loadBandit(): Promise<void> {
  ready ??= init({ module_or_path: rlWasmUrl }).then(() => undefined)
  return ready
}

export interface BanditPull {
  arm: number
  reward: number
  /** The lamp the pull was made under. */
  lamp: number
}

export interface BanditRunSpec {
  scenario: string
  /** "none.v1" (one row) or "lamp.v1" (a row per lamp colour). */
  observer?: string
  /** A strategy id, or "human". */
  strategy: string
  params?: string
  seed: number
}

/** Runs a strategy on `count` games from `first`: per game [regret, efficiency, skill, best_rate], flattened. */
export async function evaluateBandit(strategy: string, params: string, scenario: string, observer: string, first: number, count: number) {
  await loadBandit()
  return banditEvaluate(strategy, params, scenario, observer, first, count)
}

export function useBanditRun() {
  let run: BanditRun | null = null
  const spec = shallowRef<BanditRunSpec | null>(null)
  const arms = ref(0)
  const budget = ref(0)
  const pulls = ref(0)
  const lamp = ref(0)
  const done = ref(false)
  const total = ref(0)
  const skill = ref(0)
  const regret = ref(0)
  const counts = shallowRef<number[]>([])
  const history = shallowRef<BanditPull[]>([])
  /** Beliefs per row (one row, or one per lamp colour); empty in a human game. */
  const beliefs = shallowRef<(Beliefs | null)[]>([])
  const reveal = shallowRef<Reveal | null>(null)
  /** Skill after each pull, for a race's curve. */
  const skillCurve = shallowRef<number[]>([])
  const error = ref<string | null>(null)

  function sync() {
    if (!run) return
    pulls.value = run.pulls
    lamp.value = run.lamp
    done.value = run.done
    total.value = run.total
    skill.value = run.skill
    regret.value = run.regret
    counts.value = Array.from(run.counts())
    const rows = spec.value?.observer === "lamp.v1" ? 2 : 1
    beliefs.value = Array.from({ length: rows }, (_, row) => parseBeliefs(run!.beliefs(row), run!.arms))
    if (run.done && !reveal.value) reveal.value = JSON.parse(run.reveal()) as Reveal
  }

  async function start(next: BanditRunSpec) {
    await loadBandit()
    run?.free()
    try {
      run = new BanditRun(next.scenario, next.observer ?? "none.v1", next.strategy, next.params ?? "", next.seed)
      error.value = null
    } catch (e) {
      run = null
      error.value = e instanceof Error ? e.message : String(e)
      return
    }
    spec.value = next
    arms.value = run.arms
    budget.value = run.budget
    history.value = []
    skillCurve.value = []
    reveal.value = null
    sync()
  }

  /** Pull `arm` (a human, or a strategy's own choice). Returns the pull, or null if the game is over. */
  function pull(arm: number): BanditPull | null {
    if (!run || run.done) return null
    const at = run.lamp
    const reward = run.pull(arm)
    const made = { arm, reward, lamp: at }
    history.value = [...history.value, made]
    sync()
    skillCurve.value = [...skillCurve.value, run.skill]
    return made
  }

  /** The strategy's next pick, without pulling (null for a human game). */
  function choose(): number | null {
    if (!run || run.done) return null
    const arm = run.choose()
    return arm < 0 ? null : arm
  }

  /** Let the strategy choose and pull. */
  function step(): BanditPull | null {
    const arm = choose()
    return arm === null ? null : pull(arm)
  }

  /** Each machine's true mean now -- only for the reveal. */
  const means = () => (run ? Array.from(run.means()) : [])
  const bestArm = () => run?.bestArm ?? 0
  const currentRow = () => run?.row ?? 0

  onUnmounted(() => {
    run?.free()
    run = null
  })

  return {
    spec,
    arms,
    budget,
    pulls,
    lamp,
    done,
    total,
    skill,
    regret,
    counts,
    history,
    beliefs,
    reveal,
    skillCurve,
    error,
    start,
    pull,
    choose,
    step,
    means,
    bestArm,
    currentRow,
  }
}

export type BanditRunState = ReturnType<typeof useBanditRun>

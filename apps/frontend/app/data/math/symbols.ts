import type { ExplainConcept } from "~/types/explain"

// The notation registry (docs/design/0012): every symbol the Learn chapters use, defined once, so α, γ, Q and ε mean
// one thing everywhere. A formula's term points at one of these for its card, and may reword it for its own context
// ("the estimate before the pull"). Each also becomes a `symbol:<id>` explainer (`?explain=symbol:alpha`).

export interface MathSymbol {
  /** How it's written, in LaTeX. */
  tex: string
  /** And in plain text, for titles and labels. */
  plain: string
  name: string
  /** One or two sentences: what it is and what changing it does. */
  meaning: string
  /** Its range, in words ("0 to 1"). */
  range?: string
  /** Chapters (slugs) that use it. */
  chapters?: string[]
}

export const SYMBOLS = {
  Q: {
    tex: "Q",
    plain: "Q",
    name: "estimated value",
    meaning:
      "What the player believes a choice is worth: for a machine, the average of what it has paid; in a Q-table, one cell per situation and move.",
    chapters: ["multi-armed-bandits", "q-learning", "dqn"],
  },
  r: {
    tex: "r",
    plain: "r",
    name: "reward",
    meaning: "What the last pull (or move) paid. The only feedback the player ever gets.",
    chapters: ["multi-armed-bandits", "q-learning"],
  },
  n: {
    tex: "n",
    plain: "n",
    name: "pull count",
    meaning: "How many times this machine has been pulled, this one included. A step of 1/n makes the estimate exactly the running average.",
    range: "1, 2, 3, …",
    chapters: ["multi-armed-bandits"],
  },
  N: {
    tex: "N",
    plain: "N",
    name: "pulls of a machine",
    meaning: "How many times a machine has been pulled so far. The more it has been tried, the smaller its exploration bonus.",
    chapters: ["multi-armed-bandits"],
  },
  t: {
    tex: "t",
    plain: "t",
    name: "pulls so far",
    meaning: "Total pulls in this game, across every machine. It grows slowly under the logarithm, so a machine left alone long enough earns its bonus back.",
    chapters: ["multi-armed-bandits"],
  },
  alpha: {
    tex: "\\alpha",
    plain: "α",
    name: "step size",
    meaning:
      "How far each new payout moves the estimate, as a fraction of the gap. Small: slow and steady. Large: quick to follow a change, and quick to be fooled by luck.",
    range: "0 to 1",
    chapters: ["multi-armed-bandits", "q-learning", "dqn"],
  },
  epsilon: {
    tex: "\\varepsilon",
    plain: "ε",
    name: "exploration rate",
    meaning: "The chance of ignoring the estimates and pulling a machine at random. 0 is greedy; 1 is pure chance.",
    range: "0 to 1",
    chapters: ["multi-armed-bandits", "q-learning"],
  },
  c: {
    tex: "c",
    plain: "c",
    name: "exploration weight",
    meaning: "How much UCB's uncertainty bonus counts against the estimate. 0 is greedy; larger means more willing to try what it knows little about.",
    range: "0 and up (√2 is the textbook choice)",
    chapters: ["multi-armed-bandits"],
  },
  gamma: {
    tex: "\\gamma",
    plain: "γ",
    name: "discount",
    meaning: "How much what comes next counts, compared with what pays now. 0: only the next payout. Near 1: the long run counts almost fully.",
    range: "0 to 1",
    chapters: ["multi-armed-bandits", "q-learning", "dqn", "policy-gradients"],
  },
  delta: {
    tex: "\\delta",
    plain: "δ",
    name: "TD error",
    meaning: "How wrong the estimate was: the target minus the estimate. Every update moves the estimate a step of α along it.",
    chapters: ["multi-armed-bandits", "q-learning", "dqn"],
  },
  theta: {
    tex: "\\theta",
    plain: "θ",
    name: "sampled win rate",
    meaning: "Thompson sampling's draw: one plausible win rate per machine, from its belief. The machine with the highest draw is pulled.",
    range: "0 to 1",
    chapters: ["multi-armed-bandits"],
  },
  a: {
    tex: "a",
    plain: "a",
    name: "action",
    meaning: "A choice: which machine to pull, which move to make.",
    chapters: ["multi-armed-bandits", "q-learning"],
  },
  s: {
    tex: "s",
    plain: "s",
    name: "situation",
    meaning: "What the player can observe before choosing -- the lamp, the room, Snake's 11 features. It picks the row of the table.",
    chapters: ["multi-armed-bandits", "q-learning"],
  },
  G: {
    tex: "G",
    plain: "G",
    name: "return",
    meaning:
      "Everything the agent will be paid from here on, later rewards discounted by γ per step. What an agent wants to be large -- not the next reward.",
    chapters: ["q-learning", "policy-gradients"],
  },
  nstep: {
    tex: "n",
    plain: "n",
    name: "n-step horizon",
    meaning: "How many real rewards to wait for before trusting the table's guess about the rest. 1 bootstraps at once; larger waits for more evidence.",
    range: "1 and up",
    chapters: ["q-learning", "dqn"],
  },
  weights: {
    tex: "\\theta",
    plain: "θ",
    name: "the network's weights",
    meaning: "Every weight and bias of the network, as one vector. Training moves them; everything the network computes depends on all of them at once.",
    chapters: ["dqn", "policy-gradients", "self-play"],
  },
  weightsTarget: {
    tex: "\\theta^-",
    plain: "θ⁻",
    name: "the target network's weights",
    meaning:
      "A frozen copy of θ, refreshed every few thousand steps. Targets computed from it hold still while the online network learns, instead of moving with every update.",
    chapters: ["dqn"],
  },
  loss: {
    tex: "L",
    plain: "L",
    name: "loss",
    meaning: "How wrong the network is, as one number: training takes steps that make it smaller.",
    chapters: ["dqn", "policy-gradients", "autodiff", "transformers"],
  },
  eta: {
    tex: "\\eta",
    plain: "η",
    name: "learning rate",
    meaning: "How big a step down the gradient each update takes. Too small and learning crawls; too large and it overshoots and can diverge.",
    chapters: ["dqn", "policy-gradients", "autodiff"],
  },
  pi: {
    tex: "\\pi",
    plain: "π",
    name: "policy",
    meaning: "The agent's rule for choosing, as probabilities: π(a | s) is the chance of move a in situation s. A policy-gradient method learns it directly.",
    chapters: ["policy-gradients"],
  },
  J: {
    tex: "J",
    plain: "J",
    name: "expected return",
    meaning: "How much the policy earns on average, from the start of a game: what policy-gradient methods climb.",
    chapters: ["policy-gradients"],
  },
  V: {
    tex: "V",
    plain: "V",
    name: "state value",
    meaning: "How good a situation is, before choosing: the return expected from s onwards, playing as the policy plays. Learned by the critic.",
    chapters: ["policy-gradients", "self-play"],
  },
  advantage: {
    tex: "\\hat A",
    plain: "Â",
    name: "advantage",
    meaning: "How much better a move turned out than the situation's usual: positive pushes the move up, negative pushes it down.",
    chapters: ["policy-gradients"],
  },
  lambda: {
    tex: "\\lambda",
    plain: "λ",
    name: "trace decay",
    meaning: "How far back one step's surprise reaches, or how much a return trusts real rewards over the critic's guess: 0 is one step, 1 is the whole episode.",
    range: "0 to 1",
    chapters: ["policy-gradients", "self-play"],
  },
  clip: {
    tex: "\\epsilon",
    plain: "ε",
    name: "clip range",
    meaning: "PPO's leash: once a move's probability has moved this fraction away from where it was when played, that sample stops pushing it.",
    range: "0.2 here",
    chapters: ["policy-gradients"],
  },
  ratio: {
    tex: "\\rho",
    plain: "ρ",
    name: "probability ratio",
    meaning: "How much more (or less) likely a move is now than when it was played: 1 means unchanged.",
    chapters: ["policy-gradients"],
  },
  sigma: {
    tex: "\\sigma",
    plain: "σ",
    name: "mutation step size",
    meaning: "The width of the bell curve every weight's nudge is drawn from. Too small and children are copies; too large and they're random networks.",
    chapters: ["neuroevolution", "teaching-a-snake"],
  },
  fitness: {
    tex: "F",
    plain: "F",
    name: "fitness",
    meaning: "How good a genome is, as the evolutionary algorithm scores it: the one number selection compares. Higher is better.",
    chapters: ["genetic-algorithms", "selection-strategies", "neuroevolution", "neat", "multi-agent-games"],
  },
} satisfies Record<string, MathSymbol>

export type SymbolId = keyof typeof SYMBOLS

/** The registry as explainer concepts: `?explain=symbol:alpha` opens the side panel on a symbol. */
export function symbolConcepts(): ExplainConcept[] {
  return Object.entries(SYMBOLS as Record<string, MathSymbol>).map(([id, sym]) => ({
    kind: "symbol" as const,
    id,
    title: `${sym.plain} · ${sym.name}`,
    family: "Notation",
    summary: sym.meaning,
    facts: sym.range ? [{ label: "range", value: sym.range }] : [],
    chapter: (sym.chapters ?? []).map((slug) => ({ slug })),
  }))
}

export function mathSymbol(id: string | undefined): MathSymbol | undefined {
  return id ? (SYMBOLS as Record<string, MathSymbol>)[id] : undefined
}

export type ChapterArtKind =
  | "ga-loop"
  | "selection"
  | "genomes"
  | "autodiff"
  | "attention"
  | "checkers"
  | "pipeline"
  | "neuroevolution"
  | "neat"
  | "q-table"
  | "q-network"
  | "policy"
  | "self-play"
  | "bandit"
  | "strength"
  | "league"

export interface LearnChapter {
  slug: string
  title: string
  // A few words, for the map and the sidebar.
  short: string
  summary: string
  path: string
  status: "available" | "coming-soon"
  // The path the chapter belongs to when it's reached without one (a search result, a link from a game page).
  home: LearnPathId
  readMinutes?: number
  tags: string[]
  // Cover visual: a real screenshot from the running app (public/screenshots/) or a diagram drawn by
  // ChapterArt.vue. `image` wins when both are set.
  image?: string
  art: ChapterArtKind
  // Chapters this one needs (slugs). Every path must put these before it, or list them in its `assumes`.
  prerequisites?: string[]
  // Chapters it refers back to or compares against without needing them -- "pairs well with", not "read first".
  related?: string[]
  // h2 headings in the chapter, in order -- indexed by the Learn search so a query can jump straight
  // to a section. Anchors are slugify(heading), the same ids learn.vue assigns at runtime; keep in
  // sync by hand when a chapter's headings change.
  sections?: string[]
}

// Every chapter, in no particular reading order: the order lives in `learnPaths` below, and the chapters form a graph
// (their `prerequisites`), not a line. "coming-soon" entries keep a path complete if a chapter is planned before it's
// written. This file has no numbered design doc; it's the Learn section's table of contents, not an architecture doc.
export const learnChapters: LearnChapter[] = [
  {
    slug: "genetic-algorithms",
    title: "Genetic Algorithms, from Scratch",
    short: "Genetic algorithms",
    summary:
      "Population, fitness, selection, variation, generations -- the core loop every technique in this project builds on.",
    path: "/learn/genetic-algorithms",
    status: "available",
    home: "evolution",
    readMinutes: 5,
    tags: ["evolution", "linear GP", "symbolic regression", "fitness"],
    art: "ga-loop",
    sections: ["A concrete run", "What that actually looks like"],
  },
  {
    slug: "selection-strategies",
    title: "Selection Strategies",
    short: "Selection",
    summary:
      "Tournament, Lexicase, and Pareto selection -- and the real tradeoffs between them, not just the definitions.",
    path: "/learn/selection-strategies",
    status: "available",
    home: "evolution",
    readMinutes: 6,
    tags: ["tournament", "lexicase", "pareto", "multi-objective", "diversity"],
    art: "selection",
    prerequisites: ["genetic-algorithms"],
    sections: ["Tournament Selection", "Lexicase Selection", "Pareto Selection"],
  },
  {
    slug: "genome-representations",
    title: "Genome Representations: Linear vs. Tree",
    short: "Genomes",
    summary: "Register-machine programs vs. Koza-style expression trees -- the same algorithms, a different genome shape.",
    path: "/learn/genome-representations",
    status: "available",
    readMinutes: 7,
    home: "evolution",
    tags: ["linear GP", "tree GP", "introns", "bloat"],
    art: "genomes",
    prerequisites: ["genetic-algorithms"],
    sections: [
      "Linear GP: a tiny register machine",
      "Introns: code that doesn't matter",
      "Tree GP: expressions as trees",
      "Head to head",
      "Same algorithms, any genome",
    ],
  },
  {
    slug: "neuroevolution",
    title: "Neuroevolution: Evolving a Network's Weights",
    short: "Neuroevolution",
    summary:
      "A network as one flat list of numbers, scored by playing and improved by Gaussian mutation -- the loop behind the Snake policy, opened up and made touchable.",
    path: "/learn/neuroevolution",
    status: "available",
    home: "evolution",
    readMinutes: 9,
    tags: ["neuroevolution", "evolution strategies", "mutation", "weight vector", "XOR"],
    art: "neuroevolution",
    prerequisites: ["genetic-algorithms", "genome-representations"],
    sections: [
      "A network is a list of numbers",
      "Fitness comes from playing",
      "Mutation is the whole search operator",
      "The whole loop, live",
      "Why there's no crossover",
      "What a fixed shape costs",
    ],
  },
  {
    slug: "neat",
    title: "NEAT: Evolving the Structure Too",
    short: "NEAT",
    summary:
      "Innovation numbers, crossover between different structures, and speciation -- how a network can start minimal and grow, tested against Snake with tracked experiments.",
    path: "/learn/neat",
    status: "available",
    home: "evolution",
    readMinutes: 12,
    tags: ["NEAT", "topology", "speciation", "innovation numbers", "crossover", "snake"],
    art: "neat",
    prerequisites: ["neuroevolution"],
    sections: [
      "A genome is a list of genes",
      "Adding a node without breaking anything",
      "Innovation numbers: telling genes apart",
      "Speciation: protecting new structure",
      "Putting it together, live",
      "NEAT against Snake",
      "What NEAT costs",
    ],
  },
  {
    slug: "multi-armed-bandits",
    title: "Multi-Armed Bandits: Explore or Exploit",
    short: "Bandits",
    summary:
      "Five slot machines, a hundred pulls: reinforcement learning with one situation and no future. Estimates, exploring on purpose -- ε-greedy, optimism, UCB, Thompson sampling -- which one wins depends on the game, the lamp that turns one row of values into a table, and the detour that needs a future (γ) -- Q-learning in miniature.",
    path: "/learn/multi-armed-bandits",
    status: "available",
    home: "rl",
    readMinutes: 13,
    tags: ["reinforcement learning", "bandits", "exploration", "UCB", "Thompson sampling", "contextual bandit", "discount"],
    art: "bandit",
    prerequisites: ["genetic-algorithms"],
    sections: [
      "One situation, five choices",
      "Keeping score: a table with one row",
      "Greedy, and why it fails",
      "Exploring on purpose",
      "Which strategy wins depends on the game",
      "When the world changes",
      "Letting evolution choose the settings",
      "Two lamps: when the situation matters",
      "The detour: when a pull changes what comes next",
      "From here to Q-learning",
    ],
  },
  {
    slug: "q-learning",
    title: "Reinforcement Learning: Q-learning",
    short: "Q-learning",
    summary:
      "One agent learning from its own experience: rewards, returns and the Bellman update, a Q-table for Snake training live in your browser -- and why it stops at the greedy baseline while evolution goes on to 38.",
    path: "/learn/q-learning",
    status: "available",
    home: "rl",
    readMinutes: 12,
    tags: ["reinforcement learning", "Q-learning", "SARSA", "Bellman equation", "exploration", "Markov", "snake"],
    art: "q-table",
    prerequisites: ["multi-armed-bandits"],
    related: ["neuroevolution"],
    sections: [
      "Learning from experience",
      "A value for every move in every situation",
      "The update rule",
      "Exploring on purpose",
      "Watch it learn",
      "What the knobs do",
      "Why it stops at the greedy baseline",
      "Where this goes next",
    ],
  },
  {
    slug: "dqn",
    title: "Deep Q-Networks",
    short: "Deep Q-networks",
    summary:
      "Q-learning with a neural network in place of the table: what generalizing between situations buys, why it can blow up, the fixes that stop it (replay, a target network) and the ones that barely matter here -- and the observation that lifts Snake from 19 to 30.",
    path: "/learn/dqn",
    status: "available",
    home: "rl",
    readMinutes: 14,
    tags: ["reinforcement learning", "DQN", "function approximation", "experience replay", "target network", "Double DQN", "snake"],
    art: "q-network",
    prerequisites: ["q-learning", "autodiff"],
    sections: [
      "From a table to a network",
      "The update, as a loss",
      "Watch it learn",
      "Why it can blow up",
      "What each fix buys",
      "What the snake sees",
      "Where this goes next",
    ],
  },
  {
    slug: "policy-gradients",
    title: "Policy Gradients",
    short: "Policy gradients",
    summary:
      "Learn the policy itself, not values: the policy-gradient theorem, why baselines matter, actor-critic, and why PPO clips -- the method that took Snake to 63, a Gaussian policy for continuous control, and gradients against evolution on the same network.",
    path: "/learn/policy-gradients",
    status: "available",
    home: "rl",
    readMinutes: 15,
    tags: ["reinforcement learning", "policy gradients", "REINFORCE", "actor-critic", "A2C", "PPO", "continuous control", "snake"],
    art: "policy",
    prerequisites: ["dqn"],
    sections: [
      "Learning the policy itself",
      "The policy-gradient theorem",
      "Baselines: the same gradient, less noise",
      "Actor-critic",
      "Why PPO clips",
      "Watch it learn",
      "Continuous actions",
      "Gradients against evolution, on the same network",
      "What the snake sees, again",
      "Where this goes next",
    ],
  },
  {
    slug: "self-play",
    title: "Learning by Self-Play",
    short: "Self-play",
    summary:
      "No teacher and no opponent but itself: a Checkers position evaluator that learns by TD(λ) from its own games, TD-Gammon style -- why the dice mattered for backgammon, what a deterministic game needs instead, and how it fares against evolution and search. Train one, then play it.",
    path: "/learn/self-play",
    status: "available",
    home: "rl",
    readMinutes: 13,
    tags: ["reinforcement learning", "self-play", "TD(λ)", "TD-Gammon", "checkers", "two-player games", "search"],
    art: "self-play",
    prerequisites: ["dqn", "multi-agent-games"],
    related: ["policy-gradients"],
    sections: [
      "Learning with nobody to learn from",
      "A value for every position",
      "TD(λ): learning from the game's own sequence",
      "What the dice did for backgammon",
      "Train one, then play it",
      "Against the field",
      "Where this goes next",
    ],
  },
  {
    slug: "measuring-strength",
    title: "Measuring Strength: Elo, Openings and Sequential Tests",
    short: "Measuring strength",
    summary:
      "When the best player stops losing, points per game can't tell it from a better one. Level openings played in pairs, Elo ratings fitted to a whole round robin, and a sequential test that stops as soon as the games have answered -- the yardstick every two-player training experiment here is judged by. Run the test yourself.",
    path: "/learn/measuring-strength",
    status: "available",
    home: "rl",
    readMinutes: 12,
    tags: ["evaluation", "Elo", "Bradley-Terry", "SPRT", "statistics", "checkers", "two-player games", "openings"],
    art: "strength",
    prerequisites: ["self-play"],
    related: ["multi-agent-games"],
    sections: [
      "The yardstick ran out",
      "Openings, played in pairs",
      "Elo: a scale for win probability",
      "Ratings from a round robin",
      "Is A stronger than B? Ask sequentially",
      "What it says about self-play",
      "What it can't tell you",
      "Where this goes next",
    ],
  },
  {
    slug: "training-regimes",
    title: "Who to Play: Pools, Leagues and Populations",
    short: "Who to play",
    summary:
      "A self-play learner's opponents are a choice: only itself, a pool of past selves, every past self, the ones it still can't beat -- or a population of learners whose settings evolve while their weights learn. Each measured head to head on Checkers, with the results that surprised.",
    path: "/learn/training-regimes",
    status: "available",
    home: "rl",
    readMinutes: 11,
    tags: ["reinforcement learning", "self-play", "fictitious self-play", "PFSP", "league", "population-based training", "checkers"],
    art: "league",
    prerequisites: ["self-play", "measuring-strength"],
    related: ["genetic-algorithms"],
    sections: [
      "Who does a self-play learner play?",
      "A pool of past selves",
      "Every past self: a league",
      "Prioritized opponents",
      "What the games said",
      "Population-based training",
      "Where this goes next",
    ],
  },
  {
    slug: "teaching-a-snake",
    title: "Teaching a Snake to Play Itself",
    short: "Teaching a snake",
    summary:
      "A case study in neuroevolution: a real representation bug, a real reward-hacking bug, and a 26x fitness improvement.",
    path: "/learn/teaching-a-snake",
    status: "available",
    home: "evolution",
    readMinutes: 7,
    tags: ["neuroevolution", "snake", "reward hacking", "observation design", "lexicase"],
    image: "/screenshots/snake-watch.png",
    art: "ga-loop",
    prerequisites: ["neuroevolution", "selection-strategies"],
    sections: [
      "Attempt one: show it the whole board",
      "Attempt two: hand it the structure directly",
      "A second bug, found the same way: by running it",
    ],
  },
  {
    slug: "autodiff",
    title: "Neural Networks, from Scratch",
    short: "Autodiff",
    summary: "Reverse-mode automatic differentiation, built without any ML framework -- the foundation under neuroevolution and transformers alike.",
    path: "/learn/autodiff",
    status: "available",
    readMinutes: 8,
    home: "neural-nets",
    tags: ["autodiff", "backpropagation", "computation graph", "gradient checking", "adam"],
    art: "autodiff",
    sections: [
      "The chain rule, as a graph",
      "A Tensor remembers how it was made",
      "The two places it's easy to get wrong",
      "Trust, but verify: gradient checking",
      "From an engine to a network",
    ],
  },
  {
    slug: "transformers",
    title: "Transformers, from Scratch",
    short: "Transformers",
    summary: "Attention, positional embeddings, and a character-level language model trained on real text.",
    path: "/learn/transformers",
    status: "available",
    readMinutes: 7,
    home: "neural-nets",
    tags: ["attention", "language model", "tinylm", "causal mask", "layer norm"],
    art: "attention",
    prerequisites: ["autodiff"],
    sections: [
      "The task: guess the next character",
      "Attention: every position asks every earlier one",
      "Blocks, residuals, and layer norm",
      "Training it",
    ],
  },
  {
    slug: "multi-agent-games",
    title: "Multi-Agent Games and the Strategy Framework",
    short: "Two-player games",
    summary: "Pitting any strategy against any strategy -- static heuristics, evolved genomes, and (eventually) classifiers, in checkers.",
    path: "/learn/multi-agent-games",
    status: "available",
    readMinutes: 7,
    home: "rl",
    tags: ["checkers", "match fitness", "minimax", "lookahead", "co-evolution"],
    art: "checkers",
    prerequisites: ["genetic-algorithms"],
    related: ["selection-strategies"],
    sections: [
      "From one agent to two",
      "A strategy is just a function",
      "What actually makes a player good",
      "Evolving a player",
      "What's next",
    ],
  },
  {
    slug: "real-time-architecture",
    title: "Watching a Population Evolve, Live",
    short: "Live architecture",
    summary: "The telemetry/SSE architecture behind this site's real-time run pages -- how live training gets from a Python process to your browser.",
    path: "/learn/real-time-architecture",
    status: "available",
    readMinutes: 6,
    home: "under-the-hood",
    tags: ["telemetry", "SSE", "FastAPI", "WebAssembly", "pause/resume"],
    image: "/screenshots/run-detail.png",
    art: "pipeline",
    sections: [
      "The job never knows about the web",
      "Three boring stores",
      "Backfill, then live",
      "Pausing a run without new plumbing",
      "Replaying champions in the browser",
    ],
  },
]

// The chapter that explains how a run's algorithm (its config.representation) works -- "read how it was trained".
const CHAPTER_FOR_REPRESENTATION: Record<string, string> = {
  linear_gp: "genome-representations",
  tree_gp: "genome-representations",
  neuroevolution: "neuroevolution",
  neat: "neat",
  q_learning: "q-learning",
  sarsa: "q-learning",
  dqn: "dqn",
  reinforce: "policy-gradients",
  a2c: "policy-gradients",
  ppo: "policy-gradients",
  td_lambda: "self-play",
}

export function chapterForRepresentation(representation: string): LearnChapter | null {
  const slug = CHAPTER_FOR_REPRESENTATION[representation]
  return learnChapters.find((c) => c.slug === slug) ?? null
}

// --- Learning paths ------------------------------------------------------------------------------------------------

export type LearnPathId = "evolution" | "rl" | "neural-nets" | "under-the-hood"

export interface LearnPath {
  id: LearnPathId
  title: string
  // One line: what you'll be able to explain at the end.
  goal: string
  summary: string
  // A palette token (utils/palette.ts) -- the path's line on the map, its marker everywhere else.
  color: "life400" | "signal400" | "violet400" | "gold400"
  chapters: string[] // slugs, in reading order
  // Chapters from other paths this one takes as read -- offered as "start here if you haven't" rather than as stops.
  assumes?: string[]
}

// Curated routes through the chapter graph, each towards one goal. A chapter can sit on several (autodiff is on both
// the RL and neural-net paths); where paths share a chapter, the map draws an interchange.
export const learnPaths: LearnPath[] = [
  {
    id: "evolution",
    title: "Evolution",
    goal: "Evolve programs and neural networks -- weights, then structure",
    summary:
      "The loop every technique here is measured against: a population, a fitness, selection and variation -- then the genome becomes a network, and then the network's shape evolves too.",
    color: "life400",
    chapters: ["genetic-algorithms", "selection-strategies", "genome-representations", "neuroevolution", "neat", "teaching-a-snake"],
  },
  {
    id: "rl",
    title: "Reinforcement learning",
    goal: "Learn from experience -- from one slot machine to self-play at Checkers",
    summary:
      "Exploration vs. exploitation, then values, then values from a network, then the policy itself -- and finally two players, a strategy framework, and an agent with nobody to learn from but itself.",
    color: "signal400",
    chapters: ["multi-armed-bandits", "q-learning", "autodiff", "dqn", "policy-gradients", "multi-agent-games", "self-play", "measuring-strength", "training-regimes"],
    assumes: ["genetic-algorithms"],
  },
  {
    id: "neural-nets",
    title: "Neural nets & language models",
    goal: "Build gradients and a transformer from nothing",
    summary: "Reverse-mode autodiff with no framework underneath, then a character-level transformer trained on it.",
    color: "violet400",
    chapters: ["autodiff", "transformers"],
  },
  {
    id: "under-the-hood",
    title: "Under the hood",
    goal: "How live training reaches your browser",
    summary: "The telemetry, streaming and in-browser inference that make every page on this site live.",
    color: "gold400",
    chapters: ["real-time-architecture"],
  },
]

export function chapterBySlug(slug: string): LearnChapter | undefined {
  return learnChapters.find((c) => c.slug === slug)
}

export function learnPath(id: string | null | undefined): LearnPath | undefined {
  return learnPaths.find((p) => p.id === id)
}

/** The paths a chapter is on, its home path first. */
export function pathsThrough(chapter: LearnChapter): LearnPath[] {
  return learnPaths.filter((p) => p.chapters.includes(chapter.slug)).sort((a, b) => Number(b.id === chapter.home) - Number(a.id === chapter.home))
}

/** The path a chapter is being read on: the requested one if the chapter is on it, else its home. */
export function resolvePath(chapter: LearnChapter, requested: string | null | undefined): LearnPath {
  const path = learnPath(requested)
  return path?.chapters.includes(chapter.slug) ? path : learnPath(chapter.home)!
}

/** A chapter's URL on a given path (its home path needs no query). */
export function chapterHref(chapter: LearnChapter, path?: LearnPath | null): string {
  return path && path.id !== chapter.home && path.chapters.includes(chapter.slug) ? `${chapter.path}?path=${path.id}` : chapter.path
}

/** Where the graph forks from here: the chapters that list this one as a prerequisite. */
export function chaptersBuildingOn(slug: string): LearnChapter[] {
  return learnChapters.filter((c) => c.prerequisites?.includes(slug))
}

export function pathMinutes(path: LearnPath): number {
  return path.chapters.reduce((sum, slug) => sum + (chapterBySlug(slug)?.readMinutes ?? 0), 0)
}

/** Every path must respect the prerequisite graph: each chapter's prerequisites come earlier on the path or are in its
 *  `assumes`. Returns the violations (empty when consistent) -- checked at module load in development. */
export function pathProblems(): string[] {
  const problems: string[] = []
  for (const chapter of learnChapters) {
    if (!learnPath(chapter.home)?.chapters.includes(chapter.slug)) problems.push(`${chapter.slug}: not on its home path "${chapter.home}"`)
    for (const slug of [...(chapter.prerequisites ?? []), ...(chapter.related ?? [])]) {
      if (!chapterBySlug(slug)) problems.push(`${chapter.slug}: unknown chapter "${slug}"`)
    }
  }
  for (const path of learnPaths) {
    path.chapters.forEach((slug, i) => {
      const chapter = chapterBySlug(slug)
      if (!chapter) return problems.push(`${path.id}: unknown chapter "${slug}"`)
      for (const pre of chapter.prerequisites ?? []) {
        if (!path.chapters.slice(0, i).includes(pre) && !path.assumes?.includes(pre)) problems.push(`${path.id}: "${slug}" comes before its prerequisite "${pre}"`)
      }
    })
  }
  return problems
}

if (import.meta.dev) {
  for (const problem of pathProblems()) console.warn(`learnPaths: ${problem}`)
}

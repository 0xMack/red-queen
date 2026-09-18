export type ChapterArtKind = "ga-loop" | "selection" | "genomes" | "autodiff" | "attention" | "checkers" | "pipeline"

export interface LearnChapter {
  slug: string
  title: string
  summary: string
  path: string
  status: "available" | "coming-soon"
  part: string
  readMinutes?: number
  tags: string[]
  // Cover visual: a real screenshot from the running app (public/screenshots/) or a diagram drawn by
  // ChapterArt.vue. `image` wins when both are set.
  image?: string
  art: ChapterArtKind
  prerequisites?: string[] // slugs
  // h2 headings in the chapter, in order -- indexed by the Learn search so a query can jump straight
  // to a section. Anchors are slugify(heading), the same ids learn.vue assigns at runtime; keep in
  // sync by hand when a chapter's headings change.
  sections?: string[]
}

// Foundations-first order -- each chapter builds on the ones before it, same shape as
// docs/design/0003's incremental roadmap. "coming-soon" entries keep the index complete/navigable
// even though only 3 chapters have real content so far -- see docs/design (this file has no
// numbered doc of its own; it's the Learn section's table of contents, not an architecture doc).
export const learnChapters: LearnChapter[] = [
  {
    slug: "genetic-algorithms",
    title: "Genetic Algorithms, from Scratch",
    summary:
      "Population, fitness, selection, variation, generations -- the core loop every technique in this project builds on.",
    path: "/learn/genetic-algorithms",
    status: "available",
    part: "Foundations",
    readMinutes: 5,
    tags: ["evolution", "linear GP", "symbolic regression", "fitness"],
    art: "ga-loop",
    sections: ["A concrete run", "What that actually looks like"],
  },
  {
    slug: "selection-strategies",
    title: "Selection Strategies",
    summary:
      "Tournament, Lexicase, and Pareto selection -- and the real tradeoffs between them, not just the definitions.",
    path: "/learn/selection-strategies",
    status: "available",
    part: "Foundations",
    readMinutes: 6,
    tags: ["tournament", "lexicase", "pareto", "multi-objective", "diversity"],
    art: "selection",
    prerequisites: ["genetic-algorithms"],
    sections: ["Tournament Selection", "Lexicase Selection", "Pareto Selection"],
  },
  {
    slug: "genome-representations",
    title: "Genome Representations: Linear vs. Tree",
    summary: "Register-machine programs vs. Koza-style expression trees -- the same algorithms, a different genome shape.",
    path: "/learn/genome-representations",
    status: "coming-soon",
    part: "Foundations",
    tags: ["linear GP", "tree GP", "introns", "bloat"],
    art: "genomes",
    prerequisites: ["genetic-algorithms"],
  },
  {
    slug: "teaching-a-snake",
    title: "Teaching a Snake to Play Itself",
    summary:
      "A case study in neuroevolution: a real representation bug, a real reward-hacking bug, and a 26x fitness improvement.",
    path: "/learn/teaching-a-snake",
    status: "available",
    part: "Case studies",
    readMinutes: 7,
    tags: ["neuroevolution", "snake", "reward hacking", "observation design", "lexicase"],
    image: "/screenshots/snake-watch.png",
    art: "ga-loop",
    prerequisites: ["genetic-algorithms", "selection-strategies"],
    sections: [
      "Attempt one: show it the whole board",
      "Attempt two: hand it the structure directly",
      "A second bug, found the same way: by running it",
    ],
  },
  {
    slug: "autodiff",
    title: "Neural Networks, from Scratch",
    summary: "Reverse-mode automatic differentiation, built without any ML framework -- the foundation under neuroevolution and transformers alike.",
    path: "/learn/autodiff",
    status: "coming-soon",
    part: "Gradients",
    tags: ["autodiff", "backpropagation", "computation graph"],
    art: "autodiff",
  },
  {
    slug: "transformers",
    title: "Transformers, from Scratch",
    summary: "Attention, positional embeddings, and a character-level language model trained on real text.",
    path: "/learn/transformers",
    status: "coming-soon",
    part: "Gradients",
    tags: ["attention", "language model", "tinylm"],
    art: "attention",
    prerequisites: ["autodiff"],
  },
  {
    slug: "multi-agent-games",
    title: "Multi-Agent Games and the Strategy Framework",
    summary: "Pitting any strategy against any strategy -- static heuristics, evolved genomes, and (eventually) classifiers, in checkers.",
    path: "/learn/multi-agent-games",
    status: "coming-soon",
    part: "Case studies",
    tags: ["checkers", "match fitness", "co-evolution"],
    art: "checkers",
  },
  {
    slug: "real-time-architecture",
    title: "Watching a Population Evolve, Live",
    summary: "The telemetry/SSE architecture behind this site's real-time run pages -- how live training gets from a Python process to your browser.",
    path: "/learn/real-time-architecture",
    status: "coming-soon",
    part: "Under the hood",
    tags: ["telemetry", "SSE", "FastAPI", "Pyodide"],
    image: "/screenshots/run-detail.png",
    art: "pipeline",
  },
]

export function chapterNumber(slug: string): number {
  return learnChapters.findIndex((c) => c.slug === slug) + 1
}

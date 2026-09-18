export interface LearnChapter {
  title: string
  summary: string
  path: string
  status: "available" | "coming-soon"
}

// Foundations-first order -- each chapter builds on the ones before it, same shape as
// docs/design/0003's incremental roadmap. "coming-soon" entries keep the index complete/navigable
// even though only 3 chapters have real content in this pass -- see docs/design (this file has no
// numbered doc of its own; it's the Learn section's table of contents, not an architecture doc).
export const learnChapters: LearnChapter[] = [
  {
    title: "Genetic Algorithms, from Scratch",
    summary:
      "Population, fitness, selection, variation, generations -- the core loop every technique in this project builds on.",
    path: "/learn/genetic-algorithms",
    status: "available",
  },
  {
    title: "Selection Strategies",
    summary:
      "Tournament, Lexicase, and Pareto selection -- and the real tradeoffs between them, not just the definitions.",
    path: "/learn/selection-strategies",
    status: "available",
  },
  {
    title: "Genome Representations: Linear vs. Tree",
    summary: "Register-machine programs vs. Koza-style expression trees -- the same algorithms, a different genome shape.",
    path: "/learn/genome-representations",
    status: "coming-soon",
  },
  {
    title: "Teaching a Snake to Play Itself",
    summary:
      "A case study in neuroevolution: a real representation bug, a real reward-hacking bug, and a 26x fitness improvement.",
    path: "/learn/teaching-a-snake",
    status: "available",
  },
  {
    title: "Neural Networks, from Scratch",
    summary: "Reverse-mode automatic differentiation, built without any ML framework -- the foundation under neuroevolution and transformers alike.",
    path: "/learn/autodiff",
    status: "coming-soon",
  },
  {
    title: "Transformers, from Scratch",
    summary: "Attention, positional embeddings, and a character-level language model trained on real text.",
    path: "/learn/transformers",
    status: "coming-soon",
  },
  {
    title: "Multi-Agent Games and the Strategy Framework",
    summary: "Pitting any strategy against any strategy -- static heuristics, evolved genomes, and (eventually) classifiers, in checkers.",
    path: "/learn/multi-agent-games",
    status: "coming-soon",
  },
  {
    title: "Watching a Population Evolve, Live",
    summary: "The telemetry/SSE architecture behind this site's real-time run pages -- how live training gets from a Python process to your browser.",
    path: "/learn/real-time-architecture",
    status: "coming-soon",
  },
]

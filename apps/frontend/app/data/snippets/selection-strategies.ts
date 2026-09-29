import type { Snippet } from "~/types/code"

// Selection Strategies' snippets. Python is libs/evolve's own; Rust is a translation (the selection strategies exist
// only in Python), written in `rand`-crate style.

export const tournament: Snippet = {
  pseudo: `function tournament_select(population, case_fitnesses, k)
    contestants ← k individuals drawn uniformly at random (with replacement)
    return the contestant with the highest mean fitness across all cases`,
  python: {
    source: "libs/evolve/src/evolve/selection.py",
    code: `class TournamentSelection:
    """Selects the fittest (by mean fitness across cases) of \`k\`
    uniformly-random individuals."""

    def __init__(self, k: int = 3):
        self._k = k

    def select(self, population, case_fitnesses, rng):
        indices = [rng.randrange(len(population)) for _ in range(self._k)]
        best_idx = max(indices, key=lambda i: statistics.fmean(case_fitnesses[i]))
        return population[best_idx]`,
  },
  rust: `pub struct TournamentSelection {
    k: usize,
}

impl TournamentSelection {
    pub fn select<'a, G>(&self, population: &'a [G], case_fitnesses: &[Vec<f64>], rng: &mut impl Rng) -> &'a G {
        let mean = |i: usize| case_fitnesses[i].iter().sum::<f64>() / case_fitnesses[i].len() as f64;
        let best = (0..self.k)
            .map(|_| rng.gen_range(0..population.len()))
            .max_by(|&a, &b| mean(a).total_cmp(&mean(b)))
            .unwrap();
        &population[best]
    }
}`,
}

export const lexicase: Snippet = {
  pseudo: `function lexicase_select(population, case_fitnesses)
    candidates ← every individual
    for each case in the test cases, in a random order
        if only one candidate is left then break
        best ← the highest fitness any candidate scores on this case
        ε ← median absolute deviation of the candidates' scores on this case
        candidates ← the candidates scoring at least best − ε on this case
    return a random one of the candidates`,
  python: {
    source: "libs/evolve/src/evolve/selection.py",
    code: `class LexicaseSelection:
    """Filters candidates case-by-case, in random order, keeping only
    individuals within epsilon of the best remaining fitness on each
    case -- until one candidate remains or every case is used."""

    def select(self, population, case_fitnesses, rng):
        candidates = list(range(len(population)))
        cases = list(range(len(case_fitnesses[0])))
        rng.shuffle(cases)

        for case in cases:
            if len(candidates) == 1:
                break
            values = [case_fitnesses[i][case] for i in candidates]
            best = max(values)
            epsilon = self._epsilon_for(values)
            candidates = [i for i in candidates if case_fitnesses[i][case] >= best - epsilon]

        return population[rng.choice(candidates)]`,
  },
  rust: `pub struct LexicaseSelection {
    epsilon: Option<f64>, // None: the median absolute deviation of each case's values
}

impl LexicaseSelection {
    pub fn select<'a, G>(&self, population: &'a [G], case_fitnesses: &[Vec<f64>], rng: &mut impl Rng) -> &'a G {
        let mut candidates: Vec<usize> = (0..population.len()).collect();
        let mut cases: Vec<usize> = (0..case_fitnesses[0].len()).collect();
        cases.shuffle(rng);

        for case in cases {
            if candidates.len() == 1 {
                break;
            }
            let values: Vec<f64> = candidates.iter().map(|&i| case_fitnesses[i][case]).collect();
            let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let epsilon = self.epsilon.unwrap_or_else(|| median_absolute_deviation(&values));
            candidates.retain(|&i| case_fitnesses[i][case] >= best - epsilon);
        }

        &population[*candidates.choose(rng).unwrap()]
    }
}`,
}

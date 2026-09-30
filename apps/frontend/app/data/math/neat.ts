import { add, formula, frac, mul, seq, sub, term } from "~/utils/math/expr"

// NEAT's formulas (docs/design/0012): the compatibility distance, worked live in the speciation demo from its sliders
// and the two parents' genes (and checked there against the demo's own distance), the species rule with its
// threshold, and fitness sharing.

const coef = (id: string, tex: string, name: string, meaning: string) => term(id, tex, { name, meaning, num: { decimals: 1, max: 9 } })
const count = (id: string, tex: string, name: string, meaning: string) => term(id, tex, { name, meaning, num: { decimals: 0, max: 99 } })

const c1 = coef("c1", "c_1", "the excess weight", "How much each excess gene -- one past the other genome's newest -- counts.")
const c2 = coef("c2", "c_2", "the disjoint weight", "How much each disjoint gene -- missing from the other genome, within its range -- counts.")
const c3 = coef("c3", "c_3", "the weight-difference weight", "How much differences in the weights of shared genes count.")
const E = count("excess", "E", "excess genes", "Genes one genome has beyond the other's highest innovation number.")
const D = count("disjoint", "D", "disjoint genes", "Genes one genome has and the other lacks, inside the range both cover.")
const N = count("n", "N", "the larger genome's gene count", "1 for genomes under 20 genes: small genomes aren't size-normalized.")
const W = term("wbar", "\\bar W", { name: "the mean weight difference", meaning: "Averaged over the genes both genomes share (matching innovation numbers).", num: { decimals: 2, max: 9 } })
const part = { decimals: 2, max: 99 }

const distanceRhs = add(
  term("excess-part", frac(mul(c1, E), N), { name: "the excess part", num: part }),
  term("disjoint-part", frac(mul(c2, D), N), { name: "the disjoint part", num: part }),
  term("weight-part", mul(c3, W), { name: "the weight part", num: part }),
)

export const distance = formula({
  id: "neat-distance",
  title: "The compatibility distance: excess and disjoint genes, normalized by genome size, plus the mean weight difference",
  lhs: term("delta", "\\delta", { name: "the compatibility distance", meaning: "How different two genomes are: 0 for identical ones.", num: { decimals: 2, max: 99 } }),
  body: distanceRhs,
  worked: distanceRhs,
  result: "delta",
})

export const sameSpecies = formula({
  id: "neat-species",
  title: "A genome joins the first species whose representative is closer than the threshold",
  lhs: seq("\\text{same species}"),
  rel: "\\iff",
  body: seq(term("delta", "\\delta", { name: "the compatibility distance" }), "<", term("threshold", "\\delta_t", { name: "the threshold", meaning: "Closer than this to a species' representative, and a genome joins it; farther from all of them, and it founds a new one." })),
})

const shifted = term("f-shifted", "f'_i", {
  name: "a member's shifted fitness",
  meaning: "Snake's fitness can be negative, so every fitness is shifted up by the lowest one first: f'ᵢ = fᵢ − min f + 10⁻⁶.",
})

export const sharing = formula({
  id: "neat-sharing",
  title: "Fitness sharing: a species' claim on the next generation is its members' mean shifted fitness",
  lhs: term("share", "\\text{share}(s)", { name: "the species' claim", meaning: "Offspring are split between species in proportion to these." }),
  body: seq(frac("1", term("size", "|s|", { name: "the species' size" })), "\\sum_{i \\in s}", shifted),
})

export const shift = formula({
  id: "neat-shift",
  title: "Each fitness is shifted so the lowest is just above zero",
  lhs: shifted,
  body: add(sub(term("f", "f_i", { symbol: "fitness", name: "a member's fitness" }), "\\min_j f_j"), "10^{-6}"),
})

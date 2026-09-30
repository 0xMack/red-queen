import { add, formula, frac, mul, paren, pow, seq, sub, tanh, term } from "~/utils/math/expr"

// Neuroevolution's formulas (docs/design/0012): a neuron, the explorer's XOR output neuron worked live from the genome
// (its cells 9-12 and the hidden units' activations for the selected row), XOR fitness, and Gaussian mutation with σ
// linked to the microscope's slider.

export const neuron = formula({
  id: "neuro-neuron",
  title: "A unit's output: tanh of its bias plus every input times its weight",
  lhs: term("a-o", "a_o", { name: "this unit's output" }),
  body: tanh(
    add(
      term("b-o", "b_o", { name: "its bias", meaning: "Stored after the layer's weights in the flat list." }),
      seq("\\sum_k", mul(term("w-ok", "w_{ok}", { name: "the weight from input k", meaning: "genome[offset + o·n_in + k]: the list, read by position." }), term("a-k", "a_k", { name: "input k" }))),
    ),
  ),
})

const value = { decimals: 2, max: 9, signed: true }
const w = (i: number, index: number) => term(`w-out-${i}`, `w_{${i}}`, { name: `genome[${index}]: the weight from hidden ${i} into the output`, num: value })
const h = (i: number) => term(`h-${i}`, `h_{${i}}`, { name: `hidden ${i}'s activation, for this input`, num: value })

export const xorOutput = formula({
  id: "neuro-xor-output",
  title: "The XOR network's output: tanh of its bias plus each hidden activation times its weight",
  lhs: term("y", "y", { name: "the network's output", num: value }),
  body: tanh(add(term("b-out", "b", { name: "genome[12]: the output's bias", num: value }), mul(w(0, 9), h(0)), mul(w(1, 10), h(1)), mul(w(2, 11), h(2)))),
  worked: tanh(add(term("b-out", "b", { name: "genome[12]: the output's bias", num: value }), mul(w(0, 9), h(0)), mul(w(1, 10), h(1)), mul(w(2, 11), h(2)))),
  result: "y",
})

export const xorFitness = formula({
  id: "neuro-xor-fitness",
  title: "XOR fitness: one minus the squared error on each of the four cases, averaged",
  lhs: term("fitness", "F", { symbol: "fitness", name: "the genome's XOR fitness", meaning: "1.0 when all four cases are right; a network without a hidden layer can't get past 0.75." }),
  body: seq(
    frac("1", "4"),
    "\\sum_{(x,\\,t)}",
    paren(
      sub(
        "1",
        pow(sub(term("y01", frac(add("y(x)", "1"), "2"), { name: "the output, rescaled from [−1, 1] to [0, 1]" }), term("xor-target", "t", { name: "the right answer: x₀ XOR x₁" })), "2"),
      ),
    ),
  ),
})

export const mutation = formula({
  id: "neuro-mutation",
  title: "Gaussian mutation: every weight nudged by sigma times a standard normal draw",
  lhs: term("w-child", "w_i'", { name: "the child's weight" }),
  body: add(term("w-parent", "w_i", { name: "the parent's weight" }), mul(term("sigma", "\\sigma", { symbol: "sigma", num: { decimals: 3, max: 9 } }), term("noise", "\\varepsilon_i", { name: "a fresh standard-normal draw", meaning: "Independent for every weight, every child: mean 0, standard deviation 1." }))),
})

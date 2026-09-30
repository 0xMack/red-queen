import { cases, formula, frac, seq, term } from "~/utils/math/expr"

// Genome Representations' formulas (docs/design/0012): what one linear-GP instruction does -- every field taken modulo
// its range, which is why any random integers make a valid program -- and the protected division that keeps every
// program running.

export const instruction = formula({
  id: "gp-instruction",
  title: "One instruction: apply its operator to a register and a register-or-input, and write the result to a register",
  lhs: seq("r_{", term("dst", "d \\bmod R", { name: "the destination register", meaning: "Any integer works: it's taken modulo the register count." }), "}"),
  rel: "\\leftarrow",
  body: seq(
    term("op", "\\mathrm{op}_{\\,o \\bmod 4}", { name: "the operator", meaning: "+, −, × or protected ÷ -- the op index, modulo 4." }),
    "\\big(",
    term("src-a", "r_{a \\bmod R}", { name: "the first operand: a register" }),
    ",\\;",
    term("src-b", "v_{b \\bmod (R + I)}", { name: "the second operand: a register or an input", meaning: "Indices past the registers read the inputs: v = (r₀, …, r_{R−1}, x₀, …)." }),
    "\\big)",
  ),
})

export const protectedDivision = formula({
  id: "gp-protected-division",
  title: "Protected division: one, instead of dividing by nearly zero",
  lhs: seq("a \\div b"),
  body: cases(
    [term("real-div", frac("a", "b"), { name: "ordinary division" }), "|b| > 10^{-6}"],
    [term("div-fallback", "1", { name: "the fallback", meaning: "A random program divides by zero constantly; returning 1 keeps it running instead of producing infinities." }), "\\text{otherwise}"],
  ),
})

import { add, formula, frac, mul, tex, term } from "~/utils/math/expr"

// The tracing chapter's formulas (docs/design/0022, Decisions 3 and 4): clean fidelity, the description a part is chosen
// by, and what a shape's description costs. The worked numbers are the pipeline's own costs at 128 px (`Costs::new`).

const black = term("black", "\\text{PSNR}_{\\text{black}}", { name: "PSNR of the two renders over black", meaning: "A leftover background shows up here: the clean render is transparent there, so black." })
const white = term("white", "\\text{PSNR}_{\\text{white}}", { name: "PSNR of the two renders over white", meaning: "A hole or a white halo shows up here instead." })

export const fidelity = formula({
  id: "copycat-fidelity",
  title: "Clean fidelity: how close the output's render is to the clean render, over black and over white",
  lhs: term("fidelity", "\\text{fidelity}", { name: "clean fidelity, in dB" }),
  body: frac(add(black, white), tex("2")),
})

const model = term("model", "\\text{bits}(\\text{shape})", {
  name: "the bits to write the shape down",
  meaning: "Its type and its coordinates. A circle is three numbers, however large it is.",
  num: { decimals: 0, max: 999 },
})
const pixel = term("pixel", "b_{\\text{px}}", {
  name: "the bits a wrong pixel costs",
  meaning: "12 in the pipeline. 8 was 0.6 dB worse, and 16 made outlines more jagged.",
  num: { decimals: 0, max: 99 },
})
const wrong = term("wrong", "n_{\\text{wrong}}", { name: "the pixels the shape gets wrong", num: { decimals: 0, max: 999 } })

const description = add(model, mul(pixel, wrong))

export const length = formula({
  id: "copycat-description",
  title: "A shape's description length: the bits to write it down, plus the bits to correct what it gets wrong",
  lhs: term("L", "L", { name: "the shape's description length, in bits", num: { decimals: 0, max: 9999 } }),
  body: description,
  worked: description,
  result: "L",
})

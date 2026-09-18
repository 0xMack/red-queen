// Parses a linear-GP champion artifact -- the Python repr() of evolve.genome.LinearProgram that
// jobs/baseline_gp_run.py stores -- into readable register-machine instructions, with structural
// introns marked. Mirrors LinearProgram.run()/effective_instruction_count() in
// libs/evolve/src/evolve/genome.py exactly (the same modulo indexing), so what's displayed is what
// actually executes. Returns null for anything that doesn't look like that repr.

export interface ParsedInstruction {
  text: string
  dst: number
  effective: boolean
}

export interface ParsedProgram {
  instructions: ParsedInstruction[]
  numRegisters: number
  numInputs: number
  effectiveCount: number
}

const OP_SYMBOLS: Record<string, string> = { op_add: "+", op_sub: "-", op_mul: "×", op_div: "÷" }

export function parseLinearProgram(text: string): ParsedProgram | null {
  const regs = /num_registers=(\d+)/.exec(text)
  const inputs = /num_inputs=(\d+)/.exec(text)
  if (!regs || !inputs) return null
  const numRegisters = Number(regs[1])
  const numInputs = Number(inputs[1])

  const opsMatch = /ops=\((.*)\)\)?$/.exec(text)
  const opNames = opsMatch ? [...opsMatch[1]!.matchAll(/<function (\w+)/g)].map((m) => m[1]!) : []
  if (opNames.length === 0) return null

  const raw = [...text.matchAll(/Instruction\(op=(\d+), dst=(\d+), src_a=(\d+), src_b=(\d+)\)/g)].map((m) => ({
    op: Number(m[1]) % opNames.length,
    dst: Number(m[2]) % numRegisters,
    a: Number(m[3]) % numRegisters,
    b: Number(m[4]) % (numRegisters + numInputs),
  }))
  if (raw.length === 0) return null

  // Backward liveness from output register r0.
  const live = new Set([0])
  const effective: boolean[] = new Array(raw.length).fill(false)
  for (let i = raw.length - 1; i >= 0; i--) {
    const instr = raw[i]!
    if (!live.has(instr.dst)) continue
    effective[i] = true
    live.delete(instr.dst)
    live.add(instr.a)
    if (instr.b < numRegisters) live.add(instr.b)
  }

  const operand = (i: number) => (i < numRegisters ? `r${i}` : `x${i - numRegisters}`)
  const instructions = raw.map((instr, i) => {
    const name = opNames[instr.op]!
    const symbol = OP_SYMBOLS[name] ?? name
    return {
      text: `r${instr.dst} = ${operand(instr.a)} ${symbol} ${operand(instr.b)}`,
      dst: instr.dst,
      effective: effective[i]!,
    }
  })

  return { instructions, numRegisters, numInputs, effectiveCount: effective.filter(Boolean).length }
}

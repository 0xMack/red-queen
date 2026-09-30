import type { Snippet } from "~/types/code"

// Genome Representations' snippets. Python is libs/evolve's own; Rust is a translation (both genomes exist only in
// Python here -- RedQueenCbind's engine is C++).

export const linearRun: Snippet = {
  pseudo: `function run(program, inputs)
    registers ← [0, 0, …, 0]
    for each (op, dst, a, b) in program.instructions
        // b can name a register or an input: indices past the registers read the inputs
        registers[dst] ← op(registers[a], read(b))
    return registers    // the answer is whatever ends up in register 0`,
  python: {
    source: "libs/evolve/src/evolve/genome.py",
    code: `def run(self, inputs: Sequence[float]) -> list[float]:
    registers = [0.0] * self.num_registers
    for instr in self.instructions:
        a = registers[instr.src_a % self.num_registers]
        b_idx = instr.src_b % (self.num_registers + self.num_inputs)
        b = registers[b_idx] if b_idx < self.num_registers else inputs[b_idx - self.num_registers]
        op = self.ops[instr.op % len(self.ops)]
        registers[instr.dst % self.num_registers] = op(a, b)
    return registers`,
  },
  rust: `pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
    let (nr, ni) = (self.num_registers, self.num_inputs);
    let mut registers = vec![0.0; nr];
    for instr in &self.instructions {
        let a = registers[instr.src_a % nr];
        let b_idx = instr.src_b % (nr + ni);
        let b = if b_idx < nr { registers[b_idx] } else { inputs[b_idx - nr] };
        let op = self.ops[instr.op % self.ops.len()];
        registers[instr.dst % nr] = op(a, b);
    }
    registers
}`,
}

export const tree: Snippet = {
  pseudo: `// A tree is a terminal -- the input x or a constant -- or an operator over two subtrees.
function evaluate(node, x)
    if node is a terminal then
        return x if node is the input, otherwise node's constant
    return node.op(evaluate(node.left, x), evaluate(node.right, x))`,
  python: {
    source: "libs/evolve/src/evolve/tree.py",
    code: `@dataclass(frozen=True, slots=True)
class Terminal:
    value: float | None  # None means "the input variable x"; otherwise a random constant

    def evaluate(self, x, ops=DEFAULT_OPS):
        return x if self.value is None else self.value


@dataclass(frozen=True, slots=True)
class FunctionNode:
    op: int  # index into ops
    left: Node
    right: Node

    def evaluate(self, x, ops=DEFAULT_OPS):
        return ops[self.op % len(ops)](self.left.evaluate(x), self.right.evaluate(x))`,
  },
  rust: `pub enum Node {
    Terminal(Option<f64>), // None means "the input variable x"; otherwise a random constant
    Function { op: usize, left: Box<Node>, right: Box<Node> },
}

impl Node {
    pub fn evaluate(&self, x: f64, ops: &[Op]) -> f64 {
        match self {
            Node::Terminal(value) => value.unwrap_or(x),
            Node::Function { op, left, right } => ops[op % ops.len()](left.evaluate(x, ops), right.evaluate(x, ops)),
        }
    }
}`,
}

export const crossover: Snippet = {
  pseudo: `function vary(a, b)
    child ← a with one random subtree replaced by a random subtree of b
    with probability mutation_rate
        replace a random subtree of child with a freshly grown one
    if depth(child) > max_tree_depth then return a    // bloat control
    return child`,
  python: {
    source: "libs/evolve/src/evolve/tree.py",
    code: `def vary(self, parents, rng):
    a, b = parents[0], parents[1]
    # swap a random subtree of a for a random subtree of b...
    crossover_point = rng.randrange(len(_flatten(a.root)))
    donor = rng.choice(_flatten(b.root))
    child_root = _replace_at(a.root, crossover_point, donor)
    # ...then, sometimes, replace a random subtree with a fresh random one
    if rng.random() < self._mutation_rate:
        ...
    # bloat control: an offspring deeper than the limit is thrown away
    if _depth(child_root) > self._max_tree_depth:
        return a
    return replace(a, root=child_root)`,
  },
  rust: `pub fn vary(&self, a: &TreeProgram, b: &TreeProgram, rng: &mut impl Rng) -> TreeProgram {
    // swap a random subtree of a for a random subtree of b...
    let crossover_point = rng.gen_range(0..a.root.node_count());
    let donor = b.root.nodes().choose(rng).unwrap().clone();
    let mut child_root = a.root.replace_at(crossover_point, donor);
    // ...then, sometimes, replace a random subtree with a fresh random one
    if rng.gen::<f64>() < self.mutation_rate {
        // ...
    }
    // bloat control: an offspring deeper than the limit is thrown away
    if child_root.depth() > self.max_tree_depth {
        return a.clone();
    }
    TreeProgram { root: child_root, ..a.clone() }
}`,
}

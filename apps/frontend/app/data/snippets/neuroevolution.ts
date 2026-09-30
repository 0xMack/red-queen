import type { Snippet } from "~/types/code"

// Neuroevolution's snippets. The network exists in both languages: libs/evolve's WeightVector (training) and
// libs/games' Rust `Network` (native rollouts and the browser) -- the same flat layout, checked against each other.
// Gaussian mutation exists only in Python; its Rust is a translation.

export const genome: Snippet = {
  pseudo: `// A genome is two things: one flat list of numbers, and the fixed shape that reads them
WeightVector:
    weights       // every weight and bias, flattened
    layer_sizes   // e.g. (11, 16, 3) -- fixed for the whole run`,
  python: {
    source: "libs/evolve/src/evolve/neuro.py",
    code: `@dataclass(frozen=True, slots=True)
class WeightVector:
    weights: tuple[float, ...]     # every weight and bias, flattened
    layer_sizes: tuple[int, ...]   # e.g. (11, 16, 3) -- fixed for the whole run`,
  },
  rust: {
    source: "libs/games/rust/core/src/nets.rs",
    code: `/// A layered tanh network over one flat weight vector (evolve.WeightVector's layout): per layer,
/// \`out * in\` weights (row-major by output) then \`out\` biases; tanh on every layer.
pub struct Network {
    weights: Vec<f64>,
    layers: Vec<usize>,
}`,
  },
}

export const forward: Snippet = {
  pseudo: `function forward(weights, layer_sizes, observation)
    activations ← observation
    offset ← 0
    for each consecutive pair (n_in, n_out) in layer_sizes
        for each output unit o
            total ← bias of o                                  // stored after the layer's weights
            for each input k
                total ← total + weight(o, k) × activations[k]
            next[o] ← tanh(total)
        offset ← offset + n_in × n_out + n_out                 // step past this layer's numbers
        activations ← next
    return activations`,
  python: {
    source: "libs/evolve/src/evolve/neuro.py",
    code: `def _forward(weights, layer_sizes, observation):
    """A tanh-activated feedforward pass over one flat list of weights."""
    activations = list(observation)
    offset = 0
    for i in range(len(layer_sizes) - 1):
        in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
        next_activations = []
        for o in range(out_size):
            total = weights[offset + in_size * out_size + o]  # this unit's bias
            for k in range(in_size):
                total += weights[offset + o * in_size + k] * activations[k]
            next_activations.append(math.tanh(total))
        offset += in_size * out_size + out_size
        activations = next_activations
    return activations`,
  },
  rust: {
    source: "libs/games/rust/core/src/nets.rs",
    code: `pub fn activations(&self, observation: &[f64]) -> Vec<Vec<f64>> {
    let mut layers = vec![observation.to_vec()];
    let mut offset = 0;
    for pair in self.layers.windows(2) {
        let (n_in, n_out) = (pair[0], pair[1]);
        let previous = layers.last().unwrap();
        let next: Vec<f64> = (0..n_out)
            .map(|o| {
                let row = &self.weights[offset + o * n_in..offset + (o + 1) * n_in];
                let mut total = self.weights[offset + n_in * n_out + o]; // bias
                for (w, a) in row.iter().zip(previous) {
                    total += w * a;
                }
                total.tanh()
            })
            .collect();
        offset += n_out * (n_in + 1);
        layers.push(next);
    }
    layers
}`,
  },
}

export const mutation: Snippet = {
  pseudo: `function mutate(parent, σ)
    child ← a copy of parent
    for each weight w in child
        w ← w + a draw from Normal(0, σ)
    return child`,
  python: {
    source: "libs/evolve/src/evolve/neuro.py",
    code: `class GaussianMutation:
    def __init__(self, sigma: float = 0.1):
        self._sigma = sigma

    def vary(self, parents, rng):
        parent = parents[0]
        new_weights = tuple(w + rng.gauss(0.0, self._sigma) for w in parent.weights)
        return replace(parent, weights=new_weights)`,
  },
  rust: `pub struct GaussianMutation {
    sigma: f64,
}

impl GaussianMutation {
    pub fn vary(&self, parent: &WeightVector, rng: &mut impl Rng) -> WeightVector {
        let noise = Normal::new(0.0, self.sigma).unwrap();
        let weights = parent.weights.iter().map(|w| w + noise.sample(rng)).collect();
        WeightVector { weights, ..parent.clone() }
    }
}`,
}

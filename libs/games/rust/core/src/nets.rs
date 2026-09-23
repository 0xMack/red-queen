//! Trained networks the core runs itself (docs/design/0009): a Checkers evaluator searching positions, a Snake
//! policy playing a whole training episode natively. Two kinds, matching what `evolve` trains:
//!
//! - `Network` -- fixed topology, `evolve.neuro.WeightVector`'s flat layout: per layer, `out * in` weights
//!   (row-major by output) then `out` biases; tanh on every layer.
//! - `GraphNet` -- a NEAT genome compiled to its evaluation plan (`evolve.NeatGenome.graph_encoding()`); tanh on
//!   every computed node.
//!
//! Both accumulate in exactly the order the Python forward passes do (a node's total starts at its bias, or at
//! 0.0, and adds `weight * input` one input at a time), so a network gives bit-for-bit the Python outputs: a
//! policy played here makes the same moves, and earns the same fitness, as the Python loop it replaces.

/// A fixed-topology feedforward network (`evolve.neuro.WeightVector`).
#[derive(Clone, Debug)]
pub struct Network {
    weights: Vec<f64>,
    layers: Vec<usize>,
}

impl Network {
    pub fn new(weights: Vec<f64>, layers: Vec<usize>) -> Result<Network, String> {
        if layers.len() < 2 || layers.contains(&0) {
            return Err(format!("a network needs at least an input and an output layer, none empty; got {layers:?}"));
        }
        let expected: usize = layers.windows(2).map(|w| w[1] * (w[0] + 1)).sum();
        if weights.len() != expected {
            return Err(format!("layers {layers:?} need {expected} weights, got {}", weights.len()));
        }
        Ok(Network { weights, layers })
    }

    pub fn inputs(&self) -> usize {
        self.layers[0]
    }

    pub fn outputs(&self) -> usize {
        *self.layers.last().unwrap()
    }

    pub fn forward(&self, observation: &[f64]) -> Vec<f64> {
        self.activations(observation).pop().unwrap()
    }

    /// The first output: a position evaluator's score.
    pub fn score(&self, observation: &[f64]) -> f64 {
        self.forward(observation)[0]
    }

    /// Every layer's values for `observation`, input layer first (`layers.len()` vectors): what the page's
    /// network diagram lights up.
    pub fn activations(&self, observation: &[f64]) -> Vec<Vec<f64>> {
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
    }
}

/// A NEAT genome compiled to its evaluation plan: a slot per value (inputs, then the bias fixed at 1.0, then
/// every live node in topological order), and per computed node its incoming `(source slot, weight)` pairs.
#[derive(Clone, Debug)]
pub struct GraphNet {
    n_inputs: usize,
    n_slots: usize,
    steps: Vec<(usize, Vec<(usize, f64)>)>,
    outputs: Vec<usize>,
}

impl GraphNet {
    /// From `NeatGenome.graph_encoding()`: `[n_inputs, n_slots, n_steps, (slot, k, (src, w) * k) * n_steps,
    /// n_outputs, output slots...]`.
    pub fn from_flat(flat: &[f64]) -> Result<GraphNet, String> {
        let mut at = 0;
        let mut next = |what: &str| -> Result<f64, String> {
            let v = *flat.get(at).ok_or_else(|| format!("graph encoding ends early, reading {what}"))?;
            at += 1;
            Ok(v)
        };
        let n_inputs = next("n_inputs")? as usize;
        let n_slots = next("n_slots")? as usize;
        let n_steps = next("n_steps")? as usize;
        if n_slots <= n_inputs {
            return Err(format!("a graph needs a bias slot after its {n_inputs} inputs, got {n_slots} slots"));
        }
        let mut steps = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let slot = next("slot")? as usize;
            let k = next("edge count")? as usize;
            let mut incoming = Vec::with_capacity(k);
            for _ in 0..k {
                let source = next("source")? as usize;
                let weight = next("weight")?;
                if source >= slot {
                    return Err(format!("slot {slot} reads slot {source}: steps must be in topological order"));
                }
                incoming.push((source, weight));
            }
            if slot >= n_slots || slot <= n_inputs {
                return Err(format!("step writes slot {slot}, outside the computed slots {}..{n_slots}", n_inputs + 1));
            }
            steps.push((slot, incoming));
        }
        let n_outputs = next("n_outputs")? as usize;
        if n_outputs == 0 {
            return Err("a graph needs an output".into());
        }
        let mut outputs = Vec::with_capacity(n_outputs);
        for _ in 0..n_outputs {
            let output = next("output slot")? as usize;
            if output >= n_slots {
                return Err(format!("output slot {output} is outside the {n_slots} slots"));
            }
            outputs.push(output);
        }
        Ok(GraphNet { n_inputs, n_slots, steps, outputs })
    }

    pub fn inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs.len()
    }

    fn values(&self, observation: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.n_slots];
        values[..self.n_inputs].copy_from_slice(observation);
        values[self.n_inputs] = 1.0; // bias
        for (slot, incoming) in &self.steps {
            let mut total = 0.0;
            for &(source, weight) in incoming {
                total += weight * values[source];
            }
            values[*slot] = total.tanh();
        }
        values
    }

    pub fn forward(&self, observation: &[f64]) -> Vec<f64> {
        let values = self.values(observation);
        self.outputs.iter().map(|&slot| values[slot]).collect()
    }

    /// The first output: a position evaluator's score.
    pub fn score(&self, observation: &[f64]) -> f64 {
        self.values(observation)[self.outputs[0]]
    }
}

/// Either kind of network.
#[derive(Clone, Debug)]
pub enum Net {
    Layered(Network),
    Graph(GraphNet),
}

impl Net {
    pub fn inputs(&self) -> usize {
        match self {
            Net::Layered(n) => n.inputs(),
            Net::Graph(g) => g.inputs(),
        }
    }

    pub fn outputs(&self) -> usize {
        match self {
            Net::Layered(n) => n.outputs(),
            Net::Graph(g) => g.outputs(),
        }
    }

    pub fn forward(&self, observation: &[f64]) -> Vec<f64> {
        match self {
            Net::Layered(n) => n.forward(observation),
            Net::Graph(g) => g.forward(observation),
        }
    }

    pub fn score(&self, observation: &[f64]) -> f64 {
        match self {
            Net::Layered(n) => n.score(observation),
            Net::Graph(g) => g.score(observation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_checks_its_shape_and_computes_tanh() {
        assert!(Network::new(vec![0.0; 10], vec![32, 1]).is_err());
        assert!(Network::new(vec![], vec![3]).is_err());
        let mut weights = vec![0.0; 33];
        weights[32] = 0.5;
        let network = Network::new(weights, vec![32, 1]).unwrap();
        assert!((network.score(&[0.0; 32]) - 0.5f64.tanh()).abs() < 1e-15);
    }

    #[test]
    fn network_adds_to_the_bias_one_input_at_a_time() {
        // (bias + w0*a0) + w1*a1 is not always bias + (w0*a0 + w1*a1) in floating point: the order is the
        // Python forward pass's, so the result is its result bit for bit.
        let (bias, w, a) = (1.0, [1e16, -1e16], [1.0, 1.0]);
        let network = Network::new(vec![w[0], w[1], bias], vec![2, 1]).unwrap();
        let python_order = ((bias + w[0] * a[0]) + w[1] * a[1]).tanh(); // 1 is lost against 1e16: tanh(0)
        assert_eq!(network.forward(&a), vec![python_order]);
        assert_ne!(python_order, (bias + (w[0] * a[0] + w[1] * a[1])).tanh());
    }

    #[test]
    fn graph_net_matches_a_hand_computed_plan_and_returns_every_output() {
        // slots: 0..2 inputs, 2 bias, 3 hidden, 4 output, 5 output. hidden = tanh(2*in0 + 0.5*bias);
        // out4 = tanh(-1*hidden + 0.25*in1); out5 = tanh(in0).
        let flat = [
            2.0, 6.0, 3.0, //
            3.0, 2.0, 0.0, 2.0, 2.0, 0.5, //
            4.0, 2.0, 3.0, -1.0, 1.0, 0.25, //
            5.0, 1.0, 0.0, 1.0, //
            2.0, 4.0, 5.0,
        ];
        let net = GraphNet::from_flat(&flat).unwrap();
        let hidden = (2.0f64 + 0.5).tanh();
        let out = net.forward(&[1.0, 2.0]);
        assert!((out[0] - (-hidden + 0.5).tanh()).abs() < 1e-15);
        assert!((out[1] - 1.0f64.tanh()).abs() < 1e-15);
        assert_eq!(net.score(&[1.0, 2.0]), out[0]);
        assert_eq!((net.inputs(), net.outputs()), (2, 2));
        assert!(GraphNet::from_flat(&flat[..10]).is_err());
        let mut bad = flat;
        bad[11] = 4.0; // slot 4 reading slot 4 (itself)
        assert!(GraphNet::from_flat(&bad).is_err());
    }
}

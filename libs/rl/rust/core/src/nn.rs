//! A small multi-layer perceptron with explicit backpropagation, and Adam (docs/design/0010 Decision 1).
//!
//! The networks RL needs here are small (at most a few thousand weights), so this is a hand-written dense layer
//! stack, not a framework. Parameters are one flat vector in `evolve.WeightVector`'s layout -- per layer, `out * in`
//! weights (row-major by output), then `out` biases -- so an optimizer, a checkpoint and an export all see one slice.
//!
//! Arithmetic order is fixed (a unit's total starts at its bias and adds `weight * input` one input at a time), and
//! transcendentals go through `libm`, so a seed gives bit-identical results natively and in WASM.
//! `libs/rl/tests/reference_nn.py` recomputes every gradient with `libs/autodiff` and must agree to ~1e-12.

use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Relu,
    Tanh,
}

impl Activation {
    pub fn parse(name: &str) -> Result<Activation, String> {
        match name {
            "linear" => Ok(Activation::Linear),
            "relu" => Ok(Activation::Relu),
            "tanh" => Ok(Activation::Tanh),
            other => Err(format!("unknown activation {other:?} (linear, relu, tanh)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Activation::Linear => "linear",
            Activation::Relu => "relu",
            Activation::Tanh => "tanh",
        }
    }

    fn apply(self, z: f64) -> f64 {
        match self {
            Activation::Linear => z,
            Activation::Relu => {
                if z > 0.0 {
                    z
                } else {
                    0.0
                }
            }
            Activation::Tanh => libm::tanh(z),
        }
    }

    /// d(activation)/dz, from the pre-activation `z` and the activation's output `a`.
    fn derivative(self, z: f64, a: f64) -> f64 {
        match self {
            Activation::Linear => 1.0,
            Activation::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => 1.0 - a * a,
        }
    }
}

/// The shape of a network: layer sizes (input first) and one activation per non-input layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<Activation>,
}

impl Shape {
    pub fn new(layer_sizes: Vec<usize>, activations: Vec<Activation>) -> Result<Shape, String> {
        if layer_sizes.len() < 2 || layer_sizes.contains(&0) {
            return Err(format!(
                "a network needs an input and an output layer, none empty; got {layer_sizes:?}"
            ));
        }
        if activations.len() != layer_sizes.len() - 1 {
            return Err(format!(
                "{} layers of weights need {} activations, got {}",
                layer_sizes.len() - 1,
                layer_sizes.len() - 1,
                activations.len()
            ));
        }
        Ok(Shape {
            layer_sizes,
            activations,
        })
    }

    /// Hidden layers with `hidden` activation, a linear output: the usual Q-network / policy-logits shape.
    pub fn mlp(inputs: usize, hidden: &[usize], outputs: usize, hidden_activation: Activation) -> Shape {
        let mut layer_sizes = vec![inputs];
        layer_sizes.extend_from_slice(hidden);
        layer_sizes.push(outputs);
        let mut activations = vec![hidden_activation; hidden.len()];
        activations.push(Activation::Linear);
        Shape::new(layer_sizes, activations).expect("mlp() builds a valid shape")
    }

    pub fn inputs(&self) -> usize {
        self.layer_sizes[0]
    }

    pub fn outputs(&self) -> usize {
        *self.layer_sizes.last().unwrap()
    }

    pub fn parameter_count(&self) -> usize {
        self.layer_sizes.windows(2).map(|w| w[1] * (w[0] + 1)).sum()
    }
}

/// Values a forward pass keeps for the backward pass: every layer's pre-activations and outputs, `batch` rows each.
pub struct Cache {
    batch: usize,
    inputs: Vec<f64>,
    pre: Vec<Vec<f64>>,
    post: Vec<Vec<f64>>,
}

impl Cache {
    pub fn outputs(&self) -> &[f64] {
        self.post.last().unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct Mlp {
    pub shape: Shape,
    pub params: Vec<f64>,
}

impl Mlp {
    pub fn new(shape: Shape, params: Vec<f64>) -> Result<Mlp, String> {
        if params.len() != shape.parameter_count() {
            return Err(format!(
                "layers {:?} need {} parameters, got {}",
                shape.layer_sizes,
                shape.parameter_count(),
                params.len()
            ));
        }
        Ok(Mlp { shape, params })
    }

    /// He-uniform weights (scaled for the layer's fan-in: `sqrt(6 / fan_in)`), zero biases.
    pub fn init(shape: Shape, rng: &mut Rng) -> Mlp {
        let mut params = Vec::with_capacity(shape.parameter_count());
        for pair in shape.layer_sizes.windows(2) {
            let (n_in, n_out) = (pair[0], pair[1]);
            let limit = libm::sqrt(6.0 / n_in as f64);
            params.extend((0..n_in * n_out).map(|_| (rng.uniform() * 2.0 - 1.0) * limit));
            params.extend(std::iter::repeat_n(0.0, n_out));
        }
        Mlp { shape, params }
    }

    /// Outputs for one observation.
    pub fn forward(&self, observation: &[f64]) -> Vec<f64> {
        self.forward_batch(observation, 1).post.pop().unwrap()
    }

    /// `inputs` holds `batch` rows of `shape.inputs()` values; the cache's `outputs()` holds `batch` rows of
    /// `shape.outputs()`.
    pub fn forward_batch(&self, inputs: &[f64], batch: usize) -> Cache {
        assert_eq!(inputs.len(), batch * self.shape.inputs(), "input size");
        let mut pre = Vec::with_capacity(self.shape.activations.len());
        let mut post: Vec<Vec<f64>> = Vec::with_capacity(self.shape.activations.len());
        let mut offset = 0;
        for (layer, pair) in self.shape.layer_sizes.windows(2).enumerate() {
            let (n_in, n_out) = (pair[0], pair[1]);
            let activation = self.shape.activations[layer];
            let x: &[f64] = if layer == 0 { inputs } else { &post[layer - 1] };
            let weights = &self.params[offset..offset + n_in * n_out];
            let biases = &self.params[offset + n_in * n_out..offset + n_out * (n_in + 1)];
            let mut z = vec![0.0; batch * n_out];
            let mut a = vec![0.0; batch * n_out];
            for row in 0..batch {
                let xr = &x[row * n_in..(row + 1) * n_in];
                for o in 0..n_out {
                    let w = &weights[o * n_in..(o + 1) * n_in];
                    let mut total = biases[o];
                    for k in 0..n_in {
                        total += w[k] * xr[k];
                    }
                    z[row * n_out + o] = total;
                    a[row * n_out + o] = activation.apply(total);
                }
            }
            offset += n_out * (n_in + 1);
            pre.push(z);
            post.push(a);
        }
        Cache {
            batch,
            inputs: inputs.to_vec(),
            pre,
            post,
        }
    }

    /// Gradient of a loss with respect to every parameter, given the loss's gradient with respect to the outputs
    /// (`output_grad`: `batch` rows of `shape.outputs()`), summed over the batch. Same layout as `params`.
    pub fn backward(&self, cache: &Cache, output_grad: &[f64]) -> Vec<f64> {
        self.backward_inner(cache, output_grad, false).0
    }

    /// `backward`, plus the loss's gradient with respect to the *inputs* (`batch` rows of `shape.inputs()`): what a
    /// network feeding this one (a dueling Q-network's shared trunk) backpropagates from.
    pub fn backward_with_inputs(&self, cache: &Cache, output_grad: &[f64]) -> (Vec<f64>, Vec<f64>) {
        self.backward_inner(cache, output_grad, true)
    }

    fn backward_inner(&self, cache: &Cache, output_grad: &[f64], input_grad: bool) -> (Vec<f64>, Vec<f64>) {
        let batch = cache.batch;
        assert_eq!(output_grad.len(), batch * self.shape.outputs(), "output gradient size");
        let mut grads = vec![0.0; self.params.len()];
        let offsets: Vec<usize> = self
            .shape
            .layer_sizes
            .windows(2)
            .scan(0, |at, w| {
                let here = *at;
                *at += w[1] * (w[0] + 1);
                Some(here)
            })
            .collect();
        let mut upstream = output_grad.to_vec(); // dL/d(this layer's output)
        for layer in (0..self.shape.activations.len()).rev() {
            let (n_in, n_out) = (self.shape.layer_sizes[layer], self.shape.layer_sizes[layer + 1]);
            let activation = self.shape.activations[layer];
            let offset = offsets[layer];
            let x: &[f64] = if layer == 0 {
                &cache.inputs
            } else {
                &cache.post[layer - 1]
            };
            let (z, a) = (&cache.pre[layer], &cache.post[layer]);
            let dz: Vec<f64> = (0..batch * n_out)
                .map(|i| upstream[i] * activation.derivative(z[i], a[i]))
                .collect();
            let (weight_grads, bias_grads) = grads[offset..offset + n_out * (n_in + 1)].split_at_mut(n_in * n_out);
            for row in 0..batch {
                let xr = &x[row * n_in..(row + 1) * n_in];
                for o in 0..n_out {
                    let d = dz[row * n_out + o];
                    bias_grads[o] += d;
                    let wg = &mut weight_grads[o * n_in..(o + 1) * n_in];
                    for k in 0..n_in {
                        wg[k] += d * xr[k];
                    }
                }
            }
            if layer > 0 || input_grad {
                let weights = &self.params[offset..offset + n_in * n_out];
                let mut below = vec![0.0; batch * n_in];
                for row in 0..batch {
                    for o in 0..n_out {
                        let d = dz[row * n_out + o];
                        let w = &weights[o * n_in..(o + 1) * n_in];
                        let br = &mut below[row * n_in..(row + 1) * n_in];
                        for k in 0..n_in {
                            br[k] += d * w[k];
                        }
                    }
                }
                upstream = below;
            }
        }
        let inputs = if input_grad { upstream } else { Vec::new() };
        (grads, inputs)
    }
}

/// Adam (Kingma & Ba, 2015) over a flat parameter vector, with bias correction.
#[derive(Clone, Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    /// `beta1^t` and `beta2^t`, kept by multiplication rather than `powi` so every target computes the same bits.
    beta1_t: f64,
    beta2_t: f64,
    pub steps: u64,
}

impl Adam {
    pub fn new(parameters: usize, learning_rate: f64) -> Adam {
        Adam::with(parameters, learning_rate, 0.9, 0.999, 1e-8)
    }

    pub fn with(parameters: usize, learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Adam {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: vec![0.0; parameters],
            v: vec![0.0; parameters],
            beta1_t: 1.0,
            beta2_t: 1.0,
            steps: 0,
        }
    }

    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(
            params.len(),
            self.m.len(),
            "Adam was built for {} parameters",
            self.m.len()
        );
        assert_eq!(grads.len(), params.len(), "one gradient per parameter");
        self.steps += 1;
        self.beta1_t *= self.beta1;
        self.beta2_t *= self.beta2;
        let (c1, c2) = (1.0 - self.beta1_t, 1.0 - self.beta2_t);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / c1;
            let v_hat = self.v[i] / c2;
            params[i] -= self.learning_rate * m_hat / (libm::sqrt(v_hat) + self.epsilon);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Stream;

    fn net(seed: u64) -> Mlp {
        Mlp::init(
            Shape::new(
                vec![3, 5, 4, 2],
                vec![Activation::Relu, Activation::Tanh, Activation::Linear],
            )
            .unwrap(),
            &mut Rng::new(seed, Stream::Init),
        )
    }

    #[test]
    fn shape_checks_and_counts() {
        assert!(Shape::new(vec![3], vec![]).is_err());
        assert!(Shape::new(vec![3, 2], vec![]).is_err());
        let shape = Shape::mlp(11, &[64, 64], 3, Activation::Relu);
        assert_eq!(shape.parameter_count(), 11 * 64 + 64 + 64 * 64 + 64 + 64 * 3 + 3);
        assert_eq!(shape.activations.last(), Some(&Activation::Linear));
        assert!(Mlp::new(shape, vec![0.0; 3]).is_err());
    }

    #[test]
    fn batch_rows_are_independent_and_match_single_forward() {
        let mlp = net(1);
        let x = [0.5, -1.0, 2.0, 0.1, 0.2, -0.3];
        let cache = mlp.forward_batch(&x, 2);
        assert_eq!(&cache.outputs()[..2], mlp.forward(&x[..3]).as_slice());
        assert_eq!(&cache.outputs()[2..], mlp.forward(&x[3..]).as_slice());
    }

    #[test]
    fn gradients_match_finite_differences() {
        let mut mlp = net(2);
        let x = [0.3, -0.7, 1.1, -0.2, 0.4, 0.9];
        let g = [0.5, -1.5, 2.0, 0.25]; // loss = sum(outputs * g)
        let loss = |m: &Mlp| {
            m.forward_batch(&x, 2)
                .outputs()
                .iter()
                .zip(&g)
                .map(|(o, g)| o * g)
                .sum::<f64>()
        };
        let grads = mlp.backward(&mlp.forward_batch(&x, 2), &g);
        let h = 1e-6;
        for (i, &grad) in grads.iter().enumerate() {
            let original = mlp.params[i];
            mlp.params[i] = original + h;
            let up = loss(&mlp);
            mlp.params[i] = original - h;
            let down = loss(&mlp);
            mlp.params[i] = original;
            let numeric = (up - down) / (2.0 * h);
            assert!((numeric - grad).abs() < 1e-6, "param {i}: {numeric} vs {}", grad);
        }
    }

    #[test]
    fn input_gradients_match_finite_differences() {
        let mlp = net(4);
        let mut x = vec![0.3, -0.7, 1.1, -0.2, 0.4, 0.9];
        let g = [0.5, -1.5, 2.0, 0.25];
        let loss = |x: &[f64]| {
            mlp.forward_batch(x, 2)
                .outputs()
                .iter()
                .zip(&g)
                .map(|(o, g)| o * g)
                .sum::<f64>()
        };
        let (params, inputs) = mlp.backward_with_inputs(&mlp.forward_batch(&x, 2), &g);
        assert_eq!(params, mlp.backward(&mlp.forward_batch(&x, 2), &g));
        let h = 1e-6;
        for i in 0..x.len() {
            let original = x[i];
            x[i] = original + h;
            let up = loss(&x);
            x[i] = original - h;
            let down = loss(&x);
            x[i] = original;
            assert!(((up - down) / (2.0 * h) - inputs[i]).abs() < 1e-6, "input {i}");
        }
    }

    #[test]
    fn adam_fits_a_line() {
        // y = 2x - 1 with a single linear unit.
        let shape = Shape::new(vec![1, 1], vec![Activation::Linear]).unwrap();
        let mut mlp = Mlp::new(shape, vec![0.0, 0.0]).unwrap();
        let mut adam = Adam::new(2, 0.05);
        let xs = [-1.0, -0.5, 0.0, 0.5, 1.0];
        for _ in 0..2000 {
            let cache = mlp.forward_batch(&xs, 5);
            let grad: Vec<f64> = cache
                .outputs()
                .iter()
                .zip(xs)
                .map(|(y, x)| 2.0 * (y - (2.0 * x - 1.0)) / 5.0)
                .collect();
            let grads = mlp.backward(&cache, &grad);
            adam.step(&mut mlp.params, &grads);
        }
        assert!(
            (mlp.params[0] - 2.0).abs() < 1e-3 && (mlp.params[1] + 1.0).abs() < 1e-3,
            "{:?}",
            mlp.params
        );
        assert_eq!(adam.steps, 2000);
    }
}

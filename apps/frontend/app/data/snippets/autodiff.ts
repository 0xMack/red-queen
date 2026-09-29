import type { Snippet } from "~/types/code"

// Autodiff's snippets. The engine is Python (libs/autodiff, numpy arrays); its Rust is a translation onto scalar
// values (micrograd-style: the same graph and the same backward pass, without an array library). Adam exists in both:
// libs/tinylm's optimizer and the RL core's (libs/rl/rust/core/src/nn.rs). The numpy-specific pitfalls
// (broadcasting, fancy indexing) stay Python-only: they are about numpy.
const TENSOR = "libs/autodiff/src/autodiff/tensor.py"

export const mul: Snippet = {
  pseudo: `function multiply(a, b)
    out ← a new node holding a.value × b.value, remembering its parents a and b
    out.backward ← procedure
        // d(a×b)/da = b, d(a×b)/db = a -- each times the gradient flowing in from above
        a.grad ← a.grad + b.value × out.grad
        b.grad ← b.grad + a.value × out.grad
    return out`,
  python: {
    source: TENSOR,
    code: `def __mul__(self, other) -> Tensor:
    other = self._as_tensor(other)
    out = Tensor(self.data * other.data, (self, other), "*")

    def _backward():
        # d(a*b)/da = b, d(a*b)/db = a -- times the gradient flowing in from above
        self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
        other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

    out._backward = _backward
    return out`,
  },
  rust: `impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let (a, b) = (self.clone(), other.clone());
        Value::with_backward(self.data() * other.data(), vec![a.clone(), b.clone()], "*", move |out_grad| {
            // d(a*b)/da = b, d(a*b)/db = a -- times the gradient flowing in from above
            *a.grad_mut() += b.data() * out_grad;
            *b.grad_mut() += a.data() * out_grad;
        })
    }
}`,
}

export const backward: Snippet = {
  pseudo: `function backward(loss)
    order ← the graph's nodes, each one after all of its parents   // a topological sort
    loss.grad ← 1                                                   // dL/dL = 1
    for each node in order, reversed        // every node runs after all the nodes that depend on it
        node.backward()                     // push its gradient to its parents`,
  python: {
    source: TENSOR,
    code: `def backward(self) -> None:
    topo, visited = [], set()

    def build(node):
        if id(node) not in visited:
            visited.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

    build(self)
    self.grad = np.ones_like(self.data)   # dL/dL = 1
    for node in reversed(topo):           # every node after all nodes that depend on it
        node._backward()`,
  },
  rust: `pub fn backward(&self) {
    fn build(node: &Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
        if visited.insert(node.id()) {
            for parent in node.parents() {
                build(parent, visited, topo);
            }
            topo.push(node.clone());
        }
    }
    let (mut topo, mut visited) = (Vec::new(), HashSet::new());
    build(self, &mut visited, &mut topo);
    *self.grad_mut() = 1.0; // dL/dL = 1
    for node in topo.iter().rev() {
        node.run_backward(); // every node after all nodes that depend on it
    }
}`,
}

export const gradCheck: Snippet = {
  pseudo: `// The derivative's definition, computed the slow way, as the answer key
function numerical_grad(f, x, ε = 1e-6)
    for each entry x[i]
        grad[i] ← ( f(x with x[i] + ε) − f(x with x[i] − ε) ) / 2ε
    return grad

test: backward()'s gradient ≈ numerical_grad, for every operation`,
  python: {
    source: "libs/autodiff/tests/test_tensor.py",
    code: `def numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        original = x[idx]
        x[idx] = original + eps; f_plus = f(x)
        x[idx] = original - eps; f_minus = f(x)
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad

def test_matmul_gradient_matches_numerical():
    a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
    (a @ b).sum().backward()
    assert np.allclose(a.grad, numerical_grad(lambda x: (x @ b_data).sum(), a_data.copy()), atol=1e-4)`,
  },
}

export const adam: Snippet = {
  pseudo: `function adam_step(params, grads)
    t ← t + 1
    for each parameter p with gradient g
        m[p] ← β₁ × m[p] + (1 − β₁) × g            // momentum: a running mean of the gradient
        v[p] ← β₂ × v[p] + (1 − β₂) × g²           // a running mean of its square: a per-weight scale
        m̂ ← m[p] / (1 − β₁ᵗ),  v̂ ← v[p] / (1 − β₂ᵗ)  // bias correction: both start at zero
        p ← p − lr × m̂ / (√v̂ + ε)`,
  python: {
    source: "libs/tinylm/src/tinylm/optim.py",
    code: `def step(self):
    self._t += 1
    for i, p in enumerate(self.parameters):
        self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * p.grad        # momentum
        self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (p.grad**2)   # per-weight scale
        m_hat = self._m[i] / (1 - self.beta1**self._t)                         # bias correction
        v_hat = self._v[i] / (1 - self.beta2**self._t)
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)`,
  },
  rust: {
    source: "libs/rl/rust/core/src/nn.rs",
    code: `pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
    self.steps += 1;
    // beta^t kept by multiplication rather than powi, so every target computes the same bits
    self.beta1_t *= self.beta1;
    self.beta2_t *= self.beta2;
    let (c1, c2) = (1.0 - self.beta1_t, 1.0 - self.beta2_t);   // bias correction
    for i in 0..params.len() {
        let g = grads[i];
        self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;       // momentum
        self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;   // per-weight scale
        let m_hat = self.m[i] / c1;
        let v_hat = self.v[i] / c2;
        params[i] -= self.learning_rate * m_hat / (libm::sqrt(v_hat) + self.epsilon);
    }
}`,
  },
}

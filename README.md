# mini-grad
A minimal reverse-mode automatic differentiation engine with computational graphs, backpropagation, and a neural network API.
# Understanding Reverse-Mode Automatic Differentiation

> A deep dive into how MiniGrad (and PyTorch) compute gradients automatically.

---

## Table of Contents

1. [What is Automatic Differentiation?](#1-what-is-automatic-differentiation)
2. [Computational Graphs](#2-computational-graphs)
3. [Topological Sorting](#3-topological-sorting)
4. [The Chain Rule](#4-the-chain-rule)
5. [Reverse-Mode Autodiff (Backpropagation)](#5-reverse-mode-autodiff-backpropagation)
6. [Gradient Accumulation](#6-gradient-accumulation)
7. [Forward-Mode vs Reverse-Mode](#7-forward-mode-vs-reverse-mode)
8. [Why Reverse-Mode is Efficient for Neural Networks](#8-why-reverse-mode-is-efficient-for-neural-networks)
9. [Worked Example](#9-worked-example)

---

## 1. What is Automatic Differentiation?

Automatic Differentiation (autodiff) is a technique for evaluating the derivative of a function specified by a computer program by systematically applying the chain rule to elementary operations.

| Method | Description | Accuracy | Cost |
|--------|-------------|----------|------|
| **Numerical** | Finite differences: `(f(x+h) - f(x)) / h` | Approximate | O(n) evaluations |
| **Symbolic** | Algebraic manipulation | Exact | Expression swell |
| **Automatic** | Trace + chain rule | Exact (machine precision) | O(1) evaluations |

MiniGrad implements **automatic differentiation** — specifically the reverse-mode variant.

---

## 2. Computational Graphs

Every expression is a directed acyclic graph (DAG):

- **Nodes** represent values.
- **Edges** represent data flow from input to output.
- **Op labels** indicate the operation that produced each node.

For `f = (a + b) * c`:

```
   a ──┐
       ├──[+]── (a+b) ──[*]── f
   b ──┘              │
                      c ──────┘
```

Each `Value` node stores:
- `data` — the forward-pass result.
- `grad` — the accumulated gradient (filled by `backward()`).
- `_prev` — the set of input nodes.
- `_backward` — a closure that distributes gradients to `_prev`.

---

## 3. Topological Sorting

Before backpropagating we need a **topological ordering** — every input to a node must appear before that node.

```python
def build_topo(node):
    if node not in visited:
        visited.add(node)
        for child in node._prev:
            build_topo(child)   # inputs first
        topo.append(node)       # append after all inputs
```

Reversing the result gives the backpropagation order.

For `f = (a + b) * c`:
```
Forward topo:  a → b → (a+b) → c → f
Backward order: f → c → (a+b) → b → a
```

---

## 4. The Chain Rule

### Single-variable

If `y = g(x)` and `z = f(y)`:
```
dz/dx = (dz/dy) · (dy/dx)
```

### Multi-variable (general)

If node `v` contributes to `L` via paths through `u₁, …, uₙ`:
```
∂L/∂v = Σᵢ (∂L/∂uᵢ) · (∂uᵢ/∂v)
```

Gradients **accumulate** — each path adds a term. This is why every `_backward` uses `+=`.

### Local gradients per operation

| Operation | `∂out/∂a` | `∂out/∂b` |
|-----------|-----------|-----------|
| `a + b`   | `1`       | `1`       |
| `a * b`   | `b`       | `a`       |
| `a ** n`  | `n·a^(n-1)` | —       |
| `relu(a)` | `1 if a>0 else 0` | — |

Each closure multiplies the local gradient by the upstream `out.grad`:

```python
# Multiplication: out = a * b
def _backward():
    a.grad += b.data * out.grad   # ∂out/∂a = b
    b.grad += a.data * out.grad   # ∂out/∂b = a
```

---

## 5. Reverse-Mode Autodiff (Backpropagation)

```python
def backward(self):
    topo = topological_sort(self)
    self.grad = 1.0          # seed: ∂L/∂L = 1
    for node in reversed(topo):
        node._backward()     # propagate to inputs
```

The topological ordering guarantees each node's upstream gradient is ready before its `_backward` fires.

---

## 6. Gradient Accumulation

When node `v` contributes via **multiple paths**, each path adds to `v.grad`:

```
∂L/∂v = path₁_gradient + path₂_gradient + …
```

Example: `f = a² + a³`
```
a ──[**2]── a² ──┐
                 ├──[+]── f
a ──[**3]── a³ ──┘

∂f/∂a = 2a + 3a²
```

---

## 7. Forward-Mode vs Reverse-Mode

| | Forward-mode | Reverse-mode |
|--|--|--|
| Computes | ∂y/∂x for one fixed x direction | ∂y/∂xᵢ for all i, for fixed scalar y |
| Passes for full Jacobian | #inputs | #outputs |
| Best for | Few inputs, many outputs | Many inputs, one scalar output |
| Neural networks | Expensive | Efficient ✓ |

---

## 8. Why Reverse-Mode is Efficient for Neural Networks

A neural network has:
- Millions of **parameters** (inputs).
- One scalar **loss** (output).

Reverse-mode computes all gradients in **one backward pass** — the same cost as the forward pass. Forward-mode would need one pass per parameter: millions of times slower.

This is why PyTorch, JAX, and TensorFlow all use reverse-mode autodiff.

---

## 9. Worked Example

Expression: `f = (a + b) * c` with `a=2, b=-1, c=3`.

**Forward pass:**
```
s = a + b = 1.0
f = s * c = 3.0
```

**Seed:** `f.grad = 1.0`

**Backward through `f = s * c`:**
```
s.grad += c * f.grad = 3 * 1 = 3
c.grad += s * f.grad = 1 * 1 = 1
```

**Backward through `s = a + b`:**
```
a.grad += 1 * s.grad = 3
b.grad += 1 * s.grad = 3
```

**Final:** `∂f/∂a = 3 = c` ✓ · `∂f/∂b = 3 = c` ✓ · `∂f/∂c = 1 = a+b` ✓

---

*For the PyTorch comparison, see [pytorch_comparison.md](pytorch_comparison.md).*


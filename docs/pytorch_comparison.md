# MiniGrad vs PyTorch — A Detailed Comparison

---

## Table of Contents

1. [Similarities](#1-similarities)
2. [Key Differences](#2-key-differences)
3. [Missing Features](#3-missing-features)
4. [Tensor Support](#4-tensor-support)
5. [Broadcasting](#5-broadcasting)
6. [GPU Support](#6-gpu-support)
7. [Performance](#7-performance)
8. [Dynamic Graph Construction](#8-dynamic-graph-construction)
9. [API Comparison](#9-api-comparison)
10. [Why MiniGrad is Educational, Not a Replacement](#10-why-minigrad-is-educational-not-a-replacement)

---

## 1. Similarities

| Concept | MiniGrad | PyTorch |
|---------|----------|---------|
| Core abstraction | `Value` (scalar node) | `Tensor` (n-d array node) |
| Graph construction | Dynamic (eager) | Dynamic (eager) by default |
| Gradient storage | `value.grad` | `tensor.grad` |
| Gradient zeroing | `module.zero_grad()` | `optimizer.zero_grad()` |
| Backward trigger | `loss.backward()` | `loss.backward()` |
| Topological sort | DFS postorder | Similar (C++ implementation) |
| `Module` API | `Module`, `parameters()` | `nn.Module`, `parameters()` |

---

## 2. Key Differences

| | MiniGrad | PyTorch |
|-|----------|---------|
| Lines of core code | ~100 | ~1,000,000+ |
| Node type | `float` scalar | n-dimensional array |
| Supported ops | 6 primitives | 2,000+ |
| Hardware | CPU only | CPU, CUDA, ROCm, MPS |
| Gradient functions | Python closures | C++ `Function` subclasses |

---

## 3. Missing Features

| Feature | Notes |
|---------|-------|
| Tensors / batching | MiniGrad is scalar-only |
| `exp`, `log`, `tanh`, `sigmoid` | Only `relu` and `**n` built in |
| Optimizers (Adam, momentum SGD) | Only manual SGD in examples |
| GPU (CUDA) | Not applicable for scalar autodiff |
| `torch.no_grad()` | Every op builds the graph |
| Higher-order gradients | `backward()` cannot be called on grads |
| In-place operations | Not supported |
| Distributed training | Not applicable |

---

## 4. Tensor Support

PyTorch computes all element gradients in one backward call:

```python
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x.requires_grad_(True)
y = x ** 2
y.sum().backward()
print(x.grad)  # tensor([[2., 4.], [6., 8.]])
```

MiniGrad requires one `Value` per scalar. Tensors enable batching, SIMD, and GPU parallelism — none of which are possible at the scalar level.

Scalars are pedagogically superior: stepping through MiniGrad's backward shows exactly one gradient contribution at a time.

---

## 5. Broadcasting

PyTorch automatically extends shapes when operating on arrays of different sizes:

```python
a = torch.ones(3, 1)   # (3, 1)
b = torch.ones(1, 4)   # (1, 4)
c = a + b              # (3, 4) — broadcast
```

During backprop, PyTorch sums gradients over broadcasted dimensions. MiniGrad has no broadcasting because there are no tensors.

---

## 6. GPU Support

```python
x = torch.randn(1000, 1000).cuda()
y = x @ x.T  # matrix multiply on GPU — ~1000× faster
```

GPU acceleration requires a tensor abstraction, a CUDA/ROCm backend, and C++/CUDA backward functions — none of which are feasible at the scalar level.

---

## 7. Performance

Rough comparison: MLP with 10k parameters, 1k samples, 100 epochs.

| | MiniGrad | PyTorch (CPU) | PyTorch (GPU) |
|-|----------|---------------|---------------|
| Time | ~60 min | ~10 sec | ~0.5 sec |
| Memory | ~1 GB | ~10 MB | ~10 MB |

MiniGrad is ~360× slower than PyTorch on CPU. Sources of overhead:
1. One Python object per scalar (~200 bytes vs 4 bytes for `float32`).
2. One Python function call per `_backward` closure.
3. No vectorisation or operation fusion.

---

## 8. Dynamic Graph Construction

Both MiniGrad and PyTorch (by default) use **define-by-run** graphs:

- The graph is built **during** the forward pass.
- A different graph can be built on every forward call (supports `if`-branches, variable-length inputs, etc.).

This contrasts with TF1/Theano's static graph style. PyTorch 2.0 added `torch.compile()` for AOT compilation, but the default remains dynamic.

---

## 9. API Comparison

### Scalar computation

```python
# MiniGrad
from micrograd.engine import Value
a, b = Value(2.0), Value(3.0)
c = a * b + a
c.backward()
print(a.grad)  # 4.0

# PyTorch
import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b + a
c.backward()
print(a.grad)  # tensor(4.)
```

### Neural network

```python
# MiniGrad
from micrograd.nn import MLP
model = MLP(2, [4, 1])
loss = model([1.0, -0.5]) ** 2
model.zero_grad(); loss.backward()
for p in model.parameters(): p.data -= 0.01 * p.grad

# PyTorch
import torch.nn as nn
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(torch.tensor([[1.0, -0.5]])) ** 2
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

---

## 10. Why MiniGrad is Educational, Not a Replacement

MiniGrad answers one question: **How does automatic differentiation actually work?**

Its value is transparency:
- Read the entire autograd engine in 5 minutes.
- Step through a backward pass in a Python debugger.
- Break it, fix it, extend it — building intuition that transfers to PyTorch, JAX, and any other autodiff system.

**Use MiniGrad to learn. Use PyTorch to build.**

---

*For the mathematical foundations, see [autograd.md](autograd.md).*

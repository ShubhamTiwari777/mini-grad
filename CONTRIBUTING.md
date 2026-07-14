# Contributing to MiniGrad

Thank you for your interest! MiniGrad is an educational project — contributions that improve **clarity**, **correctness**, or **pedagogical value** are most welcome.

---

## What to Contribute

**Welcome:**
- Bug fixes — incorrect gradient computations, edge cases in backprop.
- New differentiable operations — `exp`, `log`, `tanh`, `sigmoid`, etc.
- New examples — interesting expressions or training scenarios.
- Documentation improvements — clearer explanations, better diagrams, typo fixes.
- Extended tests — new gradient verification cases.

**Please avoid:**
- Replacing the scalar `Value` engine with a tensor-based implementation.
- Adding production features (GPU, distributed training, serialisation).
- Significantly increasing the complexity of `engine.py` without a strong educational justification.

The goal is to keep the core **minimal and readable**.

---

## Getting Started

```bash
git clone https://github.com/ShubhamTiwari777/mini-grad.git
cd mini-grad
pip install -e ".[test]"
```

---

## Coding Standards

- Python 3.8+, standard library only (no new required dependencies).
- Follow **PEP 8**; lines ≤ 100 characters.
- **Type hints** on all function signatures.
- **Docstrings** (NumPy/Google style) for all public functions and classes.

---

## Tests

All gradient computations must be verified against PyTorch:

```python
def test_my_op():
    a_mg = Value(2.0)
    a_mg.my_op().backward()

    import torch
    a_pt = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    torch.my_op(a_pt).backward()

    assert abs(a_mg.grad - a_pt.grad.item()) < 1e-6
```

Run tests: `PYTHONPATH=. pytest tests/ -v`

---

## Adding a New Operation

1. Add the method to `Value` in `micrograd/engine.py`.
2. Implement `_backward` with the correct local gradient.
3. Add a docstring explaining the gradient rule.
4. Add PyTorch-verified tests in `tests/test_engine.py`.
5. Add a usage example in `examples/` if significant.

**Example — `tanh`:**

```python
def tanh(self) -> "Value":
    """Hyperbolic tangent. Gradient: 1 - tanh(x)²"""
    import math
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")
    def _backward() -> None:
        self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    return out
```

---

## Pull Request Checklist

- [ ] All existing tests pass.
- [ ] New code has type hints and docstrings.
- [ ] New gradients verified against PyTorch.
- [ ] README updated if necessary.
- [ ] PR description explains the change and its educational motivation.

---

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add tanh activation with gradient test
fix: correct gradient accumulation for shared nodes
docs: clarify chain rule derivation in autograd.md
test: add polynomial gradient test
```

---

*Questions? Open a GitHub Issue.*

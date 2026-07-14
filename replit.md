# MiniGrad

A minimal scalar-valued autograd engine built from scratch in Python — reverse-mode automatic differentiation (backpropagation) the way PyTorch does it, in ~100 lines of readable code.

## Run & Operate

```bash
# Run examples (from workspace root)
PYTHONPATH=. python3 examples/example_basic.py
PYTHONPATH=. python3 examples/example_relu.py
PYTHONPATH=. python3 examples/example_mlp.py
PYTHONPATH=. python3 examples/example_graph.py   # requires: pip install graphviz

# Run tests (requires PyTorch; tests gracefully skip without it)
PYTHONPATH=. python3 -m pytest tests/ -v

# Install PyTorch (CPU-only, for test comparisons)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Stack

- Python 3.8+, zero required dependencies
- Optional: `graphviz` (viz), `torch` (test comparisons), `pytest` (tests)

## Where things live

- `micrograd/engine.py` — the entire autograd engine (`Value` class, ~100 lines)
- `micrograd/nn.py` — `Neuron`, `Layer`, `MLP`
- `micrograd/viz.py` — Graphviz computational graph renderer
- `examples/` — four runnable demonstrations
- `tests/test_engine.py` — 17 PyTorch-verified gradient tests
- `docs/autograd.md` — mathematical guide to reverse-mode autodiff
- `docs/pytorch_comparison.md` — MiniGrad vs PyTorch deep dive

## User preferences

- Do NOT change `engine.py` or `nn.py` logic — only documentation/style improvements are welcome.
- Keep implementation minimal and educational (not production-grade).
- All new operations must have PyTorch-verified gradient tests.

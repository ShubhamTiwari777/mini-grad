"""
example_relu.py — ReLU activation and sub-gradient behaviour.

Run: PYTHONPATH=. python3 examples/example_relu.py
"""

from micrograd.engine import Value


def relu_demo(x_val: float) -> None:
    x = Value(x_val)
    y = x.relu()
    y.backward()
    expected = 1.0 if x_val > 0 else 0.0
    print(
        f"  x = {x_val:6.2f}  →  relu(x) = {y.data:.4f}"
        f"  |  dy/dx = {x.grad:.4f}  (expected {expected})"
    )


def main() -> None:
    print("=" * 55)
    print("  MiniGrad — ReLU Activation & Gradients")
    print("=" * 55)

    print("\nScalar ReLU examples:")
    for val in (-3.0, -0.001, 0.0, 0.001, 2.5, 10.0):
        relu_demo(val)

    print("\n" + "-" * 55)
    print("  Composed: f(a, b) = relu(a*b + 1)")
    print("-" * 55)

    a, b = Value(2.0), Value(3.0)
    out = (a * b + 1.0).relu()
    out.backward()

    # df/da = b = 3,  df/db = a = 2
    print(f"\n  a*b + 1       = {(a.data * b.data + 1):.1f}")
    print(f"  relu(a*b + 1) = {out.data}")
    print(f"  df/da         = {a.grad:.4f}  (expected 3.0)")
    print(f"  df/db         = {b.grad:.4f}  (expected 2.0)")

    print("\n" + "-" * 55)
    print("  Dead neuron: relu(-5) → gradient does NOT flow")
    print("-" * 55)

    w, x = Value(1.0), Value(-5.0)
    dead = (w * x).relu()
    dead.backward()
    print(f"\n  relu(w*x) = {dead.data}")
    print(f"  dout/dw   = {w.grad:.4f}  (expected 0.0 — dead neuron!)")


if __name__ == "__main__":
    main()

"""
example_basic.py — Basic arithmetic and backpropagation.

Run: PYTHONPATH=. python3 examples/example_basic.py
"""

from micrograd.engine import Value


def main() -> None:
    print("=" * 55)
    print("  MiniGrad — Basic Arithmetic & Backpropagation")
    print("=" * 55)

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b       # -6
    e = d + c       # 4

    print(f"\nForward pass:")
    print(f"  a       = {a.data}")
    print(f"  b       = {b.data}")
    print(f"  c       = {c.data}")
    print(f"  d = a*b = {d.data}")
    print(f"  e = d+c = {e.data}")

    e.backward()

    # de/da = b = -3,  de/db = a = 2,  de/dc = 1
    print(f"\nBackward pass (de/dx):")
    print(f"  e.grad = {e.grad:.4f}  (expected  1.0)")
    print(f"  d.grad = {d.grad:.4f}  (expected  1.0)")
    print(f"  c.grad = {c.grad:.4f}  (expected  1.0)")
    print(f"  a.grad = {a.grad:.4f}  (expected {b.data})")
    print(f"  b.grad = {b.grad:.4f}  (expected {a.data})")

    print("\n" + "-" * 55)
    print("  Chained expression: f(x) = (x + 2)² - 1")
    print("-" * 55)

    x = Value(3.0)
    f = (x + 2) ** 2 - 1   # f(3) = 24
    f.backward()            # df/dx = 2*(x+2) = 10

    print(f"\n  x     = {x.data}")
    print(f"  f(x)  = {f.data}   (expected 24.0)")
    print(f"  df/dx = {x.grad:.4f}  (expected 10.0)")


if __name__ == "__main__":
    main()

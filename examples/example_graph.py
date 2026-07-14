"""
example_graph.py — Visualise a computational graph with Graphviz.

Requires: pip install graphviz  +  system Graphviz binaries.
Run: PYTHONPATH=. python3 examples/example_graph.py
Output: graph_output.png
"""

import os

from micrograd.engine import Value
from micrograd.viz import draw_dot


def main() -> None:
    print("=" * 55)
    print("  MiniGrad — Computational Graph Visualisation")
    print("=" * 55)

    # f(a, b) = relu(a*b² + b)
    a = Value(2.0)
    b = Value(-3.0)
    out = (a * b ** 2 + b).relu()
    out.backward()

    print(f"\nf(a, b) = relu(a·b² + b)")
    print(f"  a      = {a.data}")
    print(f"  b      = {b.data}")
    print(f"  out    = {out.data}")
    print(f"  a.grad = {a.grad:.4f}")
    print(f"  b.grad = {b.grad:.4f}")

    dot = draw_dot(out, format="png", rankdir="LR")
    dot.render("graph_output", cleanup=True)
    print(f"\nGraph saved to: {os.path.abspath('graph_output.png')}")


if __name__ == "__main__":
    main()

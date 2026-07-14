"""
viz.py — Computational graph visualisation using Graphviz.

Requires: pip install graphviz  +  system Graphviz binaries.

Example
-------
>>> from micrograd.engine import Value
>>> from micrograd.viz import draw_dot
>>> a, b = Value(2.0), Value(-3.0)
>>> c = (a * b).relu()
>>> c.backward()
>>> draw_dot(c).render("graph", format="png", cleanup=True)
"""

from __future__ import annotations

from typing import Set, Tuple

from micrograd.engine import Value

try:
    from graphviz import Digraph
except ImportError as exc:
    raise ImportError(
        "graphviz is required for visualisation.\n"
        "Install it with:  pip install graphviz\n"
        "Then install system binaries: https://graphviz.org/download/"
    ) from exc


def _trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    """Collect all nodes and edges reachable from ``root``."""
    nodes: Set[Value] = set()
    edges: Set[Tuple[Value, Value]] = set()

    def _build(node: Value) -> None:
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.add((child, node))
                _build(child)

    _build(root)
    return nodes, edges


def draw_dot(root: Value, format: str = "svg", rankdir: str = "LR") -> "Digraph":
    """Build a Graphviz ``Digraph`` of the computation graph rooted at ``root``.

    Each node displays ``data`` and ``grad``; op-nodes show the operation.

    Parameters
    ----------
    root:
        The output ``Value`` (typically the loss).
    format:
        Output format: ``"svg"``, ``"png"``, ``"pdf"``…
    rankdir:
        Layout direction. ``"LR"`` (left-to-right) or ``"TB"`` (top-to-bottom).

    Returns
    -------
    graphviz.Digraph
        Render with ``.render()``, or display inline in Jupyter.
    """
    dot = Digraph(
        format=format,
        graph_attr={"rankdir": rankdir, "bgcolor": "#1e1e2e"},
        node_attr={"style": "filled", "fontname": "Helvetica", "fontsize": "11"},
    )

    nodes, edges = _trace(root)

    for node in nodes:
        node_id = str(id(node))
        label = f"{{ data = {node.data:.4f} | grad = {node.grad:.4f} }}"
        dot.node(
            node_id, label=label, shape="record",
            fillcolor="#313244", fontcolor="#cdd6f4", color="#89b4fa",
        )
        if node._op:
            op_id = node_id + node._op
            dot.node(
                op_id, label=node._op, shape="ellipse",
                fillcolor="#89b4fa", fontcolor="#1e1e2e", color="#89b4fa",
            )
            dot.edge(op_id, node_id, color="#a6e3a1")

    for src, dst in edges:
        dot.edge(str(id(src)), str(id(dst)) + dst._op, color="#f38ba8")

    return dot

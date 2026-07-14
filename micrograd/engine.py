"""
engine.py — Core scalar-valued autograd engine.

Implements the ``Value`` class: a node in a dynamic computational graph.
Each ``Value`` wraps a single float, tracks the operation that produced it,
and stores a closure that computes the local gradient contribution during
the backward pass.

The design mirrors PyTorch's ``Tensor`` at the scalar level:
  - Forward pass: build the graph via operator overloads.
  - Backward pass: call ``.backward()`` on the root; gradients flow back
    through the graph via the chain rule (reverse-mode autodiff).
"""

from __future__ import annotations

from typing import Callable, Set, Tuple, Union


# Scalar types that can be implicitly promoted to Value.
Scalar = Union[int, float]


class Value:
    """A scalar value node in a dynamic computational graph.

    Supports the following differentiable operations:
      ``+``, ``-``, ``*``, ``/``, ``**``, unary ``-``, and ``relu``.

    Parameters
    ----------
    data:
        The scalar (float or int) stored at this node.
    _children:
        The ``Value`` nodes that were inputs to the operation producing this
        node.  Empty for leaf nodes (inputs / parameters).
    _op:
        A human-readable string naming the operation, used for graph
        visualisation and debugging (e.g. ``'+'``, ``'*'``, ``'ReLU'``).

    Attributes
    ----------
    data : float
        The forward-pass value stored at this node.
    grad : float
        The gradient of the graph's root scalar with respect to this node.
        Accumulated by repeated calls to ``backward()`` or left at ``0``
        until ``backward()`` is invoked.

    Examples
    --------
    >>> a = Value(2.0)
    >>> b = Value(3.0)
    >>> c = a * b + a   # forward pass builds the graph
    >>> c.backward()    # backward pass fills .grad on every node
    >>> a.grad           # dc/da = b + 1 = 4.0
    4.0
    >>> b.grad           # dc/db = a = 2.0
    2.0
    """

    def __init__(
        self,
        data: Scalar,
        _children: Tuple["Value", ...] = (),
        _op: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0

        # The backward closure accumulates gradients into self and other.
        self._backward: Callable[[], None] = lambda: None

        # Nodes this value was computed from (inputs to the operation).
        self._prev: Set["Value"] = set(_children)

        # The operation that created this node ('' for leaf nodes).
        self._op: str = _op

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def __add__(self, other: Union["Value", Scalar]) -> "Value":
        """Element-wise addition: ``self + other``.

        Gradient rule (chain rule):
            d(out)/d(self)  = 1 · out.grad
            d(out)/d(other) = 1 · out.grad
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Value", Scalar]) -> "Value":
        """Element-wise multiplication: ``self * other``.

        Gradient rule:
            d(out)/d(self)  = other.data · out.grad
            d(out)/d(other) = self.data  · out.grad
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: Scalar) -> "Value":
        """Raise to a scalar power: ``self ** other``.

        Only integer and float exponents are supported.

        Gradient rule:
            d(out)/d(self) = other · self.data^(other-1) · out.grad
        """
        assert isinstance(other, (int, float)), (
            f"Only int/float exponents are supported; got {type(other).__name__}."
        )
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """Rectified Linear Unit: ``max(0, self)``.

        Gradient rule (sub-gradient at 0 → 0):
            d(out)/d(self) = (out.data > 0) · out.grad
        """
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass (reverse-mode autodiff)
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Compute gradients for all ancestors of this node.

        1. Topologically sort all nodes reachable from ``self``.
        2. Set ``self.grad = 1`` (∂self/∂self = 1).
        3. Walk nodes in reverse topological order, calling each node's
           ``_backward`` closure to distribute gradients via the chain rule.
        """
        topo: list["Value"] = []
        visited: Set["Value"] = set()

        def _build_topo(node: "Value") -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    _build_topo(child)
                topo.append(node)

        _build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Reflected / derived operators
    # ------------------------------------------------------------------

    def __neg__(self) -> "Value":
        """Unary negation: ``-self``."""
        return self * -1

    def __radd__(self, other: Union["Value", Scalar]) -> "Value":
        """Right-hand addition: ``other + self``."""
        return self + other

    def __sub__(self, other: Union["Value", Scalar]) -> "Value":
        """Subtraction: ``self - other``."""
        return self + (-other)

    def __rsub__(self, other: Union["Value", Scalar]) -> "Value":
        """Right-hand subtraction: ``other - self``."""
        return other + (-self)

    def __rmul__(self, other: Union["Value", Scalar]) -> "Value":
        """Right-hand multiplication: ``other * self``."""
        return self * other

    def __truediv__(self, other: Union["Value", Scalar]) -> "Value":
        """Division: ``self / other``."""
        return self * other ** -1

    def __rtruediv__(self, other: Union["Value", Scalar]) -> "Value":
        """Right-hand division: ``other / self``."""
        return other * self ** -1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

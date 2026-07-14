"""
nn.py — Neural network primitives built on top of the autograd engine.

Provides three composable building blocks:

  * :class:`Neuron`  — a single perceptron with optional ReLU activation.
  * :class:`Layer`   — a fully-connected layer of neurons.
  * :class:`MLP`     — a multi-layer perceptron (stack of ``Layer`` objects).
"""

from __future__ import annotations

import random
from typing import List, Union

from micrograd.engine import Value

Inputs = List[Union[float, Value]]


class Module:
    """Abstract base class for all neural-network components."""

    def zero_grad(self) -> None:
        """Set ``.grad`` to ``0`` for every learnable parameter."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        """Return a flat list of all learnable :class:`~micrograd.engine.Value` nodes."""
        return []


class Neuron(Module):
    """A single artificial neuron: ``activation(w · x + b)``.

    Parameters
    ----------
    nin:
        Number of input features.
    nonlin:
        If ``True`` (default), apply ReLU. Set ``False`` for output layers.
    """

    def __init__(self, nin: int, nonlin: bool = True) -> None:
        self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0.0)
        self.nonlin: bool = nonlin

    def __call__(self, x: Inputs) -> Value:
        act: Value = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """A fully-connected layer: a collection of neurons sharing the same inputs.

    Parameters
    ----------
    nin:
        Number of inputs to each neuron.
    nout:
        Number of neurons (= number of outputs).
    """

    def __init__(self, nin: int, nout: int, **kwargs) -> None:
        self.neurons: List[Neuron] = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: Inputs) -> Union[Value, List[Value]]:
        outs: List[Value] = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-Layer Perceptron: a stack of fully-connected layers.

    Hidden layers use ReLU; the final layer is linear (no activation).

    Parameters
    ----------
    nin:
        Number of input features.
    nouts:
        Neurons per layer, e.g. ``[8, 8, 1]``.

    Examples
    --------
    >>> model = MLP(2, [16, 16, 1])
    >>> pred = model([0.5, -1.2])
    >>> loss = (pred - 1.0) ** 2
    >>> loss.backward()
    """

    def __init__(self, nin: int, nouts: List[int]) -> None:
        sizes = [nin] + nouts
        self.layers: List[Layer] = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x: Inputs) -> Union[Value, List[Value]]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

"""
MiniGrad — A minimal scalar-valued autograd engine with neural network support.

Inspired by PyTorch's reverse-mode automatic differentiation, this package
demonstrates how backpropagation works under the hood.

Example
-------
>>> from micrograd.engine import Value
>>> x = Value(2.0)
>>> y = x ** 3 + 3 * x
>>> y.backward()
>>> print(x.grad)   # dy/dx = 3x² + 3 = 15.0
15.0
"""

from micrograd.engine import Value
from micrograd.nn import MLP, Layer, Neuron

__version__ = "1.0.0"
__author__ = "MiniGrad Contributors"
__all__ = ["Value", "MLP", "Layer", "Neuron"]

"""
test_engine.py — Comprehensive unit tests for the MiniGrad autograd engine.

Every test computes a gradient with MiniGrad and compares it against PyTorch
to within 1e-6 tolerance. Tests gracefully skip when PyTorch is not installed.

Run:
    PYTHONPATH=. pytest tests/ -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping comparison tests")

from micrograd.engine import Value  # noqa: E402

TOL = 1e-6


def _mg(val: float) -> Value:
    return Value(val)


def _pt(val: float) -> "torch.Tensor":
    t = torch.tensor(val, dtype=torch.float64)
    t.requires_grad_(True)
    return t


class TestAddition:

    def test_add_two_values(self):
        a_mg, b_mg = _mg(2.0), _mg(3.0)
        (a_mg + b_mg).backward()
        a_pt, b_pt = _pt(2.0), _pt(3.0)
        (a_pt + b_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL
        assert abs(b_mg.grad - b_pt.grad.item()) < TOL

    def test_add_scalar(self):
        a_mg = _mg(5.0)
        (a_mg + 3.0).backward()
        a_pt = _pt(5.0)
        (a_pt + 3.0).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

    def test_radd(self):
        a_mg = _mg(7.0)
        (3.0 + a_mg).backward()
        a_pt = _pt(7.0)
        (3.0 + a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

    def test_gradient_accumulation(self):
        a_mg = _mg(4.0)
        (a_mg + a_mg).backward()
        a_pt = _pt(4.0)
        (a_pt + a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL


class TestMultiplication:

    def test_mul_two_values(self):
        a_mg, b_mg = _mg(-2.0), _mg(5.0)
        (a_mg * b_mg).backward()
        a_pt, b_pt = _pt(-2.0), _pt(5.0)
        (a_pt * b_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL
        assert abs(b_mg.grad - b_pt.grad.item()) < TOL

    def test_mul_scalar(self):
        a_mg = _mg(3.0)
        (a_mg * 4.0).backward()
        a_pt = _pt(3.0)
        (a_pt * 4.0).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

    def test_rmul(self):
        a_mg = _mg(6.0)
        (7.0 * a_mg).backward()
        a_pt = _pt(6.0)
        (7.0 * a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

    def test_square_via_self_mul(self):
        a_mg = _mg(3.0)
        (a_mg * a_mg).backward()  # grad = 2a = 6
        a_pt = _pt(3.0)
        (a_pt * a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL


class TestDivision:

    def test_div(self):
        a_mg, b_mg = _mg(6.0), _mg(2.0)
        out_mg = a_mg / b_mg
        out_mg.backward()
        a_pt, b_pt = _pt(6.0), _pt(2.0)
        out_pt = a_pt / b_pt
        out_pt.backward()
        assert abs(out_mg.data - out_pt.item()) < TOL
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL
        assert abs(b_mg.grad - b_pt.grad.item()) < TOL

    def test_rdiv(self):
        a_mg = _mg(4.0)
        (1.0 / a_mg).backward()
        a_pt = _pt(4.0)
        (1.0 / a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL


class TestPower:

    def test_integer_power(self):
        a_mg = _mg(3.0)
        (a_mg ** 3).backward()
        a_pt = _pt(3.0)
        (a_pt ** 3).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

    def test_fractional_power(self):
        a_mg = _mg(4.0)
        (a_mg ** 0.5).backward()
        a_pt = _pt(4.0)
        (a_pt ** 0.5).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

    def test_negative_power(self):
        a_mg = _mg(2.0)
        (a_mg ** -2).backward()
        a_pt = _pt(2.0)
        (a_pt ** -2).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL


class TestReLU:

    def test_relu_positive(self):
        a_mg = _mg(3.0)
        a_mg.relu().backward()
        a_pt = _pt(3.0)
        torch.relu(a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL  # grad = 1

    def test_relu_negative(self):
        a_mg = _mg(-2.5)
        a_mg.relu().backward()
        a_pt = _pt(-2.5)
        torch.relu(a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL  # grad = 0 (dead)

    def test_relu_zero(self):
        a_mg = _mg(0.0)
        a_mg.relu().backward()
        assert a_mg.grad == 0.0


class TestComplexExpressions:

    def test_sanity_check(self):
        x_mg = _mg(-4.0)
        z = 2 * x_mg + 2 + x_mg
        q = z.relu() + z * x_mg
        h = (z * z).relu()
        y_mg = h + q + q * x_mg
        y_mg.backward()

        x_pt = _pt(-4.0)
        z = 2 * x_pt + 2 + x_pt
        q = torch.relu(z) + z * x_pt
        h = torch.relu(z * z)
        y_pt = h + q + q * x_pt
        y_pt.backward()

        assert abs(y_mg.data - y_pt.item()) < TOL
        assert abs(x_mg.grad - x_pt.grad.item()) < TOL

    def test_more_ops(self):
        a_mg, b_mg = _mg(-4.0), _mg(2.0)
        c = a_mg + b_mg
        d = a_mg * b_mg + b_mg ** 3
        c += c + 1
        c += 1 + c + (-a_mg)
        d += d * 2 + (b_mg + a_mg).relu()
        d += 3 * d + (b_mg - a_mg).relu()
        e = c - d
        f = e ** 2
        g_mg = f / 2.0
        g_mg += 10.0 / f
        g_mg.backward()

        a_pt, b_pt = _pt(-4.0), _pt(2.0)
        c = a_pt + b_pt
        d = a_pt * b_pt + b_pt ** 3
        c = c + c + 1
        c = c + 1 + c + (-a_pt)
        d = d + d * 2 + torch.relu(b_pt + a_pt)
        d = d + 3 * d + torch.relu(b_pt - a_pt)
        e = c - d
        f = e ** 2
        g_pt = f / 2.0 + 10.0 / f
        g_pt.backward()

        assert abs(g_mg.data - g_pt.item()) < TOL
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL
        assert abs(b_mg.grad - b_pt.grad.item()) < TOL

    def test_subtraction(self):
        a_mg, b_mg = _mg(7.0), _mg(3.0)
        out_mg = a_mg - b_mg
        out_mg.backward()
        a_pt, b_pt = _pt(7.0), _pt(3.0)
        out_pt = a_pt - b_pt
        out_pt.backward()
        assert abs(out_mg.data - out_pt.item()) < TOL
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL
        assert abs(b_mg.grad - b_pt.grad.item()) < TOL

    def test_polynomial(self):
        """f(x) = 3x³ - 2x + 5,  f'(x) = 9x² - 2 = 34 at x=2"""
        x_mg = _mg(2.0)
        (3 * x_mg ** 3 - 2 * x_mg + 5).backward()
        x_pt = _pt(2.0)
        (3 * x_pt ** 3 - 2 * x_pt + 5).backward()
        assert abs(x_mg.grad - x_pt.grad.item()) < TOL

    def test_diamond_graph(self):
        """Node shared by multiple paths — gradient must accumulate."""
        a_mg = _mg(3.0)
        b_mg = a_mg * a_mg     # a²
        (a_mg * b_mg).backward()  # a³ → dc/da = 3a² = 27
        a_pt = _pt(3.0)
        b_pt = a_pt * a_pt
        (a_pt * b_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL


class TestGradientAccumulation:

    def test_repeated_use(self):
        a_mg = _mg(2.0)
        (a_mg + a_mg + a_mg).backward()  # df/da = 3
        a_pt = _pt(2.0)
        (a_pt + a_pt + a_pt).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

    def test_multiple_backward_paths(self):
        """f(a) = a² + a³,  f'(a) = 2a + 3a² = 16 at a=2"""
        a_mg = _mg(2.0)
        (a_mg ** 2 + a_mg ** 3).backward()
        a_pt = _pt(2.0)
        (a_pt ** 2 + a_pt ** 3).backward()
        assert abs(a_mg.grad - a_pt.grad.item()) < TOL

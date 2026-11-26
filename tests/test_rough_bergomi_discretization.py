"""Unit tests for src/rough_bergomi_discretization.py: discretised
variational problem (4.6)-(4.7) and the analytical linear approximation of
Proposition 4.1 (Section 4.3, Figure 4.1)."""

import numpy as np
import pytest

from src.rough_bergomi_discretization import (
    J_H_n,
    analytical_linear_approximation,
    kernel_weights_rough_bergomi,
    rescaled_implied_vol_from_J,
    rescaled_implied_vol_H_half,
)


def test_kernel_weights_shape_and_causality():
    H, n = 0.3, 10
    W = kernel_weights_rough_bergomi(H, n)
    assert W.shape == (n, n)
    for i in range(n):
        for j in range(i + 1, n):
            assert W[i, j] == 0.0


def test_J_H_n_zero_at_y_zero_limit():
    """As y -> 0 the constraint is trivially satisfied by g=0, so J_{H,n}(y)
    should shrink toward 0."""
    H, n = 0.3, 20
    out_small = J_H_n(y=1e-3, H=H, n=n, v0=1.0, eta=2.0)
    assert out_small["success"]
    assert out_small["J"] < 0.05


def test_J_H_n_positive_for_nonzero_y():
    H, n = 0.3, 15
    out = J_H_n(y=0.3, H=H, n=n, v0=1.0, eta=2.0)
    assert out["success"]
    assert out["J"] > 0


def test_rescaled_implied_vol_H_half_symmetric_small_y():
    """At small |y|, the rescaled IV should be close for +y/-y given the
    quadratic leading term of J at H=1/2."""
    y = 0.05
    up = rescaled_implied_vol_H_half(y)
    down = rescaled_implied_vol_H_half(-y)
    assert up == pytest.approx(down, rel=0.05)


def test_analytical_linear_approximation_matches_level_at_y_zero():
    H, v0, eta = 0.3, 1.0, 2.0
    y = np.array([0.0])
    val = analytical_linear_approximation(y, H, v0, eta)
    from src.implied_vol import atm_vol_level

    expected_level = atm_vol_level(H, v0, v0 * eta)
    assert val[0] == pytest.approx(expected_level, rel=1e-6)


def test_rescaled_implied_vol_from_J_nan_at_zero():
    assert np.isnan(rescaled_implied_vol_from_J(0.0, 1.0))

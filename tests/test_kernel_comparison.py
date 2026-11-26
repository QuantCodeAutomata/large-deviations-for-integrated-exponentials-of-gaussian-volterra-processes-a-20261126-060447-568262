"""Unit tests for src/kernel_comparison.py: Proposition A.1, Theorem A.3,
and the Lemma 4.4 identity I(H) = R(H)/(2(H+1/2)^2)."""

import numpy as np
import pytest

from src.implied_vol import I_of_H_lemma_4_4
from src.kernel_comparison import (
    chi,
    chi_asymptotics_near_1,
    kappa_bounds,
    ode_residual_A2,
    variance_ratio_MG_vs_RL,
)
from src.rate_expansion import R_of_H
from src.variance_options import c_H_constant


def test_chi_at_one_equals_c_H():
    H = 0.3
    assert chi(1.0, H) == pytest.approx(c_H_constant(H), rel=1e-6)


def test_chi_strictly_increasing():
    H = 0.3
    xs = np.geomspace(1.0, 50.0, 25)
    vals = np.array([chi(x, H) for x in xs])
    assert np.all(np.diff(vals) > 0)


def test_chi_matches_asymptotics_near_one():
    H = 0.3
    x = 1.0 + 1e-4
    assert chi(x, H) == pytest.approx(chi_asymptotics_near_1(x, H), rel=1e-3)


def test_ode_residual_A2_near_zero():
    H = 0.3
    for x in (1.5, 2.0, 5.0, 10.0):
        assert abs(ode_residual_A2(x, H)) < 1e-3


def test_kappa_bounds_bracket_chi():
    H = 0.3
    x_grid = np.geomspace(1.001, 100.0, 40)
    bounds = kappa_bounds(H, x_grid)
    vals = np.array([chi(x, H) for x in x_grid]) * x_grid ** (H - 0.5)
    assert np.all(vals >= bounds["kappa_H"] - 1e-9)
    assert np.all(vals <= bounds["kappa_bar_H"] + 1e-9)


def test_variance_ratio_equals_one_only_at_half():
    assert variance_ratio_MG_vs_RL(0.5) == pytest.approx(1.0)
    assert variance_ratio_MG_vs_RL(0.3) != pytest.approx(1.0, abs=1e-3)


def test_lemma_4_4_identity_via_kernel_comparison_module():
    for H in (0.1, 0.25, 0.4):
        I = I_of_H_lemma_4_4(H)
        R = R_of_H(H)
        assert I == pytest.approx(R / (2 * (H + 0.5) ** 2), rel=1e-6)

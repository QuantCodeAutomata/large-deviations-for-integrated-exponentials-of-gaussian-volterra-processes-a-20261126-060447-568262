"""Unit tests for src/atm_asymptotics.py: Theorem 3.13, Corollary 3.15,
Remark 3.14."""

import numpy as np
import pytest

from src.atm_asymptotics import (
    atm_coefficient_cor_3_15,
    atm_coefficient_degenerate_remark_3_14,
    atm_coefficient_thm_3_13,
    half_normal_mean,
    integral_W_H_variance,
)


def test_atm_coefficient_thm_3_13_scales_with_derivative():
    H = 0.3
    c1 = atm_coefficient_thm_3_13(1.0, H)
    c2 = atm_coefficient_thm_3_13(2.0, H)
    assert c2 == pytest.approx(2 * c1)
    assert c1 > 0


def test_atm_coefficient_cor_3_15_positive():
    for H in (0.1, 0.3, 0.5):
        assert atm_coefficient_cor_3_15(H) > 0


def test_atm_coefficient_cor_3_15_matches_closed_form():
    """Coefficient 1/((2H+1) sqrt(pi(H+1))) from Corollary 3.15."""
    H = 0.3
    expected = 1.0 / ((2 * H + 1) * np.sqrt(np.pi * (H + 1)))
    assert atm_coefficient_cor_3_15(H) == pytest.approx(expected, rel=1e-10)


def test_degenerate_case_call_put_are_disjoint_nonneg():
    H = 0.3
    call = atm_coefficient_degenerate_remark_3_14(0.5, H, "call")
    put = atm_coefficient_degenerate_remark_3_14(0.5, H, "put")
    assert call > 0 and put == 0.0

    call2 = atm_coefficient_degenerate_remark_3_14(-0.5, H, "call")
    put2 = atm_coefficient_degenerate_remark_3_14(-0.5, H, "put")
    assert call2 == 0.0 and put2 > 0


def test_integral_W_H_variance_scaling():
    H = 0.3
    v1 = integral_W_H_variance(1.0, H)
    v2 = integral_W_H_variance(2.0, H)
    assert v2 == pytest.approx(v1 * 2 ** (2 * H + 2))


def test_half_normal_mean_formula():
    sigma = 2.0
    assert half_normal_mean(sigma) == pytest.approx(sigma / np.sqrt(2 * np.pi))

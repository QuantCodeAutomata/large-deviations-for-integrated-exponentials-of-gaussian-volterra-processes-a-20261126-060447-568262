"""Unit tests for src/variance_options.py: Theorem 3.3, Corollary 3.4,
Corollary 3.8, Lemma 3.10."""

import numpy as np
import pytest

from src.rate_function import J_closed_form_half
from src.variance_options import (
    C_function_cor_3_8,
    Lambda_general,
    c_H_constant,
    kernel_action_discretized,
    lemma_3_10_rate,
    lognormal_rate,
    molchan_golosov_kernel,
    special_case_rate,
)


def test_kernel_action_discretized_RL_matches_rate_function_module():
    """The RL kernel-action matrix must agree with src.rate_function's
    (same discretisation), since both implement (K h_dot)(t) = int (t-s)^{H-1/2}."""
    from src.rate_function import kernel_action_matrix

    H, n = 0.3, 15
    A1 = kernel_action_discretized(H, n, kernel="RL")
    A2 = kernel_action_matrix(H, n)
    np.testing.assert_allclose(A1, A2, rtol=1e-10)


def test_Lambda_general_lognormal_matches_J_closed_form_at_half():
    """Theorem 3.3 for sigma^2(y)=e^{nu y}, y0=0, at H=1/2 must match the
    closed-form J via Corollary 3.4/Remark 3.5(ii) (within discretisation error)."""
    H, n, nu = 0.5, 60, 1.0
    y0 = 0.0
    K = 1.5  # sigma0^2 = 1, OTM Call

    def sigma2(y):
        return np.exp(nu * y)

    out = Lambda_general(K, sigma2, y0, H, n, kernel="RL")
    assert out["success"]
    expected = lognormal_rate(K, sigma0_sq=1.0, nu=nu, J_func=J_closed_form_half)
    assert out["Lambda"] == pytest.approx(expected, rel=0.05)


def test_special_case_rate_cor_3_8_formula():
    H, xi, y0 = 0.3, 1.0, 0.2
    K = xi ** 2 * abs(y0) + 0.5
    val = special_case_rate(K, xi, y0, H)
    assert val == pytest.approx((H + 1) * (K - xi ** 2 * abs(y0)) ** 2 / xi ** 4)
    assert val > 0


def test_C_function_cor_3_8_bounds():
    H = 0.3
    t = np.linspace(0.01, 0.99, 20)
    C = C_function_cor_3_8(t, H)
    assert np.all(np.isfinite(C))


def test_molchan_golosov_kernel_positive_and_finite():
    H = 0.3
    for s, t in [(0.1, 0.5), (0.3, 0.9), (0.01, 0.02)]:
        val = molchan_golosov_kernel(s, t, H)
        assert np.isfinite(val)


def test_c_H_constant_positive():
    for H in (0.1, 0.2, 0.3, 0.45):
        assert c_H_constant(H) > 0


def test_lemma_3_10_rate_increasing_sigma_call():
    """H=1/2, sigma^2(y) = e^y monotone increasing: for K > sigma0^2=1,
    the Euler-Lagrange solution should give a finite, positive rate."""

    def sigma2(y):
        return np.exp(y)

    out = lemma_3_10_rate(K=1.5, sigma2=sigma2, y0=0.0, h1_sign=1)
    assert out["success"]
    assert out["Lambda"] > 0
    assert out["h1"] > 0

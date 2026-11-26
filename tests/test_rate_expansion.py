"""Unit tests for src/rate_expansion.py: R(H) (eq. 3.14), Lemma 3.20, Cor. 3.22."""

import pytest

from src.rate_expansion import (
    R_lower_bound,
    R_of_H,
    integral_identity_remark_3_21,
    j2_j3,
    rate_function_expansion,
    rough_bergomi_j2_j3,
)


def test_R_of_H_half_matches_known_value():
    """R(1/2) = 2/15 is stated explicitly in the paper (Remark 2.16(iv)/4.3)."""
    assert R_of_H(0.5) == pytest.approx(2 / 15, rel=1e-4)


def test_lemma_3_20_lower_bound_strict():
    for H in (0.05, 0.1, 0.3, 0.5):
        R = R_of_H(H)
        bound = R_lower_bound(H)
        assert R > bound


def test_integral_identity_remark_3_21():
    for H in (0.1, 0.3, 0.5):
        value = integral_identity_remark_3_21(H)
        assert value == pytest.approx(1.0 / (2 * (H + 1)), rel=1e-6)


def test_rough_bergomi_j2_j3_half_matches_paper_numbers():
    """Remark 3.23: at H=1/2, eta=2: j2=3/8, j3=-3/40 (text eta=2 example in
    Corollary 3.22/Remark 3.23: j2=3/(2 eta^2), j3=-3/(10 eta^2))."""
    out = rough_bergomi_j2_j3(0.5, eta=2.0)
    assert out["j2"] == pytest.approx(3 / (2 * 2.0 ** 2), rel=1e-4)
    assert out["j3"] == pytest.approx(-3 / (10 * 2.0 ** 2), rel=1e-4)


def test_j3_negative_in_rough_bergomi_for_all_H():
    """Corollary 3.22 + Lemma 3.20: j3 <= 0 in the rough Bergomi model."""
    for H in (0.05, 0.1, 0.3, 0.5):
        out = rough_bergomi_j2_j3(H, eta=2.0)
        assert out["j3"] <= 1e-10


def test_rate_function_expansion_vectorized():
    import numpy as np

    y = np.array([-0.1, 0.0, 0.1])
    out = rate_function_expansion(y, j2=1.0, j3=0.5)
    expected = 1.0 * y ** 2 + 0.5 * y ** 3
    np.testing.assert_allclose(out, expected)

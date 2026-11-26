"""Unit tests for src/clt_mdp.py: Theorem 2.14 (CLT), Theorem 2.15 (MDP)."""

import pytest

from src.clt_mdp import clt_variance, mdp_rate_function, mdp_rate_function_coefficient


def test_clt_variance_half_phi0_zero_matches_remark_2_16_iv():
    """Remark 2.16(iv): H=1/2, phi=0 gives limiting variance 1/3."""
    assert clt_variance(0.5, 0.0) == pytest.approx(1 / 3)


def test_mdp_rate_function_quadratic_and_zero_at_origin():
    H = 0.3
    assert mdp_rate_function(0.0, H) == 0.0
    coeff = mdp_rate_function_coefficient(H)
    for u in (0.5, -0.5, 1.0):
        assert mdp_rate_function(u, H) == pytest.approx(coeff * u ** 2)


def test_mdp_rate_function_positive_away_from_zero():
    for H in (0.1, 0.3, 0.5):
        assert mdp_rate_function(1.0, H) > 0.0
        assert mdp_rate_function(-1.0, H) > 0.0

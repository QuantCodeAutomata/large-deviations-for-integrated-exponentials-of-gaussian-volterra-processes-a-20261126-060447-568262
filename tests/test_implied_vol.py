"""Unit tests for src/implied_vol.py: Proposition 4.1, Lemma 4.4, Cor. 4.3."""

import pytest

from src.implied_vol import (
    I_of_H_lemma_4_4,
    atm_skew,
    atm_vol_level,
    black_scholes_call,
    implied_vol_from_price,
    lemma_4_4_check,
    mixed_bergomi_v1_v2,
    ratio_2v0v2_over_v1sq,
)
from src.rate_expansion import rough_bergomi_j2_j3


def test_atm_vol_level_half_matches_paper_example():
    """H=1/2, v0=1, v1=eta=2: sigma_ATM = 1/(1)*sqrt((0.5)/(1.5))*2 = 2/sqrt(3)."""
    val = atm_vol_level(0.5, v0=1.0, v1=2.0)
    assert val == pytest.approx(2.0 / 3 ** 0.5, rel=1e-6)


def test_atm_skew_half_matches_paper_example():
    jj = rough_bergomi_j2_j3(0.5, eta=2.0)
    skew = atm_skew(jj["j2"], jj["j3"])
    assert skew == pytest.approx(2.0 / (10 * 3 ** 0.5), rel=1e-3)


def test_lemma_4_4_identity_holds_numerically():
    for H in (0.1, 0.2, 0.3, 0.4, 0.49):
        out = lemma_4_4_check(H)
        assert out["abs_diff"] < 1e-6


def test_mixed_bergomi_jensen_ratio_at_least_one():
    """Corollary 4.3: 2 v0 v2 / v1^2 = sum w_i eta_i^2 / (sum w_i eta_i)^2 >= 1."""
    v0 = 1.0
    out = mixed_bergomi_v1_v2(v0, weights=[0.5, 0.5], etas=[1.0, 3.0])
    ratio = ratio_2v0v2_over_v1sq(v0, out["v1"], out["v2"])
    assert ratio >= 1.0 - 1e-10


def test_black_scholes_call_monotone_in_vol():
    F, K, T = 1.0, 1.0, 1.0
    low = black_scholes_call(F, K, T, 0.1)
    high = black_scholes_call(F, K, T, 0.5)
    assert high > low


def test_implied_vol_round_trip():
    F, K, T, sigma_true = 1.0, 1.05, 0.5, 0.3
    price = black_scholes_call(F, K, T, sigma_true)
    recovered = implied_vol_from_price(price, F, K, T)
    assert recovered == pytest.approx(sigma_true, rel=1e-5)

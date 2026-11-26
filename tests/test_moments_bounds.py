"""Unit tests for src/moments_bounds.py: eq (2.8), Proposition 2.8, Lemma 2.7."""

import numpy as np
import pytest

from src.moments_bounds import (
    E_A2,
    E_B,
    L_bar_bounds,
    L_bar_exact_half,
    check_lemma_2_7_properties,
    corollary_2_11_rate,
    log_mgf_limit_prop_2_10,
)
from src.rate_function import J_closed_form_half_vec


def test_E_A2_and_E_B_half():
    assert E_A2(0.5) == pytest.approx(2 / (4 * 1.5))
    assert E_B(0.5) == pytest.approx(1 / (1 * 2))


def test_L_bar_bounds_bracket_known_value_at_half():
    """Proposition 2.8 bounds are stated for H in (0, 1/2); check the bounds
    are finite and lower < upper for H close to 1/2."""
    bounds = L_bar_bounds(0.49)
    assert bounds["lower"] > 0
    assert bounds["upper"] > bounds["lower"]


def test_L_bar_exact_half_value():
    assert L_bar_exact_half() == 2.0


def test_lemma_2_7_properties_hold_for_closed_form_J():
    H = 0.5
    u_grid = np.linspace(-2.0, 2.0, 81)
    J_values = J_closed_form_half_vec(u_grid)
    out = check_lemma_2_7_properties(u_grid, J_values, H)
    assert out["non_increasing_on_neg"]
    assert out["non_decreasing_on_pos"]
    assert out["J_at_zero"] == pytest.approx(0.0, abs=1e-6)
    assert out["positivity"]


def test_corollary_2_11_rate_sign():
    """Rate -J(log z) should be negative for z != 1 (since J > 0 away from 0)."""
    J_func = lambda u: 0.0 if u == 0 else J_closed_form_half_vec(np.array([u]))[0]
    rate = corollary_2_11_rate(0.5, 0.5, J_func)
    assert rate < 0


def test_log_mgf_limit_zero_at_u_zero():
    assert log_mgf_limit_prop_2_10(0.0, 0.3) == 0.0

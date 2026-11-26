"""Unit tests for src/rate_function.py: Definition 2.4, Proposition 2.12, 2.6."""

import numpy as np
import pytest

from src.rate_function import (
    J_closed_form_half,
    J_closed_form_half_vec,
    fenchel_legendre_J_star,
    kernel_action_matrix,
    solve_J_general,
)


def test_J_closed_form_half_at_zero():
    assert J_closed_form_half(0.0) == 0.0


def test_J_closed_form_half_positive_away_from_zero():
    for u in (-2.0, -1.0, -0.1, 0.1, 1.0, 2.0):
        assert J_closed_form_half(u) > 0.0


def test_J_closed_form_half_monotonicity():
    """Lemma 2.7(i): J non-increasing on (-inf,0], non-decreasing on [0,inf)."""
    u_pos = np.linspace(0.0, 2.0, 25)
    J_pos = J_closed_form_half_vec(u_pos)
    assert np.all(np.diff(J_pos) >= -1e-9)

    u_neg = np.linspace(-1.5, 0.0, 25)
    J_neg = J_closed_form_half_vec(u_neg)
    assert np.all(np.diff(J_neg) <= 1e-9)


def test_J_closed_form_half_small_u_quadratic_limit():
    """As u -> 0, J(u) ~ u^2/(2 E[A^2]) with E[A^2]=2/((2*0.5+1)^2*1.5)=4/9
    (eq. 2.8 at H=1/2), i.e. J(u) ~ (9/8) u^2, consistent with the MDP
    quadratic rate function of Theorem 2.15 at H=1/2: (1)^2 * 1.5 = 1.5...
    use the direct ratio test instead of a hardcoded constant."""
    u = 1e-4
    J = J_closed_form_half(u)
    from src.moments_bounds import E_A2

    expected = u ** 2 / (2 * E_A2(0.5))
    assert J == pytest.approx(expected, rel=1e-3)


def test_kernel_action_matrix_shape_and_lower_triangular_causality():
    H, n = 0.3, 20
    A = kernel_action_matrix(H, n)
    assert A.shape == (n, n)
    # Causality: increment g_j affects (K h_dot)(t_i) only for j <= i (t_i >= (j-1)/n midpoint).
    for i in range(n):
        for j in range(i + 1, n):
            assert A[i, j] == 0.0


def test_solve_J_general_matches_closed_form_at_half():
    """At H=1/2 the general discretised variational problem should match
    the closed-form Proposition 2.12 solution (within discretisation error)."""
    H = 0.5
    for u in (-0.5, 0.5, 1.0):
        res = solve_J_general(u, H, n=80)
        assert res["success"]
        exact = J_closed_form_half(u)
        assert res["J"] == pytest.approx(exact, rel=0.03)


def test_fenchel_legendre_finite():
    J_func = lambda u: J_closed_form_half(u) if u != 0 else 0.0
    for x in (-1.0, 0.0, 1.0, 2.0):
        out = fenchel_legendre_J_star(x, J_func)
        assert np.isfinite(out["J_star"])

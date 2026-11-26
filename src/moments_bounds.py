"""Moments E[A^2], E[B] (eq. 2.8), bounds on L-bar (Prop. 2.8), Lemma 2.7
properties, Prop. 2.10 (log-mgf limit) and Cor. 2.11 (tail asymptotics).

Custom implementation of closed-form expressions from the paper; no
library provides these paper-specific constants.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma


def E_A2(H: float) -> float:
    """E[A^2] = 2 / ((2H+1)^2 (H+1)), A = int_0^1 X_s ds (eq. 2.8)."""
    return 2.0 / ((2 * H + 1) ** 2 * (H + 1))


def E_B(H: float) -> float:
    """E[B] = 1 / (2H(2H+1)), B = int_0^1 X_s^2 ds (eq. 2.8)."""
    return 1.0 / (2 * H * (2 * H + 1))


def L_bar_bounds(H: float) -> dict:
    """Lower and upper bounds for the constant L-bar from Proposition 2.8,
    valid for H in (0, 1/2)."""
    lower = (H + 0.5) ** 2 * (H + 1) / (H ** 2 * np.e ** 2)
    upper = (
        gamma(2 * H) / (np.sin(np.pi * H) * gamma(H + 0.5) ** 2)
        * (1 - H) ** (2 * (H - 1))
        / (4 * H ** (2 * H + 1))
    )
    return {"lower": float(lower), "upper": float(upper)}


def L_bar_exact_half() -> float:
    """At H = 1/2, L-bar = 2 (Remark 2.9)."""
    return 2.0


def check_lemma_2_7_properties(u_grid: np.ndarray, J_values: np.ndarray, H: float) -> dict:
    """Verify Lemma 2.7(i): J non-increasing on (-inf,0], non-decreasing on
    [0,inf), J(0)=0, J(u)>0 for u!=0, and J(u) >= u^2/(2E[A^2]) for u<=0."""
    u_grid = np.asarray(u_grid)
    J_values = np.asarray(J_values)
    neg_mask = u_grid <= 0
    pos_mask = u_grid >= 0
    neg_sorted = np.argsort(u_grid[neg_mask])
    pos_sorted = np.argsort(u_grid[pos_mask])
    non_increasing_on_neg = bool(np.all(np.diff(J_values[neg_mask][neg_sorted]) <= 1e-6))
    non_decreasing_on_pos = bool(np.all(np.diff(J_values[pos_mask][pos_sorted]) >= -1e-6))
    zero_idx = np.argmin(np.abs(u_grid))
    J_at_zero = float(J_values[zero_idx])
    positivity = bool(np.all(J_values[u_grid != 0] > -1e-8))
    ea2 = E_A2(H)
    lower_bound_ok = bool(np.all(J_values[neg_mask] >= u_grid[neg_mask] ** 2 / (2 * ea2) - 1e-6))
    return {
        "non_increasing_on_neg": non_increasing_on_neg,
        "non_decreasing_on_pos": non_decreasing_on_pos,
        "J_at_zero": J_at_zero,
        "positivity": positivity,
        "lower_bound_quadratic_ok": lower_bound_ok,
    }


def log_mgf_limit_prop_2_10(u: float, H: float) -> float:
    """Proposition 2.10: lim (1/s_eps^2) log E[(Z_eps/(eps e^phi0))^{-u}]
    = (u(u+1)E[A^2] - u E[B]) / 2."""
    return (u * (u + 1) * E_A2(H) - u * E_B(H)) / 2.0


def corollary_2_11_rate(z: float, H: float, J_func) -> float:
    """Corollary 2.11: lim s_eps^2 log P(Z_eps/(eps e^phi0) <= z) = -J(log z)
    for z in (0,1)."""
    return -J_func(np.log(z))

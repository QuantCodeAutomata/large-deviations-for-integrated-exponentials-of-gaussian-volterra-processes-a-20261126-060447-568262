"""Discretised variational problem for the rough Bergomi model (Section 4.3,
eq. 4.6-4.7), and the analytical linear implied-volatility approximation of
Proposition 4.1, reproducing Figure 4.1.

Custom implementation of the paper's discretisation scheme.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .implied_vol import atm_skew, atm_vol_level
from .rate_expansion import rough_bergomi_j2_j3
from .rate_function import J_closed_form_half


def kernel_weights_rough_bergomi(H: float, n: int) -> np.ndarray:
    """Weight matrix W (n x n) such that (K_H h_dot)((2i-1)/(2n)) =
    sqrt(2H) n/(H+1/2) sum_j W[i,j] g_j, from the piecewise-linear h
    discretisation of eq. (4.7) (midpoint rule, Riemann-Liouville kernel
    sqrt(2H)(t-s)^{H-1/2} with V[X_t]=t^{2H})."""
    power = H + 0.5
    i_idx = np.arange(1, n + 1)
    t_mid = (2 * i_idx - 1) / (2 * n)
    W = np.zeros((n, n))
    for i in range(1, n + 1):
        t = t_mid[i - 1]
        for j in range(1, i + 1):
            if j < i:
                W[i - 1, j - 1] = (t - (j - 1) / n) ** power - (t - j / n) ** power
            else:
                W[i - 1, j - 1] = (t - (i - 1) / n) ** power
    coeff = np.sqrt(2 * H) * n / power
    return coeff * W


def J_H_n(y: float, H: float, n: int, v0: float, eta: float, g0: np.ndarray | None = None) -> dict:
    """Discretised rate function J_{H,n}(y) of (4.6)-(4.7) for the rough
    Bergomi model sigma^2(y) = v0 e^{eta y}: minimise (n/2) sum g_i^2
    subject to the mid-point quadrature of int_0^1 sigma^2((K_H h_dot)(t)) dt
    equal to K = v0 e^y."""
    W = kernel_weights_rough_bergomi(H, n)
    K = v0 * np.exp(y)
    if g0 is None:
        g0 = np.full(n, np.sign(y) * 1e-3 + 1e-6)

    def objective(g):
        return 0.5 * n * np.sum(g ** 2)

    def objective_grad(g):
        return n * g

    def constraint(g):
        paths = W @ g
        return np.mean(v0 * np.exp(eta * paths)) - K

    res = minimize(objective, g0, jac=objective_grad,
                    constraints=[{"type": "eq", "fun": constraint}],
                    method="SLSQP", options={"maxiter": 500, "ftol": 1e-13})
    return {"J": float(res.fun), "success": bool(res.success), "g": res.x, "message": res.message}


def rescaled_implied_vol_from_J(y: float, J_value: float) -> float:
    """sigma_BS^2(K,T) ~ y^2 T^{2(H-1/2)} / (2 J(y)); the rescaled implied
    volatility sigma_BS(K,T) T^{-(H-1/2)} = |y| / sqrt(2 J(y))."""
    if abs(y) < 1e-12:
        return np.nan
    return abs(y) / np.sqrt(2 * J_value)


def rescaled_implied_vol_H_half(y: float) -> float:
    """H=1/2 exact benchmark via the closed-form J (Proposition 2.12)."""
    J_value = J_closed_form_half(y)
    return rescaled_implied_vol_from_J(y, J_value)


def analytical_linear_approximation(y: np.ndarray, H: float, v0: float, eta: float) -> np.ndarray:
    """Proposition 4.1 linear approximation of the rescaled implied
    volatility: sigma_ATM^{rescaled} + s^{rescaled} * y."""
    v1 = v0 * eta
    level = atm_vol_level(H, v0, v1)
    jj = rough_bergomi_j2_j3(H, eta)
    skew = atm_skew(jj["j2"], jj["j3"])
    return level + skew * np.asarray(y, dtype=float)

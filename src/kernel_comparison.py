"""Appendix A: comparison of the Molchan-Golosov and Riemann-Liouville
kernels via the linking function chi(x) (Proposition A.1, Theorem A.3).

Custom implementation of the paper's kernel-ratio function; scipy.special
and scipy.integrate supply the Beta function and quadrature only.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_fn

from .variance_options import c_H_constant


def chi(x: float, H: float) -> float:
    """chi(x) = c_H [ x^{H-1/2} + (1/2-H)(x-1)^{1/2-H} int_1^x v^{H-3/2}(v-1)^{H-1/2} dv ],
    x = t/s >= 1 (Proposition A.1)."""
    if x <= 1.0:
        return c_H_constant(H)
    c_H = c_H_constant(H)

    def integrand(v):
        return v ** (H - 1.5) * (v - 1) ** (H - 0.5)

    integral, _ = quad(integrand, 1.0, x, limit=200)
    return float(c_H * (x ** (H - 0.5) + (0.5 - H) * (x - 1) ** (0.5 - H) * integral))


def chi_vec(x_grid: np.ndarray, H: float) -> np.ndarray:
    return np.array([chi(float(x), H) for x in x_grid])


def chi_derivative_numeric(x: float, H: float, dx: float = 1e-5) -> float:
    """Numerical derivative of chi for the ODE check (A.2)."""
    return (chi(x + dx, H) - chi(x - dx, H)) / (2 * dx)


def ode_residual_A2(x: float, H: float) -> float:
    """Residual of the ODE (A.2): (x-1) chi'(x) + (H-1/2) chi(x)
    - c_H (H-1/2) x^{H-1/2}, should be ~ 0."""
    c_H = c_H_constant(H)
    chi_x = chi(x, H)
    chi_prime = chi_derivative_numeric(x, H)
    return (x - 1) * chi_prime + (H - 0.5) * chi_x - c_H * (H - 0.5) * x ** (H - 0.5)


def chi_asymptotics_near_1(x: float, H: float) -> float:
    """Theorem A.3(ii): as x->1, chi(x) = c_H[1 + ((1/2-H)^2/(H+1/2))(x-1) + O((x-1)^2)]."""
    c_H = c_H_constant(H)
    return c_H * (1 + ((0.5 - H) ** 2 / (H + 0.5)) * (x - 1))


def chi_asymptotics_infinity(x: float, H: float) -> float:
    """Theorem A.3(iii): as x->infinity, chi(x) ~ (H/c_H) x^{1/2-H}."""
    c_H = c_H_constant(H)
    return (H / c_H) * x ** (0.5 - H)


def kappa_bounds(H: float, x_grid: np.ndarray) -> dict:
    """Theorem A.3(iv): kappa_H = min_{x>=1} chi(x) x^{H-1/2},
    kappa_bar_H = max_{x>=1} chi(x) x^{H-1/2}."""
    vals = chi_vec(x_grid, H) * np.asarray(x_grid) ** (H - 0.5)
    return {"kappa_H": float(np.min(vals)), "kappa_bar_H": float(np.max(vals))}


def variance_ratio_MG_vs_RL(H: float) -> float:
    """Ratio of V[int_0^1 W_t^H dt] = 1/(2(H+1)) (Molchan-Golosov fBm) to
    E[A^2] = 2/((2H+1)^2(H+1)) (Riemann-Liouville), equal to
    (2H+1)^2 / 4, equal to 1 only at H=1/2."""
    return (2 * H + 1) ** 2 / 4.0

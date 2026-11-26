"""Out-of-the-money variance option short-maturity asymptotics (Section 3.1):
Theorem 3.3 (general sigma), Corollary 3.4 (lognormal), Corollary 3.8
(sigma(y)=xi sqrt|y|), Lemma 3.10 (H=1/2, general monotone sigma^2).

Custom implementation: these are paper-specific variational problems with a
nonlinear constraint on sigma^2 of a fractional-kernel path; no
quantitative-finance library solves them, so Theorem 3.3 is discretised
with scipy.optimize.minimize (SLSQP) and Lemma 3.10 solves the
Euler-Lagrange integral equations (3.6)-(3.7) with scipy.integrate.quad and
scipy.optimize.brentq.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import beta as beta_fn


# ---------------------------------------------------------------------------
# Molchan-Golosov kernel for fractional Brownian motion, H in (0, 1/2)
# ---------------------------------------------------------------------------
def c_H_constant(H: float) -> float:
    """c_H = (2H / ((1-2H) B(1-2H, H+1/2)))^{1/2}, Molchan-Golosov normaliser."""
    return float(np.sqrt(2 * H / ((1 - 2 * H) * beta_fn(1 - 2 * H, H + 0.5))))


def molchan_golosov_kernel(s: float, t: float, H: float) -> float:
    """K_H(s,t) for 0 < s < t <= 1, H in (0, 1/2) (Section 3):
    K_H(s,t) = c_H [ (t/s)^{H-1/2}(t-s)^{H-1/2}
                     - (H-1/2) s^{1/2-H} int_s^t u^{H-3/2}(u-s)^{H-1/2} du ]."""
    if H == 0.5:
        return 1.0
    c_H = c_H_constant(H)
    term1 = (t / s) ** (H - 0.5) * (t - s) ** (H - 0.5)

    def integrand(u):
        return u ** (H - 1.5) * (u - s) ** (H - 0.5)

    integral, _ = quad(integrand, s, t, limit=200)
    term2 = (H - 0.5) * s ** (0.5 - H) * integral
    return float(c_H * (term1 - term2))


def kernel_action_discretized(H: float, n: int, kernel: str = "RL") -> np.ndarray:
    """Matrix mapping increments g_j of piecewise-constant h_dot on a
    uniform grid of n cells to midpoint evaluations (K h_dot)(t_i).

    kernel="RL": Riemann-Liouville kernel K(t-s) = (t-s)^{H-1/2} (used in
    Definition 2.4 and exp_1/exp_3's leading-order asymptotics with ell=1).
    kernel="MG": Molchan-Golosov fBm kernel (used for the fBm-driven model
    of Section 3 when the exact kernel, not only its leading order, is
    required).
    """
    j = np.arange(1, n + 1)
    a, b = (j - 1) / n, j / n
    t_mid = (np.arange(1, n + 1) - 0.5) / n
    A = np.zeros((n, n))
    if kernel == "RL":
        power = H + 0.5
        for i in range(n):
            t = t_mid[i]
            upper = np.minimum(b, t)
            lower = np.minimum(a, t)
            valid = upper > lower
            contrib = np.zeros(n)
            contrib[valid] = ((t - lower[valid]) ** power - (t - upper[valid]) ** power) / power
            A[i, :] = n * contrib
    elif kernel == "MG":
        for i in range(n):
            t = t_mid[i]
            for jj in range(n):
                if a[jj] >= t:
                    continue
                s_lo, s_hi = a[jj], min(b[jj], t)
                if s_hi <= s_lo:
                    continue
                # integrate K_H(s,t) ds over the cell, n * that = coefficient on g_j
                val, _ = quad(lambda s: molchan_golosov_kernel(s, t, H), s_lo, s_hi, limit=100)
                A[i, jj] = n * val
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return A


# ---------------------------------------------------------------------------
# Theorem 3.3: general sigma, discretised variational problem
# ---------------------------------------------------------------------------
def Lambda_general(K: float, sigma2: Callable[[np.ndarray], np.ndarray], y0: float,
                    H: float, n: int, kernel: str = "RL", g0: np.ndarray | None = None) -> dict:
    """Lambda(K) = inf{ (1/2)||h_dot||^2_{L^2} : int_0^1 sigma^2(y0+(K_H h_dot)(t)) dt = K }
    (Theorem 3.3), discretised as in (4.6)-(4.7) with the midpoint rule."""
    A = kernel_action_discretized(H, n, kernel=kernel)
    if g0 is None:
        g0 = np.full(n, 1e-3)

    def objective(g):
        return 0.5 * n * np.sum(g ** 2)

    def objective_grad(g):
        return n * g

    def constraint(g):
        paths = A @ g
        return np.mean(sigma2(y0 + paths)) - K

    res = minimize(objective, g0, jac=objective_grad,
                    constraints=[{"type": "eq", "fun": constraint}],
                    method="SLSQP", options={"maxiter": 300, "ftol": 1e-12})
    return {"Lambda": float(res.fun), "success": bool(res.success), "g": res.x, "message": res.message}


# ---------------------------------------------------------------------------
# Corollary 3.4: lognormal case sigma^2(y) = e^{nu y}
# ---------------------------------------------------------------------------
def lognormal_rate(K: float, sigma0_sq: float, nu: float, J_func: Callable[[float], float]) -> float:
    """Lambda(K) = (1/nu^2) J(log(K/sigma0^2)) (Remark 3.5(ii)); J here
    is evaluated at nu*log-moneyness per Corollary 3.4 with the
    rate function J of Definition 2.4 using scale nu."""
    y = np.log(K / sigma0_sq)
    return J_func(y) / nu ** 2


# ---------------------------------------------------------------------------
# Corollary 3.8: sigma(y) = xi sqrt(|y|)
# ---------------------------------------------------------------------------
def special_case_rate(K: float, xi: float, y0: float, H: float) -> float:
    """Lambda(K) = (H+1)(K - xi^2|y0|)^2 / xi^4 for K > xi^2|y0| (Calls)."""
    return (H + 1) * (K - xi ** 2 * abs(y0)) ** 2 / xi ** 4


def C_function_cor_3_8(t: np.ndarray, H: float) -> np.ndarray:
    """C(t) = (1/2)(t^{2H} + (1 - t^{2H+1} - (1-t)^{2H+1})/(2H+1)), used in
    the sign condition for Puts in Corollary 3.8."""
    t = np.asarray(t, dtype=float)
    return 0.5 * (t ** (2 * H) + (1 - t ** (2 * H + 1) - (1 - t) ** (2 * H + 1)) / (2 * H + 1))


# ---------------------------------------------------------------------------
# Lemma 3.10: H = 1/2, general monotone sigma^2
# ---------------------------------------------------------------------------
def lemma_3_10_rate(K: float, sigma2: Callable[[float], float], y0: float, h1_sign: int = 1) -> dict:
    """H=1/2 Euler-Lagrange solution (3.6)-(3.7): find h(1) solving
    (1/sqrt(2|lambda|)) int_0^{h1} dx/sqrt(Delta(x)) = 1
    jointly with the ratio-of-integrals condition (3.6) equal to K, then
    Lambda(K) = (1/2)(int_0^{h1} dx/sqrt(Delta(x))) (int_0^{h1} sqrt(Delta(x)) dx),
    with Delta(x) = |sigma(y0+x)^2 - sigma(y0+h1)^2|.
    """

    def Delta(x, h1):
        return abs(sigma2(y0 + x) - sigma2(y0 + h1))

    def ratio_K(h1):
        # (3.6): K = [int_0^h1 sigma^2(y0+x)/sqrt(Delta(x)) dx] / [int_0^h1 dx/sqrt(Delta(x))]
        def num_integrand(x):
            d = Delta(x, h1)
            return sigma2(y0 + x) / np.sqrt(d) if d > 1e-14 else 0.0

        def den_integrand(x):
            d = Delta(x, h1)
            return 1.0 / np.sqrt(d) if d > 1e-14 else 0.0

        num, _ = quad(num_integrand, 0, h1, limit=200, points=[h1 * 0.999])
        den, _ = quad(den_integrand, 0, h1, limit=200, points=[h1 * 0.999])
        return num / den

    def objective(h1):
        return ratio_K(h1) - K

    lo = 1e-6 if h1_sign > 0 else -5.0
    hi = 5.0 if h1_sign > 0 else -1e-6
    # bracket search
    xs = np.linspace(lo, hi, 60) if h1_sign > 0 else np.linspace(lo, hi, 60)
    vals = [objective(x) for x in xs[1:-1]]
    h1 = None
    for i in range(len(vals) - 1):
        if vals[i] == 0:
            h1 = xs[i + 1]
            break
        if vals[i] * vals[i + 1] < 0:
            h1 = brentq(objective, xs[i + 1], xs[i + 2], xtol=1e-10)
            break
    if h1 is None:
        return {"Lambda": np.nan, "h1": np.nan, "success": False}

    def den_integrand(x):
        d = Delta(x, h1)
        return 1.0 / np.sqrt(d) if d > 1e-14 else 0.0

    def sqrt_integrand(x):
        return np.sqrt(Delta(x, h1))

    I_den, _ = quad(den_integrand, 0, h1, limit=200, points=[h1 * 0.999])
    I_sqrt, _ = quad(sqrt_integrand, 0, h1, limit=200, points=[h1 * 0.999])
    Lambda = 0.5 * I_den * I_sqrt
    return {"Lambda": float(Lambda), "h1": float(h1), "success": True}

"""Rate function J(u) of Definition 2.4 and its closed form at H = 1/2.

Custom implementation: Context7 was checked for constrained scalar
variational-problem solvers (pyportfolioopt, skfolio cover portfolio
optimisation, not calculus-of-variations problems over L^2 paths), so the
discretised variational problem and the Liouville-ODE closed form are
implemented from scratch following the paper's Definition 2.4 and
Proposition 2.12.

Definition 2.4 (Sec. 2.2): for u in R,
    J(u) = inf{ (1/2)||h_dot||^2_{L^2([0,1])} : G_1(K h_dot) = u },
with G_1(f) = log(int_0^1 e^{f(s)} ds) (eq. 2.6) and (K h_dot)(t) the
Riemann-Liouville action (K h_dot)(t) = int_0^t (t-s)^{H-1/2} h_dot(s) ds
for ell == 1 (Assumption 2.1).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar


def kernel_action_matrix(H: float, n: int) -> np.ndarray:
    """Matrix A (n x n) mapping increments g_j = h(j/n)-h((j-1)/n) to the
    midpoint evaluations of (K h_dot)(t_i), t_i = (i-1/2)/n, i=1..n.

    For piecewise-constant h_dot = g_j * n on ((j-1)/n, j/n], the
    Riemann-Liouville action is
        (K h_dot)(t) = n * sum_{j<=i} g_j * int_{(j-1)/n}^{min(j/n,t)} (t-s)^{H-1/2} ds.
    Evaluated at the midpoint t_i of cell i, using the exact antiderivative
    of (t-s)^{H-1/2}, this gives a closed-form lower-triangular matrix.
    """
    j = np.arange(1, n + 1)
    a = (j - 1) / n
    b = j / n
    power = H + 0.5
    t_mid = (np.arange(1, n + 1) - 0.5) / n  # midpoints t_i
    # antiderivative of (t-s)^{H-1/2} w.r.t. s is -(t-s)^{H+1/2}/(H+1/2)
    A = np.zeros((n, n))
    for i in range(n):
        t = t_mid[i]
        upper = np.minimum(b, t)
        lower = np.minimum(a, t)
        valid = upper > lower
        contrib = np.zeros(n)
        contrib[valid] = ((t - lower[valid]) ** power - (t - upper[valid]) ** power) / power
        A[i, :] = n * contrib
    return A


def _G1(f_values: np.ndarray, n: int) -> float:
    """G_1(f) = log( (1/n) sum_i e^{f(t_i)} ), midpoint quadrature of
    log(int_0^1 e^{f(s)} ds) (eq. 2.6)."""
    m = np.max(f_values)
    return m + np.log(np.mean(np.exp(f_values - m)))


def solve_J_general(u: float, H: float, n: int, g0: np.ndarray | None = None) -> dict:
    """Numerically solve Definition 2.4 for general H in (0, 1/2].

    Discretises h_dot by increments g in R^n (piecewise constant on a
    uniform grid of n cells), objective (n/2) sum g_j^2 approximating
    (1/2) int (h_dot)^2, subject to G_1(K h_dot) = u. Solved with SLSQP
    (scipy.optimize.minimize), as suggested by implementation_notes.
    """
    A = kernel_action_matrix(H, n)
    if g0 is None:
        # A reasonable scale-free initial guess, refined by SLSQP.
        g0 = np.full(n, np.sign(u) * 0.1 + 1e-6)

    def objective(g: np.ndarray) -> float:
        return 0.5 * n * np.sum(g ** 2)

    def objective_grad(g: np.ndarray) -> np.ndarray:
        return n * g

    def constraint(g: np.ndarray) -> float:
        f_values = A @ g
        return _G1(f_values, n) - u

    res = minimize(
        objective,
        g0,
        jac=objective_grad,
        constraints=[{"type": "eq", "fun": constraint}],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return {"J": float(res.fun), "success": bool(res.success), "g": res.x, "message": res.message}


def J_closed_form_half(u: float) -> float:
    """Closed form of J at H = 1/2 (Proposition 2.12), via the Liouville
    ODE f'' = -a e^f, f(0) = 0, f'(1) = 0.

    For u >= 0: u+ solves u = log(sinh(2 u+) / (2 u+)), J(u) = 2 u+ (u+ - tanh(u+)).
    For u < 0: u- in [0, pi/2) solves u = log(sin(2 u-) / (2 u-)), J(u) = 2 u- (tan(u-) - u-).
    """
    if u == 0.0:
        return 0.0
    if u > 0:
        def g(x):
            return np.log(np.sinh(2 * x) / (2 * x)) - u
        # sinh(2x)/(2x) is increasing from 1 (x->0) to infinity; bracket adaptively.
        lo, hi = 1e-8, 1.0
        while g(hi) < 0:
            hi *= 2.0
            if hi > 1e6:
                break
        u_plus = brentq(g, lo, hi, xtol=1e-13, rtol=1e-13)
        return 2 * u_plus * (u_plus - np.tanh(u_plus))
    else:
        def g(x):
            return np.log(np.sin(2 * x) / (2 * x)) - u
        lo, hi = 1e-10, np.pi / 2 - 1e-10
        u_minus = brentq(g, lo, hi, xtol=1e-13, rtol=1e-13)
        return 2 * u_minus * (np.tan(u_minus) - u_minus)


def J_closed_form_half_vec(u: np.ndarray) -> np.ndarray:
    u = np.atleast_1d(np.asarray(u, dtype=float))
    return np.array([J_closed_form_half(float(x)) for x in u])


def fenchel_legendre_J_star(x: float, J_func, bounds: tuple[float, float] = (-50.0, 50.0)) -> dict:
    """J*(x) = sup_{u in R} { x*u - J(u) } (Proposition 2.6), via
    scipy.optimize.minimize_scalar on -(x*u - J(u)).

    ``bounds`` restricts the search to the domain on which ``J_func`` is a
    faithful representation of J: for the H=1/2 closed form (Proposition
    2.12) J is known for every real u and the default wide bounds apply; for
    a numeric J_func built by interpolating the solved grid {u_grid}, pass
    bounds=(u_grid.min(), u_grid.max()), since outside that grid the
    interpolation is flat and does not reflect the super-linear growth of J
    (Lemma 2.7), which would otherwise push the optimizer to the search
    boundary instead of the true (finite) maximizer.
    """

    def neg_objective(u):
        return -(x * u - J_func(u))

    res = minimize_scalar(neg_objective, bounds=bounds, method="bounded",
                           options={"xatol": 1e-10})
    return {"J_star": float(-res.fun), "u_star": float(res.x)}

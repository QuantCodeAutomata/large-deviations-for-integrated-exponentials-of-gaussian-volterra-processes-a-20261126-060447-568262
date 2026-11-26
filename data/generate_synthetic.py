"""Monte Carlo path generation for the Gaussian Volterra processes studied
in the paper (Section 2.1, Assumption 2.1). No market data is used anywhere
in this repository: every experiment is a numerical/variational study of a
simulated Gaussian process, so this module is the sole place where sample
paths ("synthetic data") are produced for the Monte Carlo verifications of
exp_1, exp_2, exp_3, exp_4 and exp_7.

Two kernels are supported:
  - Riemann-Liouville, K(t) = t^{H-1/2} (ell == 1), X_t = int_0^t (t-s)^{H-1/2} dW_s.
  - Molchan-Golosov (standard fractional Brownian motion), used for the
    rough-volatility model of Section 3, with Cholesky simulation of the
    fBm covariance R_H(t,s) = (1/2)(t^{2H}+s^{2H}-|t-s|^{2H}).
"""

from __future__ import annotations

import numpy as np


def rl_covariance(t_grid: np.ndarray, H: float) -> np.ndarray:
    """Covariance C(t,s) = E[X_t X_s] = int_0^{min(t,s)} (t-u)^{H-1/2}(s-u)^{H-1/2} du
    of the Riemann-Liouville Gaussian Volterra process X_t = int_0^t (t-u)^{H-1/2} dW_u.

    Closed form (hypergeometric representation) is avoided here; instead we
    use the exact Gauss-hypergeometric expression:
        C(t,s) = s^{2H}/(2H) * 2F1(1/2-H, 1, H+3/2, s/t)   for s <= t,
    matching the covariance used in Lemma 4.4 / I(H) (Alos-Garcia Lorite-
    Muguruza representation); this is evaluated directly by numerical
    quadrature for robustness across H to avoid branch-cut issues of hyp2f1
    at s/t close to 1.
    """
    n = len(t_grid)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            t, s = t_grid[j], t_grid[i]  # t >= s since j>=i and grid increasing
            C[i, j] = C[j, i] = _rl_cov_scalar(min(t, s), max(t, s), H)
    return C


def _rl_cov_scalar(s: float, t: float, H: float) -> float:
    """int_0^s (t-u)^{H-1/2}(s-u)^{H-1/2} du, s <= t, via substitution
    u = s*v, v in [0,1]: s^{2H} int_0^1 (t/s - v)^{H-1/2} (1-v)^{H-1/2} dv."""
    if s <= 0:
        return 0.0
    from scipy.integrate import quad

    ratio = t / s

    def integrand(v):
        return (ratio - v) ** (H - 0.5) * (1 - v) ** (H - 0.5)

    value, _ = quad(integrand, 0.0, 1.0, limit=200)
    return s ** (2 * H) * value


def simulate_rl_paths(H: float, t_grid: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate n_paths sample paths of the Riemann-Liouville Gaussian
    Volterra process X_t on t_grid (t_grid[0] > 0 recommended), by Cholesky
    decomposition of the covariance matrix."""
    C = rl_covariance(t_grid, H)
    # Regularise for numerical PSD issues near t -> 0.
    C = C + 1e-12 * np.eye(len(t_grid)) * np.max(np.diag(C))
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((n_paths, len(t_grid)))
    return z @ L.T


def fbm_covariance(t_grid: np.ndarray, H: float) -> np.ndarray:
    """R_H(t,s) = (1/2)(t^{2H} + s^{2H} - |t-s|^{2H}), the Molchan-Golosov
    fractional Brownian motion covariance."""
    T, S = np.meshgrid(t_grid, t_grid, indexing="ij")
    return 0.5 * (T ** (2 * H) + S ** (2 * H) - np.abs(T - S) ** (2 * H))


def simulate_fbm_paths(H: float, t_grid: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate n_paths sample paths of standard fractional Brownian motion
    W_t^H on t_grid via Cholesky decomposition."""
    R = fbm_covariance(t_grid, H)
    R = R + 1e-12 * np.eye(len(t_grid)) * np.max(np.diag(R))
    L = np.linalg.cholesky(R)
    z = rng.standard_normal((n_paths, len(t_grid)))
    return z @ L.T


def simulate_standard_bm_paths(t_grid: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate standard Brownian motion W_t on t_grid (H=1/2 special
    case), via independent Gaussian increments (exact, no Cholesky needed)."""
    dt = np.diff(np.concatenate(([0.0], t_grid)))
    dt = np.clip(dt, 0.0, None)
    increments = rng.standard_normal((n_paths, len(t_grid))) * np.sqrt(dt)
    return np.cumsum(increments, axis=1)

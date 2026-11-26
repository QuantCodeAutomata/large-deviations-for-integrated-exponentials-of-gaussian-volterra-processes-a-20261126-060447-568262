"""At-the-money variance option short-maturity asymptotics: Theorem 3.13,
Corollary 3.15, Remark 3.14 (degenerate case), Remark 3.16.

Custom implementation of the paper's closed-form ATM expansion
coefficients; no library covers this.
"""

from __future__ import annotations

import numpy as np

from .moments_bounds import E_A2


def atm_coefficient_thm_3_13(sigma2_prime_y0: float, H: float) -> float:
    """Theorem 3.13: lim_{T->0} C(sigma0^2,T)/T^H = |(sigma^2)'(y0)| / (2 sqrt(pi(H+1)))."""
    return abs(sigma2_prime_y0) / (2 * np.sqrt(np.pi * (H + 1)))


def atm_coefficient_cor_3_15(H: float, phi0: float = 0.0) -> float:
    """Corollary 3.15: lim_{eps->0} (1/s_eps) E[(Z_eps/eps - e^{phi(0)} Phi_eps)_+]
    = e^{phi(0)} sqrt(E[A^2]/(2 pi)) = e^{phi(0)} / ((2H+1) sqrt(pi(H+1)))."""
    return np.exp(phi0) * np.sqrt(E_A2(H) / (2 * np.pi))


def atm_coefficient_degenerate_remark_3_14(sigma2_double_prime_y0: float, H: float, sign: str) -> float:
    """Remark 3.14, degenerate case (sigma^2)'(y0) = 0:
    Call: lim C(sigma0^2,T)/T^{2H} = ((sigma^2)''(y0))_+ / (2(2H+1))
    Put:  lim P(sigma0^2,T)/T^{2H} = ((sigma^2)''(y0))_- / (2(2H+1))."""
    x = sigma2_double_prime_y0
    if sign == "call":
        return max(x, 0.0) / (2 * (2 * H + 1))
    elif sign == "put":
        return max(-x, 0.0) / (2 * (2 * H + 1))
    raise ValueError("sign must be 'call' or 'put'")


def integral_W_H_variance(T: float, H: float) -> float:
    """Var[int_0^T W_t^H dt] = T^{2H+2} / (2(H+1)), used in the Gaussian
    expansion underlying Theorem 3.13."""
    return T ** (2 * H + 2) / (2 * (H + 1))


def half_normal_mean(sigma: float) -> float:
    """E[(N(0,sigma^2))_+] = sigma / sqrt(2 pi)."""
    return sigma / np.sqrt(2 * np.pi)

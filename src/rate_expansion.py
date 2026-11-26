"""Expansion of the rate function in log-moneyness (Proposition 3.19),
R(H) (eq. 3.14), Lemma 3.20, Corollary 3.22.

Custom implementation: scipy.special.hyp2f1 supplies the Gauss
hypergeometric function; the surrounding integral and the paper-specific
coefficients j2, j3 are implemented from scratch (Context7 confirms no
quantitative-finance library exposes this rate-function expansion).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1


def R_of_H(H: float) -> float:
    """R(H) = int_0^1 s^{2H+1} [2F1(1, -H-1/2, H+3/2, s)]^2 ds, eq. (3.14)."""

    def integrand(s):
        return s ** (2 * H + 1) * hyp2f1(1, -H - 0.5, H + 1.5, s) ** 2

    value, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
    return float(value)


def R_lower_bound(H: float) -> float:
    """Lemma 3.20 lower bound: R(H) > 1/(4(H+1)^2)."""
    return 1.0 / (4 * (H + 1) ** 2)


def j2_j3(H: float, v0: float, v1: float, v2: float) -> dict:
    """Coefficients j2, j3 from (3.13):
    j2 = (H+1/2)^2(H+1)/(2H) * v0^2/v1^2
    j3 = j2 * (1 - 8(H+1)^2 v0 v2 / v1^2 * R(H))."""
    R = R_of_H(H)
    j2 = (H + 0.5) ** 2 * (H + 1) / (2 * H) * (v0 ** 2 / v1 ** 2)
    j3 = j2 * (1 - 8 * (H + 1) ** 2 * v0 * v2 / v1 ** 2 * R)
    return {"j2": float(j2), "j3": float(j3), "R": float(R)}


def rough_bergomi_j2_j3(H: float, eta: float) -> dict:
    """Corollary 3.22 for the rough Bergomi model sigma^2(y)=v0 e^{eta y}:
    v1 = v0 eta, v2 = v0 eta^2/2, so 2 v0 v2 / v1^2 = 1, giving
    j2 = (H+1/2)^2(H+1)/(2 eta^2 H), j3 = j2 (1 - 4(H+1)^2 R(H))."""
    R = R_of_H(H)
    j2 = (H + 0.5) ** 2 * (H + 1) / (2 * eta ** 2 * H)
    j3 = j2 * (1 - 4 * (H + 1) ** 2 * R)
    return {"j2": float(j2), "j3": float(j3), "R": float(R)}


def rate_function_expansion(y: np.ndarray, j2: float, j3: float) -> np.ndarray:
    """J_H(K) ~ j2 y^2 + j3 y^3 + O(y^4), y = log(K / v0) (Prop. 3.19)."""
    y = np.asarray(y, dtype=float)
    return j2 * y ** 2 + j3 * y ** 3


def integral_identity_remark_3_21(H: float) -> float:
    """Remark 3.21 identity: int_0^1 s^{H+1/2} 2F1(1,-H-1/2,H+3/2,s) ds
    = 1/(2(H+1)), used in the Cauchy-Schwarz proof of Lemma 3.20."""

    def integrand(s):
        return s ** (H + 0.5) * hyp2f1(1, -H - 0.5, H + 1.5, s)

    value, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
    return float(value)

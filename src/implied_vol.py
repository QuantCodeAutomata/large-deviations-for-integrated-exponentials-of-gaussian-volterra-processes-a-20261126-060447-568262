"""ATM implied volatility level and skew asymptotics (Proposition 4.1),
Corollary 4.3 (positivity of the ATM skew), Lemma 4.4 (I(H) identity).

Custom implementation of the paper's closed-form asymptotics; the
Black-Scholes price/implied-vol inversion uses scipy.stats.norm and
scipy.optimize.brentq (no finance library implements this short-maturity
variance-option expansion).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import hyp2f1
from scipy.stats import norm

from .rate_expansion import R_of_H


def atm_vol_level(H: float, v0: float, v1: float) -> float:
    """Proposition 4.1: lim_{T->0} sigma_ATM(T)/T^{H-1/2}
    = 1/(H+1/2) sqrt(H/(H+1)) |v1|/v0."""
    return (1.0 / (H + 0.5)) * np.sqrt(H / (H + 1)) * abs(v1) / v0


def atm_skew(j2: float, j3: float) -> float:
    """Proposition 4.1: lim_{T->0} s(T)/T^{H-1/2} = -j3/(2 j2)^{3/2}."""
    return -j3 / (2 * j2) ** 1.5


def mixed_bergomi_v1_v2(v0: float, weights, etas) -> dict:
    """Mixed rough Bergomi sigma^2(y) = v0 sum_i w_i e^{eta_i y}:
    v1 = v0 sum w_i eta_i, v2 = (v0/2) sum w_i eta_i^2 (Corollary 4.3)."""
    weights = np.asarray(weights, dtype=float)
    etas = np.asarray(etas, dtype=float)
    v1 = v0 * np.sum(weights * etas)
    v2 = 0.5 * v0 * np.sum(weights * etas ** 2)
    return {"v1": float(v1), "v2": float(v2)}


def ratio_2v0v2_over_v1sq(v0: float, v1: float, v2: float) -> float:
    """2 v0 v2 / v1^2, the Jensen ratio appearing in Corollary 4.3 (>= 1 by
    Cauchy-Schwarz/Jensen for a mixture, equal to 1 for a single exponential)."""
    return 2 * v0 * v2 / v1 ** 2


def I_of_H_lemma_4_4(H: float) -> float:
    """I(H) from [Alos-Garcia Lorite-Muguruza], used in Lemma 4.4:
    I(H) = 1/((H+1/2)(4H+3)) int_0^1 t^{2H+1} 2F1(1/2-H, 1, H+3/2, t) dt."""

    def integrand(t):
        return t ** (2 * H + 1) * hyp2f1(0.5 - H, 1, H + 1.5, t)

    value, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
    return float(value / ((H + 0.5) * (4 * H + 3)))


def lemma_4_4_check(H: float) -> dict:
    """Verify I(H) = R(H) / (2 (H+1/2)^2) (Lemma 4.4)."""
    I = I_of_H_lemma_4_4(H)
    R = R_of_H(H)
    rhs = R / (2 * (H + 0.5) ** 2)
    return {"I_H": I, "R_over_2Hp5sq": rhs, "abs_diff": abs(I - rhs)}


def black_scholes_call(F: float, K: float, T: float, sigma: float) -> float:
    """Undiscounted Black-Scholes Call price with forward F, used to define
    the implied volatility sigma_BS(K,T) of options on realised variance."""
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def implied_vol_from_price(price: float, F: float, K: float, T: float) -> float:
    """Invert the Black-Scholes Call price for implied volatility with
    scipy.optimize.brentq."""
    intrinsic = max(F - K, 0.0)
    if price <= intrinsic + 1e-14:
        return 0.0

    def objective(sigma):
        return black_scholes_call(F, K, T, sigma) - price

    hi = 5.0
    while objective(hi) < 0 and hi < 1e4:
        hi *= 2.0
    return brentq(objective, 1e-8, hi, xtol=1e-12)

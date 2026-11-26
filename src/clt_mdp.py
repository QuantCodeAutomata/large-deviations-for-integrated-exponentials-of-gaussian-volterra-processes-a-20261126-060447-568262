"""Central limit theorem (Thm 2.14) and moderate deviation principle
(Thm 2.15) constants, plus Remark 2.16 special cases.

Custom implementation of the paper's closed-form CLT variance and MDP rate
function; statsmodels/scipy.stats are used only for the empirical
goodness-of-fit tests in exp/exp2.py, not for these analytical formulas.
"""

from __future__ import annotations

import numpy as np

from .moments_bounds import E_A2


def clt_variance(H: float, phi0: float = 0.0) -> float:
    """Variance of the Gaussian limit in Theorem 2.14:
    (1/s_eps)(Z_eps/eps - e^{phi(0)} Phi_eps) -> N(0, e^{2 phi(0)} E[A^2])."""
    return float(np.exp(2 * phi0) * E_A2(H))


def mdp_rate_function(u: float, H: float, phi0: float = 0.0) -> float:
    """Theorem 2.15 MDP rate function:
    J(u) = u^2 e^{-2 phi(0)} / (2 E[A^2]) = (H+1/2)^2 (H+1) e^{-2 phi(0)} u^2."""
    return (H + 0.5) ** 2 * (H + 1) * np.exp(-2 * phi0) * u ** 2


def mdp_rate_function_coefficient(H: float, phi0: float = 0.0) -> float:
    """Coefficient (H+1/2)^2 (H+1) e^{-2 phi(0)} multiplying u^2 in the MDP
    rate function, equal to 1/(2E[A^2]) e^{-2phi0} since E[A^2]=2/((2H+1)^2(H+1))."""
    return (H + 0.5) ** 2 * (H + 1) * np.exp(-2 * phi0)


a_eps_default_exponent = 0.5  # a_eps = eps^{H * a_eps_default_exponent}; see src.config.MDP_A_EPS_EXPONENT

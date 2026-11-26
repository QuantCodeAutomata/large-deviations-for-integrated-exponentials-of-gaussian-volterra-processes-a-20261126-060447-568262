"""Parameters for every experiment, taken from the paper where stated.

Paper: Jacquier, Oumgari, Pirjol, Zhu, "Large deviations for integrated
exponentials of Gaussian Volterra processes - application to variance
options". Section/equation references below are to that paper.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared Hurst grid (Section 2.1, Assumption 2.1: H in (0, 1/2])
# ---------------------------------------------------------------------------
H_VALUES = (0.1, 0.3, 0.5)  # paper's representative Hurst indices (implementation_notes)
H_GRID_FINE = tuple(np.round(np.arange(0.05, 0.51, 0.05), 2))  # for R(H), chi(x) plots (Sec 3.5, App. A)

RNG_SEED = 12345  # not specified in the paper; fixed for reproducibility

# ---------------------------------------------------------------------------
# exp_1: rate function J(u), Monte Carlo verification of the LDP (Thm 2.5)
# ---------------------------------------------------------------------------
NU_BASELINE = 1.0  # nu > 0 scale in Z_eps = int exp{nu X_s + phi(s)} ds; baseline nu=1 (implementation_notes)
PHI_BASELINE = 0.0  # phi == 0 baseline case (implementation_notes)

# Discretisation of the variational problem Def. 2.4 for general H.
N_DISCRETIZATION_PAPER = 200  # grid points n for h on {j/n}, full configuration
N_DISCRETIZATION_RUN = 60  # reduced grid used by default to keep SLSQP solves fast
U_GRID = tuple(np.round(np.arange(-3.0, 3.01, 0.25), 3))  # log-moneyness/u grid for J(u)

# Monte Carlo verification of the LDP (Corollary 2.11) and Prop 2.6/2.10.
MC_PATHS_PAPER = 100_000  # N >= 1e5 paths, per implementation_notes
MC_PATHS_RUN = 20_000  # reduced path count fitting the time budget
MC_GRID_PAPER = 400  # discretisation points per path on [0,1], full configuration
MC_GRID_RUN = 200  # reduced per-path grid
EPSILON_GRID = (1e-1, 1e-2, 1e-3, 1e-4)  # epsilon -> 0 sequence (implementation_notes)
MC_Z_LEVELS = (0.3, 0.5, 0.7, 1.5, 2.0)  # z values in (0,1) and >1 for Corollary 2.11 / Prop 2.6 tails

# ---------------------------------------------------------------------------
# exp_2: CLT (Thm 2.14) and MDP (Thm 2.15)
# ---------------------------------------------------------------------------
MDP_A_EPS_EXPONENT = 0.5  # a_eps = eps^{H/2}: a_eps -> 0 and s_eps/a_eps = eps^{H/2} -> 0 (implementation_notes)

# ---------------------------------------------------------------------------
# exp_3 / exp_4: variance option short-maturity asymptotics
# ---------------------------------------------------------------------------
Y0_BASELINE = 0.0  # spot log-vol level y0 (Section 3.1), sigma0^2 = sigma^2(y0)
NU_LOGNORMAL = (1.0, 2.0)  # nu in lognormal sigma^2(y)=e^{nu y} (implementation_notes)
XI_SPECIAL_CASE = 1.0  # xi in sigma(y) = xi*sqrt(|y|) (Corollary 3.8), not specified -> unit scale
STRIKE_GRID_OTM = tuple(np.round(np.exp(np.arange(-1.0, 1.01, 0.2)), 4))  # K/sigma0^2 grid, log-moneyness in [-1,1]

T_GRID_PAPER = (1e-1, 1e-2, 1e-3, 1e-4)  # maturities T -> 0, per implementation_notes
T_GRID_RUN = (1e-1, 1e-2, 1e-3, 1e-4)

# ---------------------------------------------------------------------------
# exp_5: rate function expansion in log-moneyness (Prop 3.19)
# ---------------------------------------------------------------------------
V0_BASELINE = 1.0  # v0 in sigma^2(y) = v0 + v1 y + v2 y^2 + O(y^3)
ETA_ROUGH_BERGOMI = 2.0  # eta in rough Bergomi sigma^2(y) = v0 e^{eta y}, matches Section 4.3

# ---------------------------------------------------------------------------
# exp_6: implied volatility expansion (Prop 4.1)
# ---------------------------------------------------------------------------
MIXED_BERGOMI_WEIGHTS = (0.5, 0.5)  # w_i, not specified -> symmetric two-component mix (implementation_notes)
MIXED_BERGOMI_ETAS = (1.0, 3.0)  # eta_i, not specified -> example in implementation_notes

# ---------------------------------------------------------------------------
# exp_7: numerical validation in the rough Bergomi model (Fig. 4.1)
# ---------------------------------------------------------------------------
DISCRETIZATION_POINTS_PAPER = (5, 10, 20, 40)  # n values reproducing Figure 4.1
DISCRETIZATION_POINTS_RUN = (5, 10, 20)  # reduced set: n=40 solves are slow (one extra point kept optional via --full)
LOG_MONEYNESS_GRID = tuple(np.round(np.arange(-1.0, 1.01, 0.2), 2))  # y in [-1, 1] (Figure 4.1)

# ---------------------------------------------------------------------------
# exp_8: Molchan-Golosov vs Riemann-Liouville kernels (Appendix A, Lemma 4.4)
# ---------------------------------------------------------------------------
X_GRID_CHI = tuple(np.round(np.geomspace(1.001, 100.0, 60), 4))  # x = t/s >= 1 grid for chi(x)

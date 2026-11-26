"""Experiment 4: at-the-money variance option short-maturity asymptotics
(Section 3.4): Theorem 3.13, Corollary 3.15, and the degenerate case of
Remark 3.14, with a Monte Carlo check of the T^H (resp. T^{2H}) scaling.

Run: python -m exp.exp4 [--full]
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from data.generate_synthetic import simulate_fbm_paths
from src import config
from src.atm_asymptotics import (
    atm_coefficient_cor_3_15,
    atm_coefficient_degenerate_remark_3_14,
    atm_coefficient_thm_3_13,
    integral_W_H_variance,
)
from src.moments_bounds import E_A2

RESULTS_DIR = "results"


def lognormal_sigma2(y: np.ndarray | float, nu: float) -> np.ndarray | float:
    """sigma^2(y) = e^{nu y} (Section 3.4 baseline example)."""
    return np.exp(nu * y)


def mc_atm_call_scaling(H: float, nu: float, T_grid, n_paths: int, rng) -> list[dict]:
    """Monte Carlo check of Theorem 3.13 for sigma^2(y)=e^{nu y}, y0=0.

    Sigma_T = (1/T) int_0^T sigma^2(Y_t) dt with Y_t = y0 + W_t^H (standard
    fBm, Molchan-Golosov representation). C(sigma0^2,T) = E[(Sigma_T -
    sigma0^2)_+]; the paper's limit is C(sigma0^2,T)/T^H ->
    |(sigma^2)'(y0)|/(2 sqrt(pi(H+1))).
    """
    rows = []
    for T in T_grid:
        n_grid = max(64, int(200 * T ** 0 ))  # fine subgrid on [0,T], uniform
        t_grid = np.linspace(0.0, T, 200)
        paths = simulate_fbm_paths(H, t_grid, n_paths, rng)  # (n_paths, len(t_grid))
        sigma2_vals = lognormal_sigma2(paths, nu)
        sigma_T = np.trapz(sigma2_vals, t_grid, axis=1) / T
        sigma0_sq = 1.0  # e^{nu*0}
        call_payoff = np.maximum(sigma_T - sigma0_sq, 0.0)
        C = float(np.mean(call_payoff))
        rows.append({"T": T, "C_over_TH": C / (T ** H), "C": C})
    return rows


def run(full: bool) -> None:
    rng = np.random.default_rng(config.RNG_SEED)
    T_grid = config.T_GRID_PAPER if full else config.T_GRID_RUN
    n_paths = config.MC_PATHS_PAPER if full else config.MC_PATHS_RUN

    thm_rows = []
    cor315_rows = []
    degenerate_rows = []
    mc_rows = []

    for H in config.H_VALUES:
        for nu in config.NU_LOGNORMAL:
            # Theorem 3.13: sigma^2(y)=e^{nu y}, y0=0 => (sigma^2)'(0) = nu.
            coeff_thm = atm_coefficient_thm_3_13(sigma2_prime_y0=nu, H=H)
            thm_rows.append({"H": H, "nu": nu, "atm_coefficient_thm_3_13": coeff_thm})

            # Corollary 3.15: Volterra process version with phi==0.
            coeff_cor = atm_coefficient_cor_3_15(H, phi0=config.PHI_BASELINE)
            e_a2 = E_A2(H)
            cor315_rows.append({
                "H": H,
                "coeff_cor_3_15": coeff_cor,
                "analytic_1_over_(2H+1)sqrt(pi(H+1))": 1.0 / ((2 * H + 1) * np.sqrt(np.pi * (H + 1))),
                "E_A2": e_a2,
            })

            # Degenerate case, Remark 3.14: (sigma^2)'(y0)=0, sigma^2(y)=1+y^2
            # so (sigma^2)''(0)=2; Call coefficient uses the positive part.
            coeff_deg_call = atm_coefficient_degenerate_remark_3_14(2.0, H, "call")
            coeff_deg_put = atm_coefficient_degenerate_remark_3_14(2.0, H, "put")
            degenerate_rows.append({
                "H": H,
                "sigma2_double_prime_y0": 2.0,
                "coeff_degenerate_call": coeff_deg_call,
                "coeff_degenerate_put": coeff_deg_put,
            })

        # Monte Carlo verification for nu=1 only (cheapest, still informative).
        mc = mc_atm_call_scaling(H, nu=1.0, T_grid=T_grid, n_paths=n_paths, rng=rng)
        for row in mc:
            row["H"] = H
            row["analytic_coeff"] = atm_coefficient_thm_3_13(1.0, H)
            mc_rows.append(row)

    with open(f"{RESULTS_DIR}/exp4_thm_3_13.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(thm_rows[0].keys()))
        writer.writeheader()
        writer.writerows(thm_rows)

    with open(f"{RESULTS_DIR}/exp4_cor_3_15.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cor315_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cor315_rows)

    with open(f"{RESULTS_DIR}/exp4_degenerate_remark_3_14.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(degenerate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(degenerate_rows)

    with open(f"{RESULTS_DIR}/exp4_mc_scaling.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(mc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mc_rows)

    # Summary: at the smallest T, how close is the MC-estimated coefficient
    # C(sigma0^2,T)/T^H to the analytic Theorem 3.13 coefficient.
    summary = {}
    for H in config.H_VALUES:
        rows_H = [r for r in mc_rows if r["H"] == H]
        smallest_T_row = min(rows_H, key=lambda r: r["T"])
        rel_err = abs(smallest_T_row["C_over_TH"] - smallest_T_row["analytic_coeff"]) / smallest_T_row["analytic_coeff"]
        summary[str(H)] = {
            "T_min": smallest_T_row["T"],
            "mc_coefficient": smallest_T_row["C_over_TH"],
            "analytic_coefficient": smallest_T_row["analytic_coeff"],
            "relative_error": rel_err,
        }
    with open(f"{RESULTS_DIR}/exp4_summary.json", "w") as fh:
        json.dump({"full_run": full, "n_paths": n_paths, "T_grid": list(T_grid), "by_H": summary}, fh, indent=2)

    print("exp4 done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(args.full)

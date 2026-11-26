"""Experiment 8: comparison of the Molchan-Golosov and Riemann-Liouville
kernels for fractional Brownian motion (Appendix A): the linking function
chi(x) (Proposition A.1), the ODE (A.2), the bounds of Theorem A.3(iv),
and the I(H) = R(H)/(2(H+1/2)^2) identity of Lemma 4.4.

Run: python -m exp.exp8 [--full]
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.implied_vol import lemma_4_4_check
from src.kernel_comparison import (
    chi_asymptotics_infinity,
    chi_asymptotics_near_1,
    chi_vec,
    kappa_bounds,
    ode_residual_A2,
    variance_ratio_MG_vs_RL,
)

RESULTS_DIR = "results"


def run(full: bool) -> None:
    H_grid = [H for H in config.H_GRID_FINE if 0 < H < 0.5]
    x_grid = np.array(config.X_GRID_CHI)

    rows = []
    for H in H_grid:
        chi_vals = chi_vec(x_grid, H)
        bounds = kappa_bounds(H, x_grid)
        ode_res = [ode_residual_A2(float(x), H) for x in x_grid if x > 1.0 + 1e-6]
        near1 = chi_asymptotics_near_1(1.0 + 1e-3, H)
        inf_asym = chi_asymptotics_infinity(float(x_grid[-1]), H)
        chi_near1_numeric = float(chi_vec(np.array([1.0 + 1e-3]), H)[0])
        lemma44 = lemma_4_4_check(H)
        var_ratio = variance_ratio_MG_vs_RL(H)
        rows.append({
            "H": H,
            "kappa_H": bounds["kappa_H"],
            "kappa_bar_H": bounds["kappa_bar_H"],
            "max_abs_ode_residual_A2": float(np.max(np.abs(ode_res))),
            "chi_near1_numeric": chi_near1_numeric,
            "chi_near1_asymptotic": near1,
            "chi_infinity_asymptotic": inf_asym,
            "chi_infinity_numeric": float(chi_vals[-1]),
            "variance_ratio_MG_vs_RL": var_ratio,
            "I_of_H": lemma44["I_H"],
            "R_over_2Hp5sq": lemma44["R_over_2Hp5sq"],
            "lemma_4_4_abs_diff": lemma44["abs_diff"],
        })

    # H=0.5: variance ratio should equal 1 (MG and RL kernels coincide).
    var_ratio_half = variance_ratio_MG_vs_RL(0.5)

    with open(f"{RESULTS_DIR}/exp8_kernel_comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    max_ode_residual = max(r["max_abs_ode_residual_A2"] for r in rows)
    max_lemma44_diff = max(r["lemma_4_4_abs_diff"] for r in rows)

    summary = {
        "full_run": full,
        "H_grid": H_grid,
        "x_grid_max": float(x_grid[-1]),
        "max_abs_ode_A2_residual_over_H": max_ode_residual,
        "max_abs_lemma_4_4_diff_over_H": max_lemma44_diff,
        "variance_ratio_MG_vs_RL_at_H_half": var_ratio_half,
        "variance_ratio_equals_1_at_H_half": bool(abs(var_ratio_half - 1.0) < 1e-10),
    }
    with open(f"{RESULTS_DIR}/exp8_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for H in H_grid:
        chi_vals = chi_vec(x_grid, H)
        axes[0].plot(x_grid, chi_vals * x_grid ** (H - 0.5), label=f"H={H}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("chi(x) x^(H-1/2)")
    axes[0].set_title("Theorem A.3(iv) bounds: chi(x) x^(H-1/2)")
    axes[0].legend(fontsize=6)

    H_arr = np.array([r["H"] for r in rows])
    axes[1].plot(H_arr, [r["I_of_H"] for r in rows], "o-", label="I(H)")
    axes[1].plot(H_arr, [r["R_over_2Hp5sq"] for r in rows], "x--", label="R(H)/(2(H+1/2)^2)")
    axes[1].set_xlabel("H")
    axes[1].set_title("Lemma 4.4 identity")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp8_kernel_comparison.png", dpi=150)
    plt.close(fig)

    print("exp8 done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(args.full)

"""Experiment 7: numerical validation of the analytical asymptotics in the
rough Bergomi model (Section 4.3): the discretised variational problem
J_{H,n}(y) of (4.6)-(4.7), compared against the analytical linear
approximation of Proposition 4.1, reproducing Figure 4.1.

Run: python -m exp.exp7 [--full]
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
from src.rough_bergomi_discretization import (
    J_H_n,
    analytical_linear_approximation,
    rescaled_implied_vol_from_J,
    rescaled_implied_vol_H_half,
)

RESULTS_DIR = "results"


def run(full: bool) -> None:
    H_values = config.H_VALUES
    n_values = config.DISCRETIZATION_POINTS_PAPER if full else config.DISCRETIZATION_POINTS_RUN
    y_grid = np.array(config.LOG_MONEYNESS_GRID)
    v0 = config.V0_BASELINE
    eta = config.ETA_ROUGH_BERGOMI

    rows = []
    colors = {5: "blue", 10: "red", 20: "orange", 40: "black"}
    fig, axes = plt.subplots(1, len(H_values), figsize=(5 * len(H_values), 4), sharey=False)
    if len(H_values) == 1:
        axes = [axes]

    for ax, H in zip(axes, H_values):
        analytic = analytical_linear_approximation(y_grid, H, v0, eta)
        ax.plot(y_grid, analytic, "k--", label="analytical linear (Prop. 4.1)")

        for n in n_values:
            iv_n = []
            g_prev = None
            for y in y_grid:
                res = J_H_n(float(y), H, n, v0, eta, g0=g_prev)
                g_prev = res["g"]
                if H == 0.5:
                    iv = rescaled_implied_vol_H_half(float(y), eta)
                else:
                    iv = rescaled_implied_vol_from_J(float(y), res["J"])
                iv_n.append(iv)
                rows.append({
                    "H": H, "n": n, "y": float(y), "J_numeric": res["J"],
                    "rescaled_iv": iv, "solver_success": res["success"],
                })
            ax.plot(y_grid, iv_n, "o-", color=colors.get(n, None), label=f"n={n}", markersize=3)

        ax.set_title(f"H={H}")
        ax.set_xlabel("log-moneyness y")
        ax.set_ylabel("rescaled implied volatility")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp7_figure_4_1.png", dpi=150)
    plt.close(fig)

    with open(f"{RESULTS_DIR}/exp7_discretized_rate_function.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Convergence check: max |numeric - analytic| over y, at the largest n,
    # should shrink relative to the smallest n.
    convergence = {}
    for H in H_values:
        analytic = analytical_linear_approximation(y_grid, H, v0, eta)
        errs = {}
        for n in n_values:
            rows_n = [r for r in rows if r["H"] == H and r["n"] == n]
            rows_n.sort(key=lambda r: r["y"])
            numeric = np.array([r["rescaled_iv"] for r in rows_n])
            mask = ~np.isnan(numeric)
            errs[str(n)] = float(np.max(np.abs(numeric[mask] - analytic[mask])))
        convergence[str(H)] = errs

    summary = {
        "full_run": full,
        "H_values": list(H_values),
        "n_values": list(n_values),
        "y_grid": list(y_grid),
        "max_abs_error_numeric_vs_analytic_by_H_and_n": convergence,
    }
    with open(f"{RESULTS_DIR}/exp7_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("exp7 done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(args.full)

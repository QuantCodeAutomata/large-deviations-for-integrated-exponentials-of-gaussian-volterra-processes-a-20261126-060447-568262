"""Experiment 5: rate function expansion in log-moneyness (Section 3.5):
R(H) from (3.14), Lemma 3.20, Corollary 3.22 for the rough Bergomi model,
and the H=1/2 Markovian check of Remark 3.23.

Run: python -m exp.exp5 [--full]
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
from src.rate_expansion import (
    R_lower_bound,
    R_of_H,
    integral_identity_remark_3_21,
    rough_bergomi_j2_j3,
)

RESULTS_DIR = "results"


def run(full: bool) -> None:
    H_grid = config.H_GRID_FINE
    eta = config.ETA_ROUGH_BERGOMI

    rows = []
    for H in H_grid:
        if H <= 0:
            continue
        R = R_of_H(H)
        bound = R_lower_bound(H)
        identity = integral_identity_remark_3_21(H)
        identity_expected = 1.0 / (2 * (H + 1))
        bergomi = rough_bergomi_j2_j3(H, eta)
        rows.append({
            "H": H,
            "R_of_H": R,
            "R_lower_bound_lemma_3_20": bound,
            "lemma_3_20_holds": bool(R > bound),
            "remark_3_21_integral": identity,
            "remark_3_21_expected": identity_expected,
            "remark_3_21_abs_error": abs(identity - identity_expected),
            "j2_rough_bergomi": bergomi["j2"],
            "j3_rough_bergomi": bergomi["j3"],
            "j3_leq_0": bool(bergomi["j3"] <= 1e-10),
        })

    with open(f"{RESULTS_DIR}/exp5_rate_expansion.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # H=1/2 explicit check (Remark 3.23): R(1/2)=2/15, j2=3/(2 eta^2),
    # j3 = -3/(10 eta^2).
    R_half = R_of_H(0.5)
    bergomi_half = rough_bergomi_j2_j3(0.5, eta)
    half_check = {
        "R_half_numeric": R_half,
        "R_half_expected_2_over_15": 2.0 / 15.0,
        "R_half_abs_error": abs(R_half - 2.0 / 15.0),
        "j2_half_numeric": bergomi_half["j2"],
        "j2_half_expected": 3.0 / (2 * eta ** 2),
        "j3_half_numeric": bergomi_half["j3"],
        "j3_half_expected": -3.0 / (10 * eta ** 2),
    }

    all_hold = all(r["lemma_3_20_holds"] for r in rows)
    all_j3_neg = all(r["j3_leq_0"] for r in rows)

    summary = {
        "full_run": full,
        "H_grid": list(H_grid),
        "eta": eta,
        "lemma_3_20_holds_for_all_H": all_hold,
        "j3_nonpositive_for_all_H_rough_bergomi": all_j3_neg,
        "h_half_check": half_check,
    }
    with open(f"{RESULTS_DIR}/exp5_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    H_arr = np.array([r["H"] for r in rows])
    ax.plot(H_arr, [r["R_of_H"] for r in rows], "o-", label="R(H) numeric")
    ax.plot(H_arr, [r["R_lower_bound_lemma_3_20"] for r in rows], "--", label="1/(4(H+1)^2) lower bound")
    ax.set_xlabel("H")
    ax.set_ylabel("R(H)")
    ax.set_title("R(H) vs. the Lemma 3.20 lower bound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp5_R_of_H.png", dpi=150)
    plt.close(fig)

    print("exp5 done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(args.full)

"""Experiment 6: implied volatility expansion for variance options
(Section 4): the ATM level and skew of Proposition 4.1, the sign of the
skew in the mixed rough Bergomi model (Corollary 4.3), and the I(H)
identity of Lemma 4.4 comparing with Alos-Garcia Lorite-Muguruza [3].

Run: python -m exp.exp6 [--full]
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from src import config
from src.implied_vol import (
    atm_skew,
    atm_vol_level,
    lemma_4_4_check,
    mixed_bergomi_v1_v2,
    ratio_2v0v2_over_v1sq,
)
from src.rate_expansion import R_of_H, rough_bergomi_j2_j3

RESULTS_DIR = "results"


def run(full: bool) -> None:
    H_grid = config.H_VALUES
    eta = config.ETA_ROUGH_BERGOMI
    v0 = config.V0_BASELINE
    weights = config.MIXED_BERGOMI_WEIGHTS
    etas = config.MIXED_BERGOMI_ETAS

    rough_bergomi_rows = []
    mixed_rows = []
    lemma44_rows = []

    for H in H_grid:
        # Rough Bergomi: v1 = v0*eta, v2 = v0*eta^2/2.
        v1 = v0 * eta
        v2 = 0.5 * v0 * eta ** 2
        bergomi = rough_bergomi_j2_j3(H, eta)
        sigma_atm = atm_vol_level(H, v0, v1)
        skew = atm_skew(bergomi["j2"], bergomi["j3"])
        rough_bergomi_rows.append({
            "H": H, "eta": eta, "v1": v1, "v2": v2,
            "sigma_atm_coefficient": sigma_atm,
            "atm_skew_coefficient": skew,
            "j2": bergomi["j2"], "j3": bergomi["j3"],
        })

        # Mixed rough Bergomi (Corollary 4.3): positivity of the ATM skew.
        v1_mix, v2_mix = mixed_bergomi_v1_v2(v0, weights, etas).values()
        R = R_of_H(H)
        j2_mix = (H + 0.5) ** 2 * (H + 1) / (2 * H) * (v0 ** 2 / v1_mix ** 2)
        ratio = ratio_2v0v2_over_v1sq(v0, v1_mix, v2_mix)
        j3_mix = j2_mix * (1 - 4 * (H + 1) ** 2 * ratio * R)
        skew_mix = atm_skew(j2_mix, j3_mix)
        mixed_rows.append({
            "H": H, "weights": str(weights), "etas": str(etas),
            "v1_mix": v1_mix, "v2_mix": v2_mix,
            "jensen_ratio_2v0v2_over_v1sq": ratio,
            "ratio_geq_1": bool(ratio >= 1.0 - 1e-10),
            "j2_mix": j2_mix, "j3_mix": j3_mix,
            "atm_skew_mix": skew_mix,
            "skew_positive": bool(skew_mix > 0),
        })

        # Lemma 4.4: I(H) = R(H) / (2 (H+1/2)^2).
        lemma44_rows.append({"H": H, **lemma_4_4_check(H)})

    # H=1/2 explicit check: sigma_ATM = eta/sqrt(3), s = eta/(10 sqrt(3)) (Remark 4.2).
    H_half = 0.5
    v1_half = v0 * eta
    bergomi_half = rough_bergomi_j2_j3(H_half, eta)
    sigma_atm_half = atm_vol_level(H_half, v0, v1_half)
    skew_half = atm_skew(bergomi_half["j2"], bergomi_half["j3"])
    half_check = {
        "sigma_atm_numeric": sigma_atm_half,
        "sigma_atm_expected_eta_over_sqrt3": eta / np.sqrt(3.0),
        "skew_numeric": skew_half,
        "skew_expected_eta_over_10sqrt3": eta / (10 * np.sqrt(3.0)),
    }

    # Finer H grid for the Lemma 4.4 identity check.
    lemma44_fine = [{"H": H, **lemma_4_4_check(H)} for H in config.H_GRID_FINE if H > 0]

    with open(f"{RESULTS_DIR}/exp6_rough_bergomi_atm.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rough_bergomi_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rough_bergomi_rows)

    with open(f"{RESULTS_DIR}/exp6_mixed_bergomi_skew.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(mixed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mixed_rows)

    with open(f"{RESULTS_DIR}/exp6_lemma_4_4.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(lemma44_fine[0].keys()))
        writer.writeheader()
        writer.writerows(lemma44_fine)

    max_lemma44_diff = max(r["abs_diff"] for r in lemma44_fine)
    all_skew_positive = all(r["skew_positive"] for r in mixed_rows)

    summary = {
        "full_run": full,
        "H_values": list(H_grid),
        "eta": eta,
        "mixed_weights": list(weights),
        "mixed_etas": list(etas),
        "lemma_4_4_max_abs_diff": max_lemma44_diff,
        "mixed_bergomi_skew_always_positive": all_skew_positive,
        "h_half_check": half_check,
    }
    with open(f"{RESULTS_DIR}/exp6_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("exp6 done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(args.full)

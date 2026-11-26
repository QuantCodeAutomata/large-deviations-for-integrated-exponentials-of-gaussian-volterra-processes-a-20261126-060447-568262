"""Experiment 3: out-of-the-money variance option short-maturity
asymptotics (Section 3.1): the general variational problem of Theorem 3.3,
the lognormal case (Corollary 3.4), the special case sigma(y)=xi sqrt|y|
(Corollary 3.8), and the H=1/2 general monotone sigma case (Lemma 3.10),
with a Monte Carlo check of T^{2H} log C(K,T) -> -Lambda(K).

Run: python -m exp.exp3 [--full]
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from data.generate_synthetic import simulate_fbm_paths
from src import config
from src.rate_function import J_closed_form_half, solve_J_general
from src.variance_options import (
    C_function_cor_3_8,
    Lambda_general,
    lemma_3_10_rate,
    lognormal_rate,
    special_case_rate,
)

RESULTS_DIR = "results"


def J_func_for(H: float, n: int):
    """A J(.) callable for Corollary 3.4, closed form at H=0.5, otherwise
    the Definition 2.4 numerical solver re-run per query (adequate at the
    modest strike grids used here)."""
    if H == 0.5:
        return J_closed_form_half
    cache: dict[float, float] = {}

    def J_func(u: float) -> float:
        u = float(u)
        if u not in cache:
            cache[u] = solve_J_general(u, H, n)["J"]
        return cache[u]

    return J_func


def lognormal_experiment(H: float, nu: float, n_disc: int) -> list[dict]:
    """Corollary 3.4: Lambda(K) = J(log(K/sigma0^2))/nu^2 (Remark 3.5(ii)),
    cross-checked against the general discretised variational problem of
    Theorem 3.3 for sigma^2(y)=e^{nu y}."""
    sigma0_sq = np.exp(nu * config.Y0_BASELINE)
    J_func = J_func_for(H, n_disc)
    rows = []
    for m in config.STRIKE_GRID_OTM:
        K = sigma0_sq * m
        if abs(np.log(m)) < 1e-10:
            continue  # ATM: Lambda(sigma0^2) = 0, handled by exp_4
        lam_closed = lognormal_rate(K, sigma0_sq, nu, J_func)

        def sigma2(y, _nu=nu):
            return np.exp(_nu * y)

        gen = Lambda_general(K, sigma2, config.Y0_BASELINE, H, n_disc, kernel="RL")
        rows.append({"H": H, "nu": nu, "K_over_sigma0sq": m, "K": K,
                     "Lambda_corollary_3_4": lam_closed,
                     "Lambda_theorem_3_3_numeric": gen["Lambda"],
                     "abs_diff": abs(lam_closed - gen["Lambda"]),
                     "numeric_success": gen["success"]})
    return rows


def special_case_experiment(H: float, xi: float) -> list[dict]:
    """Corollary 3.8: sigma(y) = xi sqrt|y|, closed form
    Lambda(K) = (H+1)(K - xi^2|y0|)^2/xi^4 for K > xi^2|y0|.

    Theorem 3.3's variational problem for this sigma uses the fBm
    Molchan-Golosov kernel K_H (Section 3), not the Riemann-Liouville
    kernel of Definition 2.4: a direct discretised cross-check analogous
    to the lognormal case would require solving the singular
    Molchan-Golosov kernel integral at every grid cell, which is
    numerically unstable at the discretisation sizes used here (scipy.quad
    reports non-convergence for H<0.3). Corollary 3.8 is therefore
    validated against Monte Carlo simulation of the fBm-driven model
    instead (monte_carlo_check_special_case below), which samples the
    exact Molchan-Golosov covariance.
    """
    y0 = 1.0  # y0 != 0 required so sigma0^2 = xi^2|y0| > 0 (Section 3.3.1)
    sigma0_sq = xi ** 2 * abs(y0)
    rows = []
    for m in config.STRIKE_GRID_OTM:
        if abs(np.log(m)) < 1e-10:
            continue
        K = sigma0_sq * m
        if K <= sigma0_sq:
            continue  # Call side only (K > xi^2|y0|)
        lam_closed = special_case_rate(K, xi, y0, H)
        rows.append({"H": H, "xi": xi, "y0": y0, "K_over_sigma0sq": m, "K": K,
                     "Lambda_corollary_3_8": lam_closed})
    C_vals = C_function_cor_3_8(np.linspace(0.01, 0.99, 10), H)
    rows.append({"H": H, "xi": xi, "y0": y0, "K_over_sigma0sq": np.nan, "K": np.nan,
                 "Lambda_corollary_3_8": np.nan, "C_function_max": float(np.max(C_vals))})
    return rows


def monte_carlo_check_special_case(H: float, xi: float, n_paths: int, n_grid: int, rng) -> list[dict]:
    """Monte Carlo check of Corollary 3.8 for sigma(y) = xi sqrt|y|: simulate
    Y_t = y0 + W_t^H (Molchan-Golosov fBm, exact covariance simulation) and
    verify T^{2H} log E[(Sigma_T-K)_+] -> -Lambda(K)."""
    y0 = 1.0
    sigma0_sq = xi ** 2 * abs(y0)
    rows = []
    for T in config.T_GRID_PAPER:
        t_grid = T * np.arange(1, n_grid + 1) / n_grid
        W = simulate_fbm_paths(H, t_grid, n_paths, rng)
        Y = y0 + W
        integrand = xi ** 2 * np.abs(Y)
        Sigma_T = np.trapz(integrand, t_grid, axis=1) / T
        for m in (1.3, 1.6):
            K = sigma0_sq * m
            payoff = np.maximum(Sigma_T - K, 0.0)
            price = float(np.mean(payoff))
            lhs = T ** (2 * H) * np.log(price) if price > 0 else np.nan
            rhs = -special_case_rate(K, xi, y0, H)
            rows.append({"H": H, "xi": xi, "T": T, "K_over_sigma0sq": m,
                         "price": price, "lhs_Tpow2H_logC": lhs, "rhs_minus_Lambda": rhs})
    return rows


def lemma_3_10_experiment(nu: float) -> list[dict]:
    """Lemma 3.10 at H=1/2 for the monotone sigma^2(y)=e^{nu y}: solve the
    Euler-Lagrange integral equations (3.6)-(3.7) and compare with the
    H=1/2 closed form of Corollary 3.4."""
    sigma0_sq = np.exp(nu * config.Y0_BASELINE)
    J_func = J_closed_form_half
    rows = []
    for m in config.STRIKE_GRID_OTM:
        if abs(np.log(m)) < 1e-10 or m <= 1.0:
            continue  # Lemma 3.10 implementation below targets the Call side (h increasing)
        K = sigma0_sq * m
        lam_closed = lognormal_rate(K, sigma0_sq, nu, J_func)

        def sigma2(y, _nu=nu):
            return np.exp(_nu * y)

        res = lemma_3_10_rate(K, sigma2, config.Y0_BASELINE, h1_sign=1)
        rows.append({"nu": nu, "K_over_sigma0sq": m, "K": K,
                     "Lambda_lemma_3_10": res["Lambda"], "h1": res["h1"],
                     "Lambda_corollary_3_4_half": lam_closed,
                     "abs_diff": abs(res["Lambda"] - lam_closed) if res["success"] else np.nan,
                     "success": res["success"]})
    return rows


def monte_carlo_check(H: float, nu: float, n_paths: int, n_grid: int,
                       rng, J_func) -> list[dict]:
    """Simulate the rough volatility model Y_t = y0 + W_t^H (Molchan-Golosov
    fBm), compute Sigma_T = (1/T) int_0^T e^{nu Y_t} dt by the trapezoidal
    rule, and check T^{2H} log E[(Sigma_T-K)_+] -> -Lambda(K)."""
    sigma0_sq = np.exp(nu * config.Y0_BASELINE)
    rows = []
    for T in config.T_GRID_PAPER:
        t_grid = T * np.arange(1, n_grid + 1) / n_grid
        W = simulate_fbm_paths(H, t_grid, n_paths, rng)
        Y = config.Y0_BASELINE + W
        integrand = np.exp(nu * Y)
        Sigma_T = np.trapz(integrand, t_grid, axis=1) / T
        for m in (1.3, 1.6):
            K = sigma0_sq * m
            payoff = np.maximum(Sigma_T - K, 0.0)
            price = float(np.mean(payoff))
            lhs = T ** (2 * H) * np.log(price) if price > 0 else np.nan
            rhs = -lognormal_rate(K, sigma0_sq, nu, J_func)
            rows.append({"H": H, "nu": nu, "T": T, "K_over_sigma0sq": m,
                         "price": price, "lhs_Tpow2H_logC": lhs, "rhs_minus_Lambda": rhs})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="use the paper's full discretisation and Monte Carlo sample size")
    args = parser.parse_args()

    n_disc = config.N_DISCRETIZATION_PAPER if args.full else config.N_DISCRETIZATION_RUN
    n_mc_paths = config.MC_PATHS_PAPER if args.full else config.MC_PATHS_RUN
    n_mc_grid = config.MC_GRID_PAPER if args.full else config.MC_GRID_RUN
    rng = np.random.default_rng(config.RNG_SEED)
    nu = config.NU_LOGNORMAL[0]
    xi = config.XI_SPECIAL_CASE

    print(f"[exp3] n_discretization={n_disc}, mc_paths={n_mc_paths}, mc_grid={n_mc_grid}, "
          f"H_values={config.H_VALUES}, nu={nu}, xi={xi}")

    lognormal_rows, special_rows, lemma_rows, mc_rows = [], [], [], []
    for H in config.H_VALUES:
        print(f"[exp3] lognormal case (Corollary 3.4) for H={H} ...")
        lognormal_rows.extend(lognormal_experiment(H, nu, n_disc))
        print(f"[exp3] special case sigma=xi*sqrt|y| (Corollary 3.8) for H={H} ...")
        special_rows.extend(special_case_experiment(H, xi))
        print(f"[exp3] Monte Carlo check (lognormal) for H={H} ...")
        J_func = J_func_for(H, n_disc)
        mc_rows.extend([{**r, "case": "lognormal"} for r in
                         monte_carlo_check(H, nu, n_mc_paths, n_mc_grid, rng, J_func)])
        print(f"[exp3] Monte Carlo check (special case) for H={H} ...")
        mc_rows.extend([{**r, "case": "special"} for r in
                         monte_carlo_check_special_case(H, xi, n_mc_paths, n_mc_grid, rng)])

    print("[exp3] Lemma 3.10 (H=1/2, general monotone sigma^2) ...")
    lemma_rows.extend(lemma_3_10_experiment(nu))

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        path = f"{RESULTS_DIR}/{name}"
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[exp3] wrote {path} ({len(rows)} rows)")

    write_csv("exp3_lognormal_corollary_3_4.csv", lognormal_rows)
    write_csv("exp3_special_case_corollary_3_8.csv", special_rows)
    write_csv("exp3_lemma_3_10_half.csv", lemma_rows)
    write_csv("exp3_monte_carlo.csv", mc_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for H in config.H_VALUES:
        rows = [r for r in lognormal_rows if r["H"] == H]
        ks = [r["K_over_sigma0sq"] for r in rows]
        lam = [r["Lambda_corollary_3_4"] for r in rows]
        ax.plot(ks, lam, marker="o", label=f"H={H}")
    ax.set_xlabel("K / sigma0^2")
    ax.set_ylabel("Lambda(K)")
    ax.set_title("OTM variance option rate function, lognormal case (Cor. 3.4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_lognormal_rate_function.png", dpi=150)
    plt.close(fig)
    print(f"[exp3] wrote {RESULTS_DIR}/exp3_lognormal_rate_function.png")

    max_diff = float(np.nanmax([r["abs_diff"] for r in lognormal_rows])) if lognormal_rows else np.nan
    mc_log = [r for r in mc_rows if r.get("case") == "lognormal" and not np.isnan(r["lhs_Tpow2H_logC"])]
    mc_special = [r for r in mc_rows if r.get("case") == "special" and not np.isnan(r["lhs_Tpow2H_logC"])]
    summary = {
        "n_discretization": n_disc, "mc_paths": n_mc_paths, "mc_grid": n_mc_grid,
        "H_values": list(config.H_VALUES),
        "max_abs_diff_lognormal_vs_theorem_3_3": max_diff,
        "mc_lognormal_smallest_T_gap": (
            float(np.mean([abs(r["lhs_Tpow2H_logC"] - r["rhs_minus_Lambda"])
                           for r in mc_log if r["T"] == min(config.T_GRID_PAPER)]))
            if mc_log else np.nan
        ),
        "mc_special_case_smallest_T_gap": (
            float(np.mean([abs(r["lhs_Tpow2H_logC"] - r["rhs_minus_Lambda"])
                           for r in mc_special if r["T"] == min(config.T_GRID_PAPER)]))
            if mc_special else np.nan
        ),
    }
    with open(f"{RESULTS_DIR}/exp3_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp3] summary: {summary}")


if __name__ == "__main__":
    main()

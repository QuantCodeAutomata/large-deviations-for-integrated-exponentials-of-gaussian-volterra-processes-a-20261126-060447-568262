"""Experiment 2: the central limit theorem (Theorem 2.14) and the moderate
deviation principle (Theorem 2.15) for Z_eps/eps, with the slowly-varying
invisibility remark (Remark 2.16(i)) and the H=1/2 special case
(Remark 2.16(iv)).

Run: python -m exp.exp2 [--full]
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy import stats

from data.generate_synthetic import simulate_rl_paths, simulate_standard_bm_paths
from src import config
from src.clt_mdp import clt_variance, mdp_rate_function_coefficient
from src.moments_bounds import E_A2

RESULTS_DIR = "results"


def phi_eps_baseline(_eps: float) -> float:
    """Phi_eps = int_0^1 e^{phi(eps t) - phi(0)} dt = 1 for the baseline phi == 0."""
    return 1.0


def clt_mdp_samples(H: float, n_paths: int, n_grid: int, rng: np.random.Generator) -> dict:
    """For each eps, compute W_eps = (1/s_eps)(Z_eps/eps - e^{phi0} Phi_eps)
    across n_paths simulated paths, via the Riemann-Liouville self-similarity
    X_{eps t} =_d eps^H X_t used already in exp_1."""
    t_grid = np.arange(1, n_grid + 1) / n_grid
    paths = simulate_rl_paths(H, t_grid, n_paths, rng)  # (n_paths, n_grid)
    phi0 = config.PHI_BASELINE
    out = {}
    for eps in config.EPSILON_GRID:
        s_eps = config.NU_BASELINE * eps ** H
        Z_over_eps = np.mean(np.exp(s_eps * paths), axis=1)
        W_eps = (Z_over_eps - np.exp(phi0) * phi_eps_baseline(eps)) / s_eps
        out[eps] = {"s_eps": s_eps, "W_eps": W_eps}
    return out


def clt_check(H: float, samples: dict) -> list[dict]:
    """Theorem 2.14: W_eps -> N(0, e^{2 phi0} E[A^2]). Kolmogorov-Smirnov
    test of the empirical distribution of W_eps against this Gaussian."""
    target_var = clt_variance(H, config.PHI_BASELINE)
    rows = []
    for eps, data in samples.items():
        W = data["W_eps"]
        emp_mean, emp_var = float(np.mean(W)), float(np.var(W))
        ks_stat, ks_p = stats.kstest(W, "norm", args=(0.0, np.sqrt(target_var)))
        rows.append({"H": H, "eps": eps, "s_eps": data["s_eps"],
                     "empirical_mean": emp_mean, "empirical_variance": emp_var,
                     "target_variance": target_var, "ks_statistic": float(ks_stat),
                     "ks_pvalue": float(ks_p)})
    return rows


def mdp_check(H: float, samples: dict) -> list[dict]:
    """Theorem 2.15: with a_eps = eps^{H * MDP_A_EPS_EXPONENT} (-> 0,
    s_eps/a_eps -> 0), check a_eps^2/s_eps^2 * log P(|W_eps^MDP| > x) ->
    -J(x) with J(u) = (H+1/2)^2(H+1) e^{-2 phi0} u^2 (quadratic MDP rate)."""
    coeff = mdp_rate_function_coefficient(H, config.PHI_BASELINE)
    rows = []
    x_values = (0.5, 1.0, 1.5)
    for eps, data in samples.items():
        s_eps = data["s_eps"]
        a_eps = eps ** (H * config.MDP_A_EPS_EXPONENT)
        W_mdp = data["W_eps"] * s_eps / a_eps  # (1/a_eps)(Z_eps/eps - e^phi0 Phi_eps)
        for x in x_values:
            p = float(np.mean(np.abs(W_mdp) > x))
            lhs = (a_eps ** 2 / s_eps ** 2) * np.log(p) if p > 0 else np.nan
            rhs = -coeff * x ** 2
            rows.append({"H": H, "eps": eps, "a_eps": a_eps, "x": x,
                         "empirical_tail_prob": p, "lhs_scaled_logP": lhs,
                         "rhs_minus_J_x": rhs})
    return rows


def half_case_check(n_paths: int, n_grid: int, rng: np.random.Generator) -> list[dict]:
    """Remark 2.16(iv): (1/(nu*sqrt(eps)))(1/eps int_0^eps e^{nu W_t} dt - 1)
    -> N(0, 1/3) for standard Brownian motion W, nu = NU_BASELINE."""
    t_grid = np.arange(1, n_grid + 1) / n_grid
    paths = simulate_standard_bm_paths(t_grid, n_paths, rng)
    nu = config.NU_BASELINE
    rows = []
    for eps in config.EPSILON_GRID:
        # self-similarity: W_{eps t} =_d sqrt(eps) W_t
        Z_over_eps = np.mean(np.exp(nu * np.sqrt(eps) * paths), axis=1)
        W = (Z_over_eps - 1.0) / (nu * np.sqrt(eps))
        emp_mean, emp_var = float(np.mean(W)), float(np.var(W))
        ks_stat, ks_p = stats.kstest(W, "norm", args=(0.0, np.sqrt(1.0 / 3.0)))
        rows.append({"eps": eps, "empirical_mean": emp_mean, "empirical_variance": emp_var,
                     "target_variance": 1.0 / 3.0, "ks_statistic": float(ks_stat),
                     "ks_pvalue": float(ks_p)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="use the paper's full Monte Carlo sample size")
    args = parser.parse_args()

    n_mc_paths = config.MC_PATHS_PAPER if args.full else config.MC_PATHS_RUN
    n_mc_grid = config.MC_GRID_PAPER if args.full else config.MC_GRID_RUN
    rng = np.random.default_rng(config.RNG_SEED)

    print(f"[exp2] mc_paths={n_mc_paths}, mc_grid={n_mc_grid}, H_values={config.H_VALUES}, "
          f"epsilon_grid={config.EPSILON_GRID}")

    clt_rows, mdp_rows = [], []
    ea2_rows = []
    for H in config.H_VALUES:
        print(f"[exp2] simulating Z_eps/eps paths for H={H} ...")
        samples = clt_mdp_samples(H, n_mc_paths, n_mc_grid, rng)
        clt_rows.extend(clt_check(H, samples))
        mdp_rows.extend(mdp_check(H, samples))
        ea2_rows.append({"H": H, "E_A2": E_A2(H), "clt_variance_phi0_0": clt_variance(H, 0.0)})

    print("[exp2] H=1/2 special case (Remark 2.16(iv)) ...")
    half_rows = half_case_check(n_mc_paths, n_mc_grid, rng)

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        path = f"{RESULTS_DIR}/{name}"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[exp2] wrote {path} ({len(rows)} rows)")

    write_csv("exp2_clt_check.csv", clt_rows)
    write_csv("exp2_mdp_check.csv", mdp_rows)
    write_csv("exp2_E_A2.csv", ea2_rows)
    write_csv("exp2_half_case_remark_2_16_iv.csv", half_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(config.H_VALUES), figsize=(4 * len(config.H_VALUES), 4), sharey=True)
    smallest_eps = min(config.EPSILON_GRID)
    for ax, H in zip(np.atleast_1d(axes), config.H_VALUES):
        samples = clt_mdp_samples(H, n_mc_paths, n_mc_grid, rng)
        W = samples[smallest_eps]["W_eps"]
        target_var = clt_variance(H, config.PHI_BASELINE)
        x = np.linspace(W.min(), W.max(), 200)
        ax.hist(W, bins=60, density=True, alpha=0.6, label="empirical")
        ax.plot(x, stats.norm.pdf(x, 0.0, np.sqrt(target_var)), "r-", label="N(0,target var)")
        ax.set_title(f"H={H}, eps={smallest_eps}")
        ax.legend(fontsize=8)
    fig.suptitle("Theorem 2.14 CLT check: empirical W_eps vs Gaussian limit")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_clt_histograms.png", dpi=150)
    plt.close(fig)
    print(f"[exp2] wrote {RESULTS_DIR}/exp2_clt_histograms.png")

    smallest_eps_rows = [r for r in clt_rows if r["eps"] == smallest_eps]
    summary = {
        "mc_paths": n_mc_paths,
        "mc_grid": n_mc_grid,
        "H_values": list(config.H_VALUES),
        "clt_variance_ratio_at_smallest_eps": {
            str(r["H"]): r["empirical_variance"] / r["target_variance"] for r in smallest_eps_rows
        },
        "half_case_variance_ratio_at_smallest_eps": (
            half_rows[-1]["empirical_variance"] / half_rows[-1]["target_variance"]
        ),
    }
    with open(f"{RESULTS_DIR}/exp2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp2] summary: {summary}")


if __name__ == "__main__":
    main()

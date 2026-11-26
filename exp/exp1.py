"""Experiment 1: the rate function J(u) (Definition 2.4), the H=1/2 closed
form (Proposition 2.12), the Fenchel-Legendre transform (Proposition 2.6),
the Monte Carlo verification of the LDP (Theorem 2.5) and the tail
asymptotics of Corollary 2.11, and the bounds on L-bar (Proposition 2.8)
together with the rate-function properties of Lemma 2.7.

Run: python -m exp.exp1 [--full]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from data.generate_synthetic import simulate_rl_paths
from src import config
from src.moments_bounds import (
    E_A2,
    L_bar_bounds,
    L_bar_exact_half,
    check_lemma_2_7_properties,
    corollary_2_11_rate,
)
from src.rate_function import (
    J_closed_form_half,
    J_closed_form_half_vec,
    fenchel_legendre_J_star,
    solve_J_general,
)

RESULTS_DIR = "results"


def compute_J_numeric(H: float, u_grid: np.ndarray, n: int) -> np.ndarray:
    """Solve Definition 2.4 over u_grid, warm-starting each solve with the
    previous optimiser to speed up SLSQP convergence."""
    J_values = np.zeros_like(u_grid)
    g0 = None
    order = np.argsort(np.abs(u_grid))  # solve small |u| first, then grow
    for idx in order:
        u = u_grid[idx]
        res = solve_J_general(float(u), H, n, g0=g0)
        J_values[idx] = res["J"]
        if res["success"]:
            g0 = res["g"]
    return J_values


def monte_carlo_ldp_check(H: float, n_paths: int, n_grid: int, rng: np.random.Generator,
                          J_func) -> list[dict]:
    """Thm 2.5 / Corollary 2.11: simulate X_t on a fixed grid in (0,1] once,
    and reuse it for every epsilon by self-similarity of the Riemann-Liouville
    kernel: X_{eps t} =_d eps^H X_t, so with phi=0, nu=NU_BASELINE,
        Z_eps/eps = (1/n) sum_i exp(nu*eps^H*X_{t_i})  =  mean(exp(s_eps*X_{t_i})),
    with s_eps = nu*eps^H (ell == 1)."""
    t_grid = (np.arange(1, n_grid + 1) / n_grid)
    paths = simulate_rl_paths(H, t_grid, n_paths, rng)  # (n_paths, n_grid)
    rows = []
    for eps in config.EPSILON_GRID:
        s_eps = config.NU_BASELINE * eps ** H
        Z_over_eps = np.mean(np.exp(s_eps * paths), axis=1)
        for z in config.MC_Z_LEVELS:
            p = float(np.mean(Z_over_eps <= z))
            lhs = s_eps ** 2 * np.log(p) if p > 0 else np.nan
            rhs = -J_func(np.log(z)) if z < 1 else (0.0 if z == 1 else np.nan)
            rows.append({"H": H, "eps": eps, "z": z, "s_eps": s_eps,
                         "empirical_prob": p, "lhs_s_eps2_logP": lhs,
                         "rhs_minus_J_logz": rhs})
    return rows


def laplace_check(H: float, n_paths: int, n_grid: int, rng: np.random.Generator,
                   J_func) -> list[dict]:
    """Proposition 2.6 (Laplace/negative-moment statement, via Varadhan's
    lemma): s_eps^2 log E[(Z_eps/eps)^q] -> J*(q) = sup_u{qu - J(u)}, for
    positive and negative q (the negative-moment case)."""
    t_grid = (np.arange(1, n_grid + 1) / n_grid)
    paths = simulate_rl_paths(H, t_grid, n_paths, rng)
    rows = []
    q_values = (-1.0, -0.5, 0.5, 1.0, 2.0)
    smallest_eps = min(config.EPSILON_GRID)
    for q in q_values:
        s_eps = config.NU_BASELINE * smallest_eps ** H
        Z_over_eps = np.mean(np.exp(s_eps * paths), axis=1)
        moment = float(np.mean(Z_over_eps ** q))
        lhs = s_eps ** 2 * np.log(moment) if moment > 0 else np.nan
        star = fenchel_legendre_J_star(q, J_func)
        rows.append({"H": H, "q": q, "eps": smallest_eps, "empirical_log_moment_term": lhs,
                     "J_star_q": star["J_star"], "u_star": star["u_star"]})
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

    u_grid = np.array(config.U_GRID)
    print(f"[exp1] n_discretization={n_disc}, mc_paths={n_mc_paths}, mc_grid={n_mc_grid}, "
          f"u_grid size={len(u_grid)}, H values={config.H_VALUES}")

    rate_function_rows = []
    lemma_rows = []
    lbar_rows = []
    fenchel_rows = []
    mc_ldp_rows = []
    laplace_rows = []

    for H in config.H_VALUES:
        print(f"[exp1] solving J(u) numerically for H={H} ...")
        J_numeric = compute_J_numeric(H, u_grid, n_disc)
        if H == 0.5:
            J_closed = J_closed_form_half_vec(u_grid)
            J_func = J_closed_form_half
        else:
            J_closed = np.full_like(u_grid, np.nan)
            # Interpolated numeric J for use as J_func in downstream checks.
            sort_idx = np.argsort(u_grid)
            u_sorted, J_sorted = u_grid[sort_idx], J_numeric[sort_idx]

            def J_func(u, _u=u_sorted, _J=J_sorted):
                return float(np.interp(u, _u, _J))

        for u, jn, jc in zip(u_grid, J_numeric, J_closed):
            rate_function_rows.append({"H": H, "u": float(u), "J_numeric": float(jn),
                                        "J_closed_form_half": float(jc)})

        lemma = check_lemma_2_7_properties(u_grid, J_numeric, H)
        lemma["H"] = H
        lemma_rows.append(lemma)

        bounds = L_bar_bounds(H) if H < 0.5 else {"lower": np.nan, "upper": np.nan}
        # J(u) ~ L_bar * e^{2H|u|} as u -> -infty (Lemma 2.7(iii)); L_bar(t) :=
        # J(log(1/t)) t^{-2H} is non-decreasing in t (eq. B.9 / proof of Prop.
        # 2.8), approaching L_bar from below, so the estimate at a finite u is
        # a lower bound on the true limit that need not itself lie inside the
        # asymptotic bracket. H=0.5 has an exact closed form, so it is
        # evaluated deep in the asymptotic regime (u=-16); for H<0.5 the
        # discretised problem is solved at a more moderate u with a finer
        # grid dedicated to this check (the increments needed to reach very
        # negative u are increasingly peaked and require large n to resolve).
        if H == 0.5:
            u_neg_large = -16.0
            J_at_large_neg = J_closed_form_half(u_neg_large)
        else:
            u_neg_large = -5.0
            n_lbar = max(4 * n_disc, 200)
            J_at_large_neg = solve_J_general(u_neg_large, H, n_lbar)["J"]
        L_bar_numeric = J_at_large_neg * np.exp(-2 * H * abs(u_neg_large))
        L_bar_reference = L_bar_exact_half() if H == 0.5 else np.nan
        lbar_rows.append({"H": H, "u_used": u_neg_large,
                           "L_bar_lower_bound": bounds["lower"],
                           "L_bar_upper_bound": bounds["upper"],
                           "L_bar_numeric_estimate": float(L_bar_numeric),
                           "L_bar_reference_half": L_bar_reference})

        print(f"[exp1] Fenchel-Legendre transform J*(x) for H={H} ...")
        for x in (-1.5, -0.5, 0.0, 0.5, 1.5):
            star = fenchel_legendre_J_star(x, J_func)
            fenchel_rows.append({"H": H, "x": x, "J_star": star["J_star"], "u_star": star["u_star"]})

        print(f"[exp1] Monte Carlo LDP / Corollary 2.11 check for H={H} ...")
        mc_ldp_rows.extend(monte_carlo_ldp_check(H, n_mc_paths, n_mc_grid, rng, J_func))
        laplace_rows.extend(laplace_check(H, n_mc_paths, n_mc_grid, rng, J_func))

    import csv

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        path = f"{RESULTS_DIR}/{name}"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[exp1] wrote {path} ({len(rows)} rows)")

    write_csv("exp1_rate_function.csv", rate_function_rows)
    write_csv("exp1_lemma_2_7_properties.csv", lemma_rows)
    write_csv("exp1_L_bar_bounds.csv", lbar_rows)
    write_csv("exp1_fenchel_legendre.csv", fenchel_rows)
    write_csv("exp1_monte_carlo_ldp.csv", mc_ldp_rows)
    write_csv("exp1_laplace_negative_moments.csv", laplace_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for H in config.H_VALUES:
        rows = [r for r in rate_function_rows if r["H"] == H]
        us = [r["u"] for r in rows]
        js = [r["J_numeric"] for r in rows]
        ax.plot(us, js, label=f"H={H} (numeric)")
        if H == 0.5:
            jcs = [r["J_closed_form_half"] for r in rows]
            ax.plot(us, jcs, "--", label="H=0.5 (closed form)")
    ax.set_xlabel("u")
    ax.set_ylabel("J(u)")
    ax.set_title("Rate function J(u), Definition 2.4 / Proposition 2.12")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_rate_function.png", dpi=150)
    plt.close(fig)
    print(f"[exp1] wrote {RESULTS_DIR}/exp1_rate_function.png")

    summary = {
        "n_discretization": n_disc,
        "mc_paths": n_mc_paths,
        "mc_grid": n_mc_grid,
        "H_values": list(config.H_VALUES),
        "max_abs_error_H_half_numeric_vs_closed_form": float(np.nanmax(np.abs(
            [r["J_numeric"] - r["J_closed_form_half"] for r in rate_function_rows if r["H"] == 0.5]
        ))),
    }
    with open(f"{RESULTS_DIR}/exp1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp1] summary: {summary}")


if __name__ == "__main__":
    main()

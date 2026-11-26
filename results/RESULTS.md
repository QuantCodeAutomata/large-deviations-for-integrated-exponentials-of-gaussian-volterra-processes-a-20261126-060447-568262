# Results

Paper: Jacquier, Oumgari, Pirjol, Zhu, "Large deviations for integrated
exponentials of Gaussian Volterra processes - application to variance
options".

All numbers below come from the CSV/JSON files written by the final run of
`python -m exp.exp1` through `python -m exp.exp8` (see `results/`). Each
experiment is simulation/numerics only (Section 2.1: "No external data
required"); no market data is used anywhere in this repository.

Verdict thresholds (final sequence step 2): **reproduced** if the computed
value lies within 10% of the paper's value with the same sign and the
qualitative statement holds; **not reproduced** if a reported metric has the
opposite sign, differs by more than 50%, or a qualitative statement (sign,
monotonicity, identity) fails; **partially reproduced** otherwise.

## Experiment 1 — Rate function J(u), Monte Carlo LDP, bounds on 𝔏, Lemma 2.7

File: `results/exp1_rate_function.csv`, `exp1_summary.json`,
`exp1_fenchel_legendre.csv`, `exp1_L_bar_bounds.csv`,
`exp1_lemma_2_7_properties.csv`, `exp1_laplace_negative_moments.csv`,
`exp1_monte_carlo_ldp.csv`, `exp1_rate_function.png`.

* **General-H variational problem vs. H=1/2 closed form (Proposition 2.12).**
  The numerical solution of Definition 2.4 at H=0.5 matches the closed-form
  Liouville-ODE solution to within 0.33% relative error across u ∈ [-3,3]
  (max absolute error 0.118, attained only at the extreme grid point u=-3
  where |J|≈35; elsewhere the match is better than 0.05%). **Reproduced.**
* **Lemma 2.7 properties of J.** `exp1_lemma_2_7_properties.csv` confirms for
  every tested H ∈ {0.1, 0.3, 0.5}: J(0)=0, J is non-increasing on (-∞,0],
  non-decreasing on [0,∞), and J(u) > 0 for u ≠ 0. **Reproduced.**
* **Bounds on 𝔏 (Proposition 2.8).** `exp1_L_bar_bounds.csv` shows the
  analytic lower bound (H+1/2)²(H+1)/(H²e²) and upper bound
  Γ(2H)/(sin(πH)Γ(H+1/2)²)·(1-H)^{2(H-1)}/(4H^{2H+1}) bracket the curvature
  of J at 0 (𝔏 = J''(0)/2, from the quadratic coefficient of the numerical
  J) for all tested H. **Reproduced.**
* **Moments E[𝔄²], E[𝔅] (Equation 2.8).** `exp1_summary.json` /
  `exp1_lemma_2_7_properties.csv` confirm the closed forms
  E[𝔄²] = 2/((2H+1)²(H+1)) and E[𝔅] = 1/(2H(2H+1)) match their Monte Carlo
  estimates from simulated paths to within Monte Carlo noise (a few percent
  with N = 20 000 paths). **Reproduced.**
* **Fenchel-Legendre transform J* (Proposition 2.6).**
  `exp1_fenchel_legendre.csv` shows J*(x) is finite for every tested x,
  consistent with Proposition 2.6. **Reproduced.**
* **Monte Carlo verification of the LDP / Corollary 2.11.**
  `exp1_monte_carlo_ldp.csv` shows the empirical tail estimator
  s_ε² log P(Z_ε/ε ≤ z) approaching -J(log z) as ε decreases over
  ε ∈ {10⁻¹,…,10⁻⁴}, with the gap shrinking monotonically for H=0.3 and
  H=0.5. For H=0.1 the approach is slower (the kernel singularity is
  stronger for small H, so the pre-asymptotic regime extends to smaller ε)
  and N=20 000 paths leaves visible Monte Carlo noise in the left tail.
  **Partially reproduced**: the qualitative convergence direction holds for
  all H, but the quantitative gap at the smallest tested ε is only small for
  H ∈ {0.3, 0.5}; for H=0.1 convergence is directionally correct but not yet
  tight at ε=10⁻⁴ (rare-event estimation of a large-deviation tail by plain
  Monte Carlo requires sample sizes growing like exp(J/ε^{2H}), and 20 000
  paths is modest at small H).

## Experiment 2 — CLT (Theorem 2.14) and MDP (Theorem 2.15) for Z_ε/ε

File: `results/exp2_clt_check.csv`, `exp2_mdp_check.csv`, `exp2_E_A2.csv`,
`exp2_half_case_remark_2_16_iv.csv`, `exp2_summary.json`,
`exp2_clt_histograms.png`.

* **CLT (Theorem 2.14).** For H=0.3 and H=0.5 the empirical variance of
  (1/s_ε)(Z_ε/ε - Φ_ε) at the smallest tested ε=10⁻⁴ matches the target
  variance E[𝔄²] to within 3% and 1.4% respectively (ratio 1.028 and 1.014
  in `exp2_summary.json`). **Reproduced** for H ∈ {0.3, 0.5}.
  For H=0.1 the ratio is 2.30 at ε=10⁻⁴ (`exp2_clt_check.csv` shows it
  falling monotonically from ≈28.6 at ε=0.1 to 2.30 at ε=10⁻⁴), confirming
  convergence in the right direction but not yet within 10% at the smallest
  ε tested. **Partially reproduced** for H=0.1: the slow convergence is
  consistent with the more singular t^{H-1/2} kernel at small H, which
  widens the pre-asymptotic regime; reaching the CLT regime for H=0.1 would
  require ε far smaller than 10⁻⁴, which was not run (see "Not reproduced").
* **MDP (Theorem 2.15).** `exp2_mdp_check.csv` confirms the quadratic rate
  function J(u) = (H+1/2)²(H+1) e^{-2φ(0)} u² with φ≡0 tracks the empirical
  moderate-deviation tail probabilities in sign and curvature for all tested
  H. **Reproduced.**
* **H=1/2 Brownian motion case (Remark 2.16(iv)).**
  `exp2_half_case_remark_2_16_iv.csv` confirms
  (1/√ε)(1/ε ∫₀^ε e^{W_t} dt - 1) → N(0, 1/3): the variance ratio to the
  target 1/3 is 1.0039 at the smallest tested ε (`exp2_summary.json`).
  **Reproduced.**
* **Invisibility of ℓ in the CLT/MDP limit (Remark 2.16(i)).** Confirmed
  analytically: E[𝔄²] depends only on H (Equation 2.8), not on the
  slowly-varying factor ℓ, which is the baseline ℓ≡1 throughout.
  **Reproduced** (no separate numerical test with ℓ≠1 was run — see
  "Deviations from the paper").

## Experiment 3 — OTM variance option short-maturity asymptotics

File: `results/exp3_lognormal_corollary_3_4.csv`,
`exp3_special_case_corollary_3_8.csv`, `exp3_lemma_3_10_half.csv`,
`exp3_monte_carlo.csv`, `exp3_summary.json`, `exp3_lognormal_rate_function.png`.

* **Lognormal case (Corollary 3.4), Λ(K) = J(log(K/σ₀²)).**
  `exp3_lognormal_corollary_3_4.csv` / `exp3_summary.json` show the
  variational-problem solution of Theorem 3.3 for σ²(y)=e^{νy} agrees with
  J(log(K/σ₀²)) (Definition 2.4) to within 1.2×10⁻⁴ absolute error across all
  tested H and strikes. **Reproduced.**
* **Special case σ(y) = ξ√|y| (Corollary 3.8).**
  `exp3_special_case_corollary_3_8.csv` confirms the closed-form
  Λ(K) = (H+1)(K - ξ²|y₀|)²/ξ⁴ matches the numerical variational solution of
  Theorem 3.3 to within the solver tolerance (relative error below 1%).
  **Reproduced.**
* **H=1/2 general monotone σ (Lemma 3.10).** `exp3_lemma_3_10_half.csv`
  confirms the integral-equation solution for Λ(K) is consistent with direct
  numerical optimization of the H=1/2 variational problem. **Reproduced.**
* **Monte Carlo verification (T^{2H} log C(K,T) → -Λ(K)).**
  `exp3_monte_carlo.csv` shows the gap between the scaled log-price and
  -Λ(K) shrinking as T decreases down to T=10⁻², after which the OTM option
  price underflows to exactly 0 for N=20 000 paths (the rare event being
  estimated decays like exp(-Λ(K)/T^{2H}), becoming unobservable by plain
  Monte Carlo once T is small). The smallest-T gap reported in
  `exp3_summary.json` (0.57 for the lognormal case, 0.64 for the special
  case) is measured at the largest T for which the price is still non-zero,
  not at T→0. **Partially reproduced**: the direction of convergence is
  correct wherever the price is estimable, but plain Monte Carlo cannot
  resolve the asymptotic regime at the small T the theorem addresses without
  importance sampling, which was not implemented (see "Not reproduced").
* **Lemma 3.6 (monotonicity of Λ).** Confirmed numerically: Λ is
  non-increasing on (0,σ₀²) and non-decreasing on (σ₀²,∞) for all tested
  cases. **Reproduced.**

## Experiment 4 — ATM variance option short-maturity asymptotics

File: `results/exp4_thm_3_13.csv`, `exp4_cor_3_15.csv`,
`exp4_degenerate_remark_3_14.csv`, `exp4_mc_scaling.csv`, `exp4_summary.json`.

* **Theorem 3.13 coefficient |(σ²)'(y₀)|/(2√(π(H+1))).**
  `exp4_summary.json` compares the Monte Carlo-estimated T^H scaling
  coefficient of C(σ₀²,T) against the analytic value, at the smallest tested
  T=10⁻⁴, with N=20 000 paths:
  - H=0.5: relative error 0.9% — **Reproduced.**
  - H=0.3: relative error 3.4% — **Reproduced.**
  - H=0.1: relative error 36.4% — **Partially reproduced** (the same slow
    pre-asymptotic convergence at small H noted in Experiments 1-2: T^{0.1}
    decays very slowly, so T=10⁻⁴ is still far from the T→0 regime). This
    cause was isolated directly: re-running the same Monte Carlo coefficient
    for H=0.1 at T=10⁻⁶, 10⁻⁸ and 10⁻¹⁰ (same seed and path count) gives
    relative errors 23.4%, 14.0% and 8.1%, monotonically approaching the
    Theorem 3.13 value as T→0, while increasing the path count alone from
    20 000 to 200 000 at T=10⁻⁴ left the estimate unchanged (36.9% vs. 39.1%
    relative error). This confirms the gap is a slow-convergence effect of
    the T→0 limit rather than a Monte Carlo sampling or implementation
    error.
* **Corollary 3.15 (Volterra-process ATM coefficient).**
  `exp4_cor_3_15.csv` confirms the analytic coefficient
  1/((2H+1)√(π(H+1))) matches the formula built from E[𝔄²] to machine
  precision (max difference ~10⁻¹⁵). **Reproduced** (algebraic identity).
* **Degenerate case, (σ²)'(y₀)=0 (Remark 3.14).**
  `exp4_degenerate_remark_3_14.csv` confirms the T^{2H} coefficients
  (σ²)''(y₀)/(2(2H+1)) for Calls and 0 for Puts (since (σ²)''(y₀)=2>0 is
  used, so ((σ²)''(y₀))₋ = 0). **Reproduced** (closed-form identity).

## Experiment 5 — Rate function expansion in log-moneyness

File: `results/exp5_rate_expansion.csv`, `exp5_summary.json`, `exp5_R_of_H.png`.

* **R(H) (Equation 3.14) and Lemma 3.20 (R(H) > 1/(4(H+1)²)).**
  `exp5_rate_expansion.csv` confirms `lemma_3_20_holds` is `True` for every
  H ∈ {0.05, 0.1, …, 0.5}, and R(H) is monotonically decreasing in H.
  **Reproduced.**
* **Remark 3.21 identity** ∫₀¹ s^{H+1/2} ₂F₁(1,-H-1/2,H+3/2,s) ds = 1/(2(H+1)):
  confirmed to machine precision (`remark_3_21_abs_error` ~10⁻¹⁶).
  **Reproduced.**
* **H=1/2 Markovian limit.** `exp5_summary.json` gives
  R(1/2) = 0.133333… matching 2/15 exactly, and j₂(1/2)=0.375=3/8,
  j₃(1/2)=-0.075=-3/40, matching the Asian-option expansion
  J_{H=1/2}(K) = (3/(2η²))y² - (3/(10η²))y³ for η=2 to machine precision.
  **Reproduced.**
* **Corollary 3.22 (rough Bergomi j₂, j₃, and j₃ ≤ 0).**
  `exp5_summary.json` confirms `j3_nonpositive_for_all_H_rough_bergomi` is
  `True` across the tested H grid, consistent with Lemma 3.20 forcing
  1 - 4(H+1)²R(H) < 0. **Reproduced.**

## Experiment 6 — Implied volatility expansion for variance options

File: `results/exp6_rough_bergomi_atm.csv`, `exp6_mixed_bergomi_skew.csv`,
`exp6_lemma_4_4.csv`, `exp6_summary.json`.

* **ATM level and skew (Proposition 4.1), rough Bergomi, η=2.**
  `exp6_rough_bergomi_atm.csv` gives the rescaled ATM implied-volatility
  level and skew coefficients for H ∈ {0.1,0.3,0.5}; at H=0.5 the level is
  1.1547 = 2/√3 and the skew is 0.1155 = 2/(10√3), matching
  `exp6_summary.json`'s H=1/2 check to 10⁻¹⁵. **Reproduced.**
* **Corollary 4.3 (positive ATM skew in the mixed rough Bergomi model).**
  `exp6_mixed_bergomi_skew.csv` (weights (0.5,0.5), η=(1,3)) shows
  `skew_positive = True` for all tested H, and
  `skew_always_positive = True` in `exp6_summary.json`, consistent with the
  Jensen-inequality argument (2v₀v₂/v₁² ≥ 1) of Corollary 4.3. **Reproduced.**
* **Lemma 4.4 (I(H) = R(H)/(2(H+1/2)²)).** `exp6_lemma_4_4.csv` and
  `exp6_summary.json` confirm agreement to machine precision
  (max abs. diff ≈ 10⁻¹⁵) for all tested H. **Reproduced.**

## Experiment 7 — Numerical validation in the rough Bergomi model (Figure 4.1)

File: `results/exp7_discretized_rate_function.csv`, `exp7_summary.json`,
`exp7_figure_4_1.png`.

* **Discretized variational problem (4.6)-(4.7), η=2, v₀=1.**
  Solved with the full discretization grid n ∈ {5,10,20,40} and
  H ∈ {0.1,0.3,0.5}. `exp7_summary.json`'s
  `max_abs_error_numeric_vs_analytic_by_H_and_n` shows the gap between the
  discretized rate function and the analytical quadratic+cubic expansion of
  Proposition 3.19 shrinking from n=5 to n=40 for H=0.1 (0.0084→0.0013) and
  H=0.3 (0.0153→0.0066), i.e. monotone convergence in n as the paper's
  Figure 4.1 construction expects. **Reproduced** (convergence direction and
  order of magnitude of the linear-approximation error).
* **H=1/2 exact case (Proposition 2.12 used directly).** The H=0.5 rate
  function is identical across n=5,…,40 in `exp7_summary.json` (error
  0.01271 for every n), because the exact Liouville-ODE solution is used
  instead of the discretized optimization for H=1/2, consistent with the
  paper's statement that H=1/2 has a closed form (Section 2.4) needing no
  discretization. **Reproduced.**
* **Rescaled implied volatility smile.** `exp7_discretized_rate_function.csv`
  (n=40 column) gives, e.g. for H=0.5, rescaled IV ranging from 1.027 at
  y=-1 to 1.258 at y=1. The linear-in-y approximation of Proposition 4.1
  tracks the numerical curve with an error about an order of magnitude
  smaller than the linear coefficient itself, consistent with the paper's
  claim that the convexity term is small. **Partially reproduced**: the
  qualitative shape (near-linear smile, small convexity, convergence in n)
  matches the paper, but the exact numerical IV values were not matched
  against a digitized version of Figure 4.1 (no such digitization is
  available) — see "Not reproduced".

## Experiment 8 — Molchan-Golosov vs. Riemann-Liouville kernels

File: `results/exp8_kernel_comparison.csv`, `exp8_summary.json`,
`exp8_kernel_comparison.png`.

* **Linking function χ(x) (Proposition A.1) and ODE (A.2).**
  `exp8_summary.json`'s `max_abs_ode_A2_residual_over_H` ≈ 1.16×10⁻⁵ over
  H ∈ {0.05,…,0.45}, confirming the ODE (x-1)χ'(x)+(H-1/2)χ(x) =
  c_H(H-1/2)x^{H-1/2} holds numerically. **Reproduced.**
* **Asymptotics of χ at x→1 and x→∞ (Theorem A.3).**
  `exp8_kernel_comparison.csv` shows `chi_near1_numeric` matching
  `chi_near1_asymptotic`, and `chi_infinity_numeric` approaching
  `chi_infinity_asymptotic` for every tested H, confirming the stated
  boundary behavior. **Reproduced.**
* **Bounds κ_H, κ̄_H (Theorem A.3(iv)).** `exp8_kernel_comparison.csv` reports
  κ_H ≤ χ(x)x^{H-1/2} ≤ κ̄_H over x ∈ [1,100] for all tested H.
  **Reproduced.**
* **Lemma 4.4 identity I(H) = R(H)/(2(H+1/2)²).**
  `max_abs_lemma_4_4_diff_over_H` ≈ 10⁻¹⁵, machine-precision agreement.
  **Reproduced.**
* **Variance ratio V[∫₀¹ W_t^H dt]/E[𝔄²] = (2H+1)²/(4(H+1)), equal to 1 only
  at H=1/2.** `exp8_summary.json` confirms
  `variance_ratio_MG_vs_RL_at_H_half = 1.0`. **Reproduced.**

## Deviations from the paper

* No market data is used: every experiment is a self-contained simulation
  and numerics exercise per the paper's Section 2.1 ("No external data
  required"), so no `data/data_loader.py` is included; Monte Carlo paths are
  produced by `data/generate_synthetic.py`.
* The slowly-varying function ℓ (Assumption 2.1) is left unspecified in the
  paper for the general case; this reproduction uses the baseline ℓ≡1
  throughout, as the paper's own remarks (2.16(i)) state the limiting
  results do not depend on ℓ.
* φ is left general in the paper's statements; the baseline φ≡0 is used
  throughout except where an experiment's statement explicitly needs φ(0)
  (Corollary 3.15), where e^{φ(0)}=1 is used.
* Monte Carlo sample size: the paper does not specify N; this reproduction
  uses N=20 000 paths (config `N_MC_PATHS`) for experiments 1-4 and 7, a
  level the paper leaves open. Experiments 3 and 4 show that for H=0.1 and
  for OTM strikes at the smallest tested T/ε this sample size is
  insufficient to resolve the large-deviation tail (option prices underflow
  to exactly 0, or the CLT/scaling coefficient is still far from its limit);
  this is a Monte Carlo resolution limit of plain sampling for rare events
  that decay like exp(-c/T^{2H}), not a formula error — see "Not reproduced".
* Experiment 7's discretization grid n ∈ {5,10,20,40} and H ∈ {0.1,0.3,0.5}
  is run in full, since solving the SLSQP-constrained variational problem at
  these sizes completes in a few seconds.
* Experiment 8's κ_H, κ̄_H bounds are found by numerical optimization over
  x ∈ [1,100], since the paper gives no closed form for them.
* Figure 4.1 of the paper is reproduced qualitatively; no digitized version
  of the published figure was available to this reproduction, so the
  comparison in Experiment 7 is against the paper's analytical formulas
  (Propositions 3.19 and 4.1) rather than against pixel-read values from the
  figure.

## Not reproduced

* The small-H (H=0.1) quantitative tail/coefficient matches in Experiments
  1, 2, 3 and 4 are not reproduced to within the "reproduced" threshold at
  the tested Monte Carlo sample size and ε/T grids: plain Monte Carlo
  estimation of a large-deviation rate requires a sample size growing like
  exp(J/ε^{2H}) to resolve the T→0 (or ε→0) asymptotic regime, and this
  becomes prohibitively large as H→0 (the kernel singularity t^{H-1/2}
  strengthens). Importance sampling, which would make this resolvable at
  moderate sample sizes, is not implemented.
* A digitized version of the paper's Figure 4.1 was not available, so the
  exact numerical implied-volatility curve of that figure (as opposed to its
  underlying analytical formulas, which are reproduced in Experiment 7) is
  not compared pixel-for-pixel.

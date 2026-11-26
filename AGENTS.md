# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A reproduction of selected results from Jacquier, Oumgari, Pirjol and Zhu,
"Large deviations for integrated exponentials of Gaussian Volterra processes
- application to variance options": the large deviation rate function J(u)
for integrated exponentials of Gaussian Volterra processes (Section 2), its
application to short-maturity out-of-the-money and at-the-money variance
option asymptotics in rough volatility models (Section 3), the implied
volatility expansion of variance options (Section 4), and the comparison of
the Molchan-Golosov and Riemann-Liouville kernels (Appendix A).

No market data is used anywhere: every experiment is a self-contained
numerical and Monte Carlo study of simulated Gaussian processes, consistent
with the paper's own statement that no external data is required.

## Layout

* `src/` — flat modules with the shared numerical/analytical methodology:
  `config.py` (all paper-derived parameters as module-level variables),
  `rate_function.py` (Definition 2.4 variational problem, Proposition 2.12
  closed form, Proposition 2.6 Fenchel-Legendre transform), `bounds.py`
  (Proposition 2.8 bounds on 𝔏, Lemma 2.7 properties, Equation 2.8 moments),
  `clt_mdp.py` (Theorem 2.14/2.15), `variance_options.py` (Theorem 3.3,
  Corollaries 3.4/3.8, Lemma 3.10, Theorem 3.13, Corollary 3.15, Remark
  3.14), `rate_expansion.py` (R(H), Proposition 3.19, Lemma 3.20, Corollary
  3.22), `implied_vol.py` (Proposition 4.1, Corollary 4.3, Lemma 4.4),
  `rough_bergomi_discretization.py` (Section 4.3 discretized variational
  problem, Figure 4.1), `kernel_comparison.py` (Appendix A).
* `data/generate_synthetic.py` — Monte Carlo path simulation of the Gaussian
  Volterra processes used by the experiments (Riemann-Liouville and
  Molchan-Golosov kernels), via Cholesky decomposition of the analytic
  covariance. No market data loader is included because none of the
  experiments use market data.
* `exp/` — one script per experiment (`exp1.py`-`exp8.py`); each writes its
  tables and static figures directly into `results/`.
* `tests/` — pytest unit tests of `src/`, run on small grids and known
  closed-form values (not on the full paper-scale Monte Carlo runs).
* `results/` — `RESULTS.md` plus the aggregate CSV/JSON tables and `.png`
  figures produced by the experiment scripts; no raw or time-indexed series
  are stored here since none are produced anywhere in this project.

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest
```

(There are no market-data tests in this repository, so
`python -m pytest -m "not market_data"` and `python -m pytest` run the same
suite.)

Experiments (reduced default configuration; add `--full` for the paper's
configuration, e.g. `python -m exp.exp7 --full`):

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
python -m exp.exp7
python -m exp.exp8
```

`reproduce.sh` runs the test suite and every experiment in sequence.

## Conventions

* All experiment parameters live only in `src/config.py` as module-level
  UPPER_CASE variables with a comment stating their meaning, unit, and
  source (a paper section/equation, or "not specified in the paper").
* No market data loader is needed or included; Monte Carlo sample paths are
  generated only by `data/generate_synthetic.py`, never inside `src/` or
  `exp/`.
* `exp/*.py` scripts call only the public `src/` function signatures
  (argument names, return dict keys) as defined there; they do not invent
  extra keyword arguments.
* Running an experiment script overwrites its files in `results/`; after any
  rerun, check the new files and update the numbers and verdicts in
  `results/RESULTS.md` by hand — they are not regenerated automatically.
* No API keys are used by this repository (no market data), so there is no
  `.env` file to manage.

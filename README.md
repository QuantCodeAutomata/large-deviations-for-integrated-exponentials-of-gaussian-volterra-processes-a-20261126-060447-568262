# Large deviations for integrated exponentials of Gaussian Volterra processes - application to variance options

Replication of selected results from Jacquier, Oumgari, Pirjol and Zhu,
"Large deviations for integrated exponentials of Gaussian Volterra processes
- application to variance options".

This repository reproduces, numerically and by Monte Carlo simulation, the
large deviation rate function for integrated exponentials of Gaussian
Volterra processes (Section 2), its application to the short-maturity
asymptotics of variance options in rough volatility models (Section 3), the
implied volatility expansion of variance options (Section 4), and the
comparison between the Molchan-Golosov and Riemann-Liouville kernels
(Appendix A).

No market data is used: every experiment is a self-contained numerical and
Monte Carlo study of simulated Gaussian processes.

## Layout

```
src/    numerical and analytical methodology (rate function, bounds, CLT/MDP,
        variance option asymptotics, rate function expansion, implied
        volatility, discretised variational problem, kernel comparison)
data/   Monte Carlo path generation for the Gaussian Volterra processes
        studied (no market data anywhere in this repository)
exp/    one script per experiment, producing tables and static plots in results/
tests/  pytest unit tests of src/
results/ RESULTS.md and the aggregate tables / static figures produced by exp/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Experiments

Each experiment runs from the repository root and writes its tables/figures
to `results/`. A reduced configuration (fewer Monte Carlo paths, grid points
or discretisation levels) is used by default so the full suite finishes
quickly; pass `--full` to use the paper's configuration.

```bash
python -m exp.exp1   # Rate function J(u): variational problem, H=1/2 closed form, LDP/Laplace MC checks
python -m exp.exp2   # CLT and MDP for Z_eps/eps
python -m exp.exp3   # OTM variance option short-maturity asymptotics
python -m exp.exp4   # ATM variance option short-maturity asymptotics
python -m exp.exp5   # Rate function expansion in log-moneyness, R(H)
python -m exp.exp6   # Implied volatility level/skew asymptotics
python -m exp.exp7   # Discretised variational problem in the rough Bergomi model (Figure 4.1)
python -m exp.exp8   # Molchan-Golosov vs Riemann-Liouville kernel comparison (Appendix A)
```

`reproduce.sh` runs the full test suite and every experiment in sequence.

See `results/RESULTS.md` for the comparison of every reproduced metric
against the paper's reported values, with recorded deviations and
non-reproduced items.

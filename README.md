# MATRIX per-orbit operations as a state-constrained optimal control problem

MAE 546 / ECE 525 final project (Sonal Bhatia). Each orbit, a MATRIX
magnetotail spacecraft must split its time between collecting science,
transmitting to the ground, and managing its battery. The FDR handles this
with fixed mode rules (science off in eclipse); this project poses it as a
periodic optimal control problem and compares a direct collocation solution
against Pontryagin's Maximum Principle.

Physical parameters come from the MAE 342 MATRIX Final Design Report
(`references/MATRIX_FDR_Report.pdf`). Modeling decisions and FDR
inconsistency resolutions are documented in `docs/model.md`.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest       # should pass
```

## Reproducing results

Every experiment reads a YAML config and writes figures + numbers to
`results/<case_name>/`:

```bash
.venv/bin/python scripts/run_phase1.py --config configs/nominal.yaml
.venv/bin/python scripts/run_phase2.py     # all cases; --cases to subset
```

Phase 1 outputs: `fig1_orbit_timeline`, `fig2_baseline_policy` (PDF + PNG),
`audit.json`, `audit.md`. Phase 2 outputs per case: `fig3_optimal_solution`,
`refinement.{json,md}`, `comparison.json`, `solution_N2880.npz`, plus
`results/sweeps/` (DoD-floor sweep) and `results/checkpoint2.md`.

## Layout

```
src/matrix_ocp/   params.py (units, config), orbit.py (Kepler timeline),
                  baseline.py (FDR policy + audit), plotting.py
configs/          one YAML per case
scripts/          one script per experiment phase
tests/            pytest suite (orbit checks, balances, periodicity)
docs/             model.md and, later, direct_method.md / pmp_derivation.md
references/       FDR report and slides (lecture notes + proposal to be added)
results/          generated, gitignored; regenerate with the commands above
```

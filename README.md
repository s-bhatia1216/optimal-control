# MATRIX per-orbit operations as a state-constrained optimal control problem

MAE 546 / ECE 525 final project (Sonal Bhatia). Each orbit, a MATRIX
magnetotail spacecraft must split its time between collecting science,
transmitting to the ground, and managing its battery. The MAE 342 MATRIX
Final Design Report handles this with fixed mode rules (science off in
eclipse); this project poses it as a periodic optimal control problem with
battery and data-storage state constraints, solves it by direct collocation
(CasADi/IPOPT), and verifies the solution against the first-order conditions
of Pontryagin's Maximum Principle: switching functions, costate jumps at the
state-constraint junctions, singular arcs, and periodic transversality.

Physical parameters come from the FDR; modeling decisions and resolutions of
internal FDR inconsistencies are documented in `docs/model.md`. The FDR,
slides, and course notes live in a local `references/` directory that is not
tracked (course material and the team report are not redistributable).

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .      # numpy, scipy, casadi, matplotlib, pyyaml, pytest
```

## Tests

```bash
.venv/bin/python -m pytest      # 24 checks: orbit geometry, FDR cross-checks,
                                # balances, LQR multiplier convention, PMP spot checks
```

## Reproducing results

Every experiment reads a YAML config from `configs/` and writes figures and
numbers to `results/` (gitignored; everything below is regenerable).

```bash
# Phase 1: orbit timeline, FDR baseline policy, constraint audit
.venv/bin/python scripts/run_phase1.py --config configs/nominal.yaml

# Phase 2: direct collocation on all five cases, mesh refinement,
# lexicographic polish, DoD sweep, mesh-error study
.venv/bin/python scripts/run_phase2.py            # --cases to subset

# Phase 3: PMP consistency checks along the saved solutions
.venv/bin/python scripts/run_phase3.py

# Phase 4: sensitivity sweeps, paper figure set, summary
.venv/bin/python scripts/run_phase4.py            # --skip-sweeps to reuse saved sweeps
```

Outputs per case: `fig1_orbit_timeline`, `fig2_baseline_policy`,
`fig3_optimal_solution` (polished trajectory), `fig3_stage1_solution`
(analytic-center trajectory), `fig4_switching_functions` (PDF + PNG each),
`audit.{json,md}`, `refinement.{json,md}`, `comparison.json`,
`pmp_checks.{json,md}`, and the solution arrays
`solution_N2880{,_polished}.npz`. Repo-level outputs: `results/sweeps/`
(DoD sweep, mesh error, sensitivity), `results/paper/` (the six paper
figures plus a mesh-error candidate), `results/direct_summary.md`,
`results/pmp_summary.md`, and `results/SUMMARY.md` (key numbers and figure
paths).

## Layout

```
src/matrix_ocp/   params.py (units, config), orbit.py (Kepler timeline),
                  baseline.py (FDR policy + audit), direct.py (collocation,
                  duals, polish), pmp.py (PMP checks), plotting.py
configs/          one YAML per case (nominal, S1/S2 stress, uniform/apogee value)
scripts/          one script per experiment phase
tests/            pytest suite
docs/             model.md, direct_method.md, pmp_derivation.md
references/       local only, untracked (FDR report/slides, course notes)
results/          generated, gitignored; regenerate with the commands above
```

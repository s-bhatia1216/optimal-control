#!/usr/bin/env python
"""Phase 3: numerical PMP consistency checks along the direct solutions.

Usage:
    .venv/bin/python scripts/run_phase3.py            # all cases
    .venv/bin/python scripts/run_phase3.py --cases s1_uniform

Loads the finest-mesh stage-1 solution (duals) and the polished solution
(for the E_max boundary-arc check) saved by run_phase2.py, evaluates the
first-order conditions from docs/pmp_derivation.md, and writes per case:
    fig4_switching_functions.{pdf,png}
    pmp_checks.json / pmp_checks.md
plus results/checkpoint3.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from matrix_ocp.orbit import OrbitTimeline
from matrix_ocp.params import Params
from matrix_ocp.plotting import plot_switching
from matrix_ocp.pmp import run_all_checks, switching_functions

REPO = Path(__file__).resolve().parents[1]
CASES = ["nominal", "s1_uniform", "s1_apogee", "s2_uniform", "s2_apogee"]
N_FINEST = 2880


def _load(case: str):
    d = REPO / "results" / case
    sol = dict(np.load(d / f"solution_N{N_FINEST}.npz"))
    pol = dict(np.load(d / f"solution_N{N_FINEST}_polished.npz"))
    meta = json.loads((d / f"solution_N{N_FINEST}_meta.json").read_text())
    return sol, pol, meta


def run_case(case: str) -> dict:
    p = Params.from_yaml(REPO / "configs" / f"{case}.yaml")
    tl = OrbitTimeline(p)
    outdir = REPO / "results" / case
    sol, pol, meta = _load(case)

    checks = run_all_checks(p, sol, pol, meta)
    theta = switching_functions(p, sol)
    plot_switching(p, tl, sol, theta, outdir)

    (outdir / "pmp_checks.json").write_text(json.dumps(checks, indent=2))
    _write_md(case, p, checks, outdir / "pmp_checks.md")
    return checks


def _write_md(case: str, p: Params, c: dict, path: Path):
    bs, bc, bh = c["bang_s"], c["bang_c"], c["bang_h"]
    jE, jD = c["costate_jumps"]["E"], c["costate_jumps"]["D"]
    tE, tD = c["transversality"]["E"], c["transversality"]["D"]
    ba, ham = c["boundary_arc"], c["hamiltonian"]
    lines = [
        f"# PMP consistency checks, case `{case}` (stage-1 duals, N = {N_FINEST})",
        "",
        "## Bang-bang law vs switching functions (per interval)",
        "",
        "| control | decided | agree | singular | max |theta| on singular |"
        " worst bang violation |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in [("u_s", bs), ("u_c", bc), ("u_h", bh)]:
        ag = "n/a" if b["agreement"] is None else f"{100*b['agreement']:.2f}%"
        worst = max(b["worst_on_violation"], b["worst_off_violation"])
        lines.append(
            f"| {name} | {b['n_decided']}/{b['n_free']} | {ag} |"
            f" {b['n_singular']} | {b['max_abs_theta_singular']:.2e} |"
            f" {worst:.2e} |")
    lines += [
        "",
        "## Costate jumps vs state-bound multipliers",
        "",
        f"Fitted discrete identity: delta p_E = ({jE['sign_upper']:+.0f})*mu_Emax"
        f" + ({jE['sign_lower']:+.0f})*mu_Emin at each interior node;"
        f" max residual {jE['max_residual']:.2e}.",
        f"Same for p_D: signs ({jD['sign_upper']:+.0f}, {jD['sign_lower']:+.0f}),"
        f" max residual {jD['max_residual']:.2e}.",
        "",
        f"Junctions (|delta p_E| > 1e-3): {jE['n_junctions']}",
    ]
    for j in jE["junctions"]:
        lines.append(
            f"- t = {j['t_h']:.3f} h: delta p_E = {j['delta_p']:+.4f},"
            f" mu_Emax = {j['mu_upper']:+.4f}, mu_Emin = {j['mu_lower']:+.4f}")
    lines += [
        "",
        "## E_max boundary arc (checked on the polished trajectory)",
        "",
        (f"{ba['n_intervals']} sunlit intervals on E = E_max"
         f" ({ba['arc_hours']:.2f} h); interior-control law"
         f" u_h = (P_0 + P_s u_s + P_c u_c)/P_in holds with max residual"
         f" {ba['max_residual']:.2e}; fraction strictly interior"
         f" {100*ba['fraction_interior']:.1f}%.") if ba["n_intervals"]
        else "No E_max arc in this case.",
        "",
        "## Periodic transversality p(0) = p(T)",
        "",
        f"p_E: first {tE['p_first']:+.6f}, last {tE['p_last']:+.6f},"
        f" gap {tE['gap']:.2e}; periodicity dual matches to"
        f" {tE['lam_matches_p0']:.2e}.",
        f"p_D: first {tD['p_first']:+.6f}, last {tD['p_last']:+.6f},"
        f" gap {tD['gap']:.2e}; periodicity dual matches to"
        f" {tD['lam_matches_p0']:.2e}.",
        "",
        "## Hamiltonian continuity",
        "",
        f"Across the {ham['n_plain_nodes']} interior nodes that are neither"
        " window edges nor costate junctions, |delta H| after removing the"
        f" explicit time dependence is at most {ham['max_residual_plain']:.2e}"
        " Mbit/h.",
        "",
        "## E_min touch point",
        "",
        f"Best achievable dE/dt in eclipse is {c['eclipse_best_dEdt_W']:.1f} W"
        " < 0 (no sun, loads off), so no sustained E = E_min arc can exist"
        " there: the constraint can only be touched, with the costate jump"
        " listed above.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_checkpoint3(all_checks: dict):
    lines = [
        "# Checkpoint 3 summary: PMP checks along the direct solutions", "",
        "| case | u_s agree | u_c agree | singular u_s ints | jump resid p_E |"
        " boundary-arc resid | p(0)=p(T) gap (E, D) | H resid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case, c in all_checks.items():
        bs, bc = c["bang_s"], c["bang_c"]
        f = lambda b: "n/a" if b["agreement"] is None else f"{100*b['agreement']:.1f}%"
        ba = c["boundary_arc"]["max_residual"]
        lines.append(
            f"| {case} | {f(bs)} | {f(bc)} | {bs['n_singular']} |"
            f" {c['costate_jumps']['E']['max_residual']:.1e} |"
            f" {'n/a' if ba is None else f'{ba:.1e}'} |"
            f" {c['transversality']['E']['gap']:.1e},"
            f" {c['transversality']['D']['gap']:.1e} |"
            f" {c['hamiltonian']['max_residual_plain']:.1e} |")
    lines += ["", "Details per case in results/<case>/pmp_checks.md;"
              " derivation in docs/pmp_derivation.md."]
    (REPO / "results" / "checkpoint3.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=CASES)
    args = ap.parse_args()
    all_checks = {}
    for case in args.cases:
        print(f"[case] {case} ...")
        all_checks[case] = run_case(case)
        c = all_checks[case]
        print(f"   u_s decided {c['bang_s']['n_decided']}, singular"
              f" {c['bang_s']['n_singular']}, jump resid"
              f" {c['costate_jumps']['E']['max_residual']:.2e}")
    write_checkpoint3(all_checks)
    print("[ok] wrote results/checkpoint3.md")


if __name__ == "__main__":
    main()

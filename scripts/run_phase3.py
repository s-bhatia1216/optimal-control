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
plus results/pmp_summary.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from matrix_ocp.direct import switch_times
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
    checks["independent"] = independent_checks(p, tl, sol, case)
    theta = switching_functions(p, sol)
    plot_switching(p, tl, sol, theta, outdir)

    (outdir / "pmp_checks.json").write_text(json.dumps(checks, indent=2))
    _write_md(case, p, checks, outdir / "pmp_checks.md")
    return checks


def independent_checks(p: Params, tl: OrbitTimeline, sol: dict,
                       case: str) -> list[dict]:
    """Group (b): predictions derived by hand from the mission physics,
    compared against the solver output (docs/pmp_derivation.md, Section 7).
    Unlike the KKT-consistency checks, these could fail if the model or the
    solution were wrong."""
    h, us = sol["h_h"], sol["u_s"]
    roi, ecl = sol["roi"], sol["eclipse"]
    rows = []

    def add(name, pred, obs):
        rows.append({"check": name, "predicted": float(pred),
                     "observed": float(obs),
                     "rel_err": float(abs(obs - pred) / max(1e-12, abs(pred)))})

    def interior_switches():
        """u_s switch times away from every window edge (those crossings are
        bound changes, not PMP switches)."""
        edges = tl.window_edge_times()
        return [t for t in switch_times(sol["t_mid_h"], us * (roi > 0.5))
                if np.min(np.abs(edges - t)) > 0.05]

    sci_roi_h = float(np.sum(h * us * roi))
    if case == "nominal":
        add("full-ROI science [h]", tl.roi_duration_h, sci_roi_h)
    if case.startswith("s1"):
        ration = (p.dod_floor * p.capacity_Wh
                  - p.P0_eclipse * p.eclipse_duration_h) / p.P_s
        add("eclipse ration (energy budget) [h]", ration,
            float(np.sum(h * us * ecl)))
        pE_arc = float(np.mean(sol["p_E"][ecl > 0.999]))
        if case == "s1_uniform":
            add("p_E on constrained arc = r_s/P_s [Mbit/Wh]",
                p.r_s / p.P_s, pE_arc)
            dp = np.diff(sol["p_E"])
            big = np.flatnonzero(np.abs(dp) > 1e-3)
            for i in big:
                t = sol["t_edges_h"][1:-1][i]
                add(f"|p_E jump| at t = {t:.3f} h = r_s/P_s",
                    p.r_s / p.P_s, abs(dp[i]))
        else:
            thresh = p.P_s * pE_arc / p.r_s
            for t in interior_switches():
                add(f"w at switch t = {t:.3f} h = P_s p_E / r_s",
                    thresh, float(tl.w(np.array([t]))[0]))
    if case.startswith("s2"):
        cap_h = (p.r_c * p.contact_duration_h - p.r_hk * tl.T_h) / p.r_s
        add("science cap (downlink budget) [h]", cap_h, sci_roi_h)
        pD = float(np.median(sol["p_D"]))
        if case == "s2_uniform":
            add("p_D = -1 (unit marginal value) [-]", -1.0, pD)
        else:
            for t in interior_switches():
                add(f"w at switch t = {t:.3f} h = -p_D",
                    -pD, float(tl.w(np.array([t]))[0]))
    return rows


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


def write_pmp_summary(all_checks: dict):
    lines = [
        "# PMP checks along the direct solutions",
        "",
        "## (a) KKT-consistency checks (validate the multiplier extraction)",
        "",
        "These restate the transcribed LP's KKT conditions through the",
        "validated dual-to-costate mapping, so they hold at any converged",
        "solution regardless of the model (the covector mapping idea: the",
        "discrete multipliers are samples of the continuous costate, and the",
        "discrete stationarity conditions are samples of the continuous",
        "first-order conditions). Passing proves the extraction and the",
        "transcription are implemented correctly.",
        "",
        "| case | u_s agree | u_c agree | singular u_s ints | jump resid p_E |"
        " theta_h max (E_max arc) | p(0)=p(T) gap (E, D) | H resid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case, c in all_checks.items():
        bs, bc = c["bang_s"], c["bang_c"]
        f = lambda b: "n/a" if b["agreement"] is None else f"{100*b['agreement']:.1f}%"
        lines.append(
            f"| {case} | {f(bs)} | {f(bc)} | {bs['n_singular']} |"
            f" {c['costate_jumps']['E']['max_residual']:.1e} |"
            f" {c['bang_h']['max_abs_theta_singular']:.1e} |"
            f" {c['transversality']['E']['gap']:.1e},"
            f" {c['transversality']['D']['gap']:.1e} |"
            f" {c['hamiltonian']['max_residual_plain']:.1e} |")
    lines += [
        "",
        "## (b) Independent verification against hand-derived predictions",
        "",
        "These numbers come from the mission physics (energy and data",
        "budgets, marginal values), derived before looking at the solver",
        "output; agreement is evidence about the model and the solution, not",
        "the plumbing. The E_max boundary-arc control law"
        " u_h = (P_0 + P_s u_s + P_c u_c)/P_in belongs here too"
        " (per-case residuals in pmp_checks.md), as does the sign argument"
        " that no sustained E_min arc can exist in eclipse"
        " (best dE/dt = -59 W < 0).",
        "",
        "| case | check | predicted | observed | rel err |",
        "|---|---|---|---|---|",
    ]
    for case, c in all_checks.items():
        for r in c.get("independent", []):
            lines.append(
                f"| {case} | {r['check']} | {r['predicted']:.5f} |"
                f" {r['observed']:.5f} | {r['rel_err']:.1e} |")
    lines += ["", "Details per case in results/<case>/pmp_checks.md;"
              " derivation and the two-group rationale in"
              " docs/pmp_derivation.md Section 7."]
    (REPO / "results" / "pmp_summary.md").write_text("\n".join(lines) + "\n")


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
    write_pmp_summary(all_checks)
    print("[ok] wrote results/pmp_summary.md")


if __name__ == "__main__":
    main()

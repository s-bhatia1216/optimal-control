#!/usr/bin/env python
"""Phase 2: direct-collocation solves, mesh refinement, DoD sweep, comparisons.

Usage:
    .venv/bin/python scripts/run_phase2.py            # all cases + sweep
    .venv/bin/python scripts/run_phase2.py --cases nominal s1_uniform

Per case, writes to results/<case_name>/:
    fig3_optimal_solution.{pdf,png}   trajectories, controls, costates
    refinement.json / refinement.md   objective & arc times vs mesh size
    solution_N<L>.npz              finest-mesh arrays (reused in Phase 3)
    comparison.json                   optimal vs FDR baseline
Also writes results/sweeps/ (DoD-floor sweep) and results/checkpoint2.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from matrix_ocp.baseline import simulate_baseline
from matrix_ocp.direct import polish_solution, solve_ocp
from matrix_ocp.orbit import OrbitTimeline
from matrix_ocp.params import Params
from matrix_ocp.plotting import plot_dod_sweep, plot_mesh_error, plot_solution

REPO = Path(__file__).resolve().parents[1]
CASES = ["nominal", "s1_uniform", "s1_apogee", "s2_uniform", "s2_apogee"]
MESHES = [360, 720, 1440, 2880]

REFINE_KEYS = [
    "J_weighted_Mbit", "science_roi_Mbit", "science_eclipse_h",
    "science_offroi_h", "min_E_Wh", "downlinked_Mbit",
]


def run_case(case: str) -> dict:
    p = Params.from_yaml(REPO / "configs" / f"{case}.yaml")
    tl = OrbitTimeline(p)
    outdir = REPO / "results" / p.case_name
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = simulate_baseline(p, tl)

    rows, finest = [], None
    for N in MESHES:
        sol = solve_ocp(p, tl, N)
        assert sol["status"] == "ok", f"{case} N={N}: {sol.get('status')}"
        row = {"N": N, "n_intervals": sol["n_intervals"],
               "return_status": sol["return_status"]}
        row.update({k: sol[k] for k in REFINE_KEYS})
        # Switch times of ROI science (junk off-ROI science masked out).
        row["u_s_switch_times_h"] = [
            round(t, 4) for t in _roi_switches(sol)]
        row["Emax_arcs_h"] = sol["Emax_arcs_h"]
        row["Emin_active_t_h"] = [round(t, 3) for t in sol["Emin_active_t_h"]]
        rows.append(row)
        finest = sol

    # Persist the finest solution for Phase 3 (arrays only -> npz).
    np.savez(outdir / f"solution_N{MESHES[-1]}.npz",
             **{k: v for k, v in finest.items() if isinstance(v, np.ndarray)})
    (outdir / f"solution_N{MESHES[-1]}_meta.json").write_text(json.dumps(
        {k: v for k, v in finest.items()
         if isinstance(v, (int, float, str, list, bool))}, indent=2))

    # Second-stage lexicographic polish: same optimal value, flight-like
    # trajectory (E rides E_max, buffer empties). Used for plots only; the
    # stage-1 duals it carries remain the ones used in all PMP checks.
    polished = polish_solution(p, tl, finest)
    np.savez(outdir / f"solution_N{MESHES[-1]}_polished.npz",
             **{k: v for k, v in polished.items() if isinstance(v, np.ndarray)})
    plot_solution(p, tl, polished, baseline, outdir)
    # Stage-1 figure kept alongside: it shows the analytic-center smear on
    # singular/degenerate arcs (the honest LP picture the text discusses).
    plot_solution(p, tl, finest, baseline, outdir, name="fig3_stage1_solution")

    comparison = {
        "baseline_science_Mbit": baseline["science_Mbit"],
        "optimal_science_Mbit": finest["science_roi_Mbit"],
        "gain_percent": 100.0 * (finest["science_roi_Mbit"]
                                 - baseline["science_Mbit"])
        / baseline["science_Mbit"],
        "baseline_min_E_Wh": baseline["min_E_Wh"],
        "optimal_min_E_Wh": finest["min_E_Wh"],
        "E_min_Wh": p.E_min,
        "baseline_science_h": baseline["science_h"],
        "optimal_science_roi_h": finest["science_roi_h"],
        "optimal_eclipse_science_h": finest["science_eclipse_h"],
        "optimal_offroi_science_h": finest["science_offroi_h"],
        "downlink_capacity_Mbit": p.r_c * p.contact_duration_h,
        "baseline_downlinked_Mbit": baseline["downlinked_Mbit"],
        "optimal_downlinked_Mbit": finest["downlinked_Mbit"],
        "Emin_touch_times_h": finest["Emin_active_t_h"],
        "p_E_eclipse_mean": float(np.mean(finest["p_E"][finest["eclipse"] > 0.5])),
        "p_D_mean": float(np.mean(finest["p_D"])),
        "r_s_over_P_s": p.r_s / p.P_s,
        # Polish diagnostics: same science value, complementary slackness of
        # (stage-1 duals, polished primal) should hold to solver tolerance.
        "polished_J_gap_Mbit": polished["J_gap_vs_stage1_Mbit"],
        "polished_cs_violation": polished["cs_violation"],
        "polished_min_E_Wh": polished["min_E_Wh"],
        "polished_dumped_Wh": polished["dumped_Wh"],
    }
    (outdir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (outdir / "refinement.json").write_text(json.dumps(rows, indent=2))
    _write_refinement_md(case, rows, outdir / "refinement.md")
    return {"params": p, "rows": rows, "comparison": comparison}


def _roi_switches(sol: dict) -> list[float]:
    """Switch times of u_s restricted to the ROI (masks degenerate off-ROI
    science, which has zero weight and shows up as smeared interior values)."""
    from matrix_ocp.direct import switch_times
    return switch_times(sol["t_mid_h"], sol["u_s"] * (sol["roi"] > 0.5))


def _write_refinement_md(case: str, rows: list[dict], path: Path):
    lines = [
        f"# Mesh refinement, case `{case}`",
        "",
        "| N | J_w [Mbit] | sci ROI [Mbit] | ecl sci [h] | off-ROI [h] | min E [Wh] | u_s switches (ROI) [h] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sw = ", ".join(f"{t:.3f}" for t in r["u_s_switch_times_h"])
        lines.append(
            f"| {r['N']} | {r['J_weighted_Mbit']:.4f} | {r['science_roi_Mbit']:.4f} |"
            f" {r['science_eclipse_h']:.4f} | {r['science_offroi_h']:.3f} |"
            f" {r['min_E_Wh']:.3f} | {sw} |")
    path.write_text("\n".join(lines) + "\n")


def run_mesh_comparison() -> dict:
    """Snapped vs midpoint-sampled transcription error vs mesh size.

    Error metric per case is the distance of a hand-derivable quantity from
    its analytic value: the S1 eclipse ration (energy budget) and the S2
    total science time (downlink capacity). Snapped meshes hit these exactly
    at every N; midpoint sampling converges at O(h)."""
    specs = {}
    for case in ["s1_uniform", "s2_uniform"]:
        p = Params.from_yaml(REPO / "configs" / f"{case}.yaml")
        tl = OrbitTimeline(p)
        if case == "s1_uniform":
            exact = (p.dod_floor * p.capacity_Wh
                     - p.P0_eclipse * p.eclipse_duration_h) / p.P_s
            key, label = "science_eclipse_h", "S1 eclipse science ration"
        else:
            exact = (p.r_c * p.contact_duration_h - p.r_hk * tl.T_h) / p.r_s
            key, label = "science_roi_h", "S2 total science time"
        rows = []
        for N in [90, 180, 360, 720, 1440, 2880]:
            row = {"N": N}
            for mesh in ("snapped", "midpoint"):
                sol = solve_ocp(p, tl, N, max_iter=600, mesh=mesh)
                row[mesh] = sol[key] if sol["status"] == "ok" else None
            rows.append(row)
        specs[case] = {"exact_h": exact, "label": label, "rows": rows}
    out = {"cases": specs}
    outdir = REPO / "results" / "sweeps"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mesh_error.json").write_text(json.dumps(out, indent=2))
    plot_mesh_error(out, outdir)
    return out


def run_dod_sweep(N: int = 720) -> dict:
    """Solve the OCP across (capacity, DoD floor) to map the three regimes."""
    base = Params.from_yaml(REPO / "configs" / "nominal.yaml")
    capacities = {"BOL": 829.44, "EOL": 734.4}
    floors = [round(f, 3) for f in np.arange(0.28, 0.445, 0.01)]
    results = {}
    for cap_name, cap in capacities.items():
        rows = []
        for f in floors:
            p = base.replace(capacity_Wh=cap, dod_floor=f,
                             case_name=f"sweep_{cap_name}_{f:.3f}")
            tl = OrbitTimeline(p)
            sol = solve_ocp(p, tl, N, max_iter=600)
            row = {"dod_floor": f, "status": sol["status"]}
            if sol["status"] == "ok":
                row.update({k: sol[k] for k in
                            ["science_eclipse_h", "science_roi_Mbit", "min_E_Wh"]})
                row["return_status"] = sol["return_status"]
            rows.append(row)
        results[cap_name] = rows
    sweep = {
        "N": N,
        "capacities_Wh": capacities,
        "need0_Wh": base.P0_eclipse * base.eclipse_duration_h,
        "P_s_W": base.P_s,
        "eclipse_h": base.eclipse_duration_h,
        "results": results,
    }
    outdir = REPO / "results" / "sweeps"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "dod_sweep.json").write_text(json.dumps(sweep, indent=2))
    plot_dod_sweep(sweep, outdir)
    return sweep


def write_checkpoint2(case_results: dict, sweep: dict | None):
    lines = ["# Checkpoint 2 summary", ""]
    lines += [
        "| case | baseline sci [Mbit] | optimal sci [Mbit] | gain | ecl sci [h] |"
        " min E [Wh] (E_min) | downlink used/cap [Mbit] | mean p_E in eclipse | mean p_D |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for case, res in case_results.items():
        c = res["comparison"]
        lines.append(
            f"| {case} | {c['baseline_science_Mbit']:.1f} |"
            f" {c['optimal_science_Mbit']:.1f} | {c['gain_percent']:+.1f}% |"
            f" {c['optimal_eclipse_science_h']:.3f} |"
            f" {c['optimal_min_E_Wh']:.1f} ({c['E_min_Wh']:.1f}) |"
            f" {c['optimal_downlinked_Mbit']:.1f}/{c['downlink_capacity_Mbit']:.1f} |"
            f" {c['p_E_eclipse_mean']:+.3f} | {c['p_D_mean']:+.3f} |")
    if sweep is not None:
        lines += ["", "DoD sweep: see results/sweeps/fig5_dod_sweep and dod_sweep.json."]
    (REPO / "results" / "checkpoint2.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=CASES)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    case_results = {}
    for case in args.cases:
        print(f"[case] {case} ...")
        case_results[case] = run_case(case)
        c = case_results[case]["comparison"]
        print(f"   optimal {c['optimal_science_Mbit']:.1f} Mbit "
              f"({c['gain_percent']:+.1f}% vs baseline), "
              f"ecl sci {c['optimal_eclipse_science_h']:.3f} h, "
              f"min E {c['optimal_min_E_Wh']:.1f} Wh")
    sweep = None
    if not args.skip_sweep:
        print("[sweep] DoD floor sweep ...")
        sweep = run_dod_sweep()
        print("[sweep] snapped vs midpoint mesh comparison ...")
        run_mesh_comparison()
    write_checkpoint2(case_results, sweep)
    print(f"[ok] wrote results/checkpoint2.md")


if __name__ == "__main__":
    main()

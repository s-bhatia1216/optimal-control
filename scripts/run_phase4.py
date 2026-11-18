#!/usr/bin/env python
"""Phase 4: sensitivity sweeps, the paper figure set, and results/SUMMARY.md.

Usage:
    .venv/bin/python scripts/run_phase4.py              # everything
    .venv/bin/python scripts/run_phase4.py --skip-sweeps

Writes:
    results/sweeps/sensitivity.json + fig7_sensitivity.{pdf,png}
    results/paper/  (the six paper figures + one candidate, copied/rendered)
    results/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from matrix_ocp.baseline import simulate_baseline
from matrix_ocp.direct import solve_ocp
from matrix_ocp.orbit import OrbitTimeline
from matrix_ocp.params import Params
from matrix_ocp.plotting import (plot_costate_jumps, plot_s1_solution_paper,
                                 plot_sensitivity)

REPO = Path(__file__).resolve().parents[1]
N_SWEEP = 720
N_FINEST = 2880


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------

def _solve_point(p: Params, tl: OrbitTimeline):
    sol = solve_ocp(p, tl, N_SWEEP, max_iter=600)
    if sol["status"] != "ok":
        return "infeasible", None
    return "ok", float(sol["science_roi_Mbit"])


def run_sensitivity() -> dict:
    """Four one-parameter sweeps about the nominal case. Each panel crosses
    the regime boundary where its constraint starts to bind, and overlays
    the analytic budget prediction (docs/pmp_derivation.md, Section 7b)."""
    base = Params.from_yaml(REPO / "configs" / "nominal.yaml")
    tl0 = OrbitTimeline(base)
    roi = tl0.roi_duration_h
    T = tl0.T_h
    sweeps = []

    def sci_battery(cap, dod, T_ecl):
        """Science [Mbit] under the battery budget (nan if infeasible)."""
        usable = dod * cap - base.P0_eclipse * T_ecl
        ecl_sci = np.clip(usable / base.P_s, 0.0, T_ecl)
        out = base.r_s * (roi - T_ecl + ecl_sci)
        return np.where(usable < 0.0, np.nan, out)

    def sci_downlink(r_c, t_c):
        """Science [Mbit] under the downlink budget (nan if infeasible)."""
        cap_h = (r_c * t_c - base.r_hk * T) / base.r_s
        return np.where(cap_h < 0.0, np.nan,
                        base.r_s * np.clip(cap_h, 0.0, roi))

    # (a) battery capacity at the EOL-representative DoD floor 0.36.
    xs = list(np.round(np.arange(650.0, 900.1, 12.5), 2))
    rows = {"xs": xs, "status": [], "sci": []}
    for cap in xs:
        p = base.replace(capacity_Wh=cap, dod_floor=0.36,
                         case_name=f"sens_cap_{cap:.0f}")
        s, v = _solve_point(p, OrbitTimeline(p))
        rows["status"].append(s)
        rows["sci"].append(v)
    xf = np.linspace(xs[0], xs[-1], 400)
    sweeps.append({**rows, "param": "capacity_Wh",
                   "title": "battery capacity (DoD floor 0.36)",
                   "label": "capacity [Wh]",
                   "x_fine": list(xf),
                   "pred": list(sci_battery(xf, 0.36,
                                            base.eclipse_duration_h)),
                   "marks": [{"x": 734.4, "label": "EOL"},
                             {"x": 829.44, "label": "BOL"}]})

    # (b) eclipse duration at EOL capacity and DoD floor 0.36.
    xs = list(np.round(np.arange(2.0, 5.01, 0.2), 2))
    rows = {"xs": xs, "status": [], "sci": []}
    for T_ecl in xs:
        p = base.replace(eclipse_duration_h=T_ecl, capacity_Wh=734.4,
                         dod_floor=0.36, case_name=f"sens_ecl_{T_ecl:.2f}")
        s, v = _solve_point(p, OrbitTimeline(p))
        rows["status"].append(s)
        rows["sci"].append(v)
    xf = np.linspace(xs[0], xs[-1], 400)
    sweeps.append({**rows, "param": "eclipse_duration_h",
                   "title": "eclipse duration (EOL, DoD 0.36)",
                   "label": "eclipse duration [h]",
                   "x_fine": list(xf),
                   "pred": list(sci_battery(734.4, 0.36, xf)),
                   "marks": [{"x": base.eclipse_duration_h,
                              "label": "FDR worst case"}]})

    # (c) downlink rate at the nominal 1 h contact.
    xs = list(np.round(np.arange(150.0, 1000.1, 50.0), 1))
    rows = {"xs": xs, "status": [], "sci": []}
    for rc in xs:
        p = base.replace(r_c=rc, case_name=f"sens_rc_{rc:.0f}")
        s, v = _solve_point(p, OrbitTimeline(p))
        rows["status"].append(s)
        rows["sci"].append(v)
    xf = np.linspace(xs[0], xs[-1], 400)
    sweeps.append({**rows, "param": "r_c",
                   "title": "downlink rate (1 h contact)",
                   "label": "downlink rate [Mbit/h]",
                   "x_fine": list(xf),
                   "pred": list(sci_downlink(xf, base.contact_duration_h)),
                   "marks": [{"x": base.r_c, "label": "FDR link budget"}]})

    # (d) contact duration at the nominal downlink rate.
    xs = list(np.round(np.linspace(0.15, 1.5, 19), 4))
    rows = {"xs": xs, "status": [], "sci": []}
    for tc in xs:
        p = base.replace(contact_duration_h=tc,
                         case_name=f"sens_tc_{tc:.2f}")
        s, v = _solve_point(p, OrbitTimeline(p))
        rows["status"].append(s)
        rows["sci"].append(v)
    xf = np.linspace(xs[0], xs[-1], 400)
    sweeps.append({**rows, "param": "contact_duration_h",
                   "title": "contact duration (FDR rate)",
                   "label": "contact duration [h]",
                   "x_fine": list(xf),
                   "pred": list(sci_downlink(base.r_c, xf)),
                   "marks": [{"x": 1.0, "label": "FDR"},
                             {"x": 0.5, "label": "S2"}]})

    def _clean(v):
        return [None if (isinstance(x, float) and np.isnan(x)) else x
                for x in v]
    for s in sweeps:
        s["pred"] = _clean([float(x) for x in s["pred"]])
    data = {"N": N_SWEEP, "sweeps": sweeps}
    outdir = REPO / "results" / "sweeps"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sensitivity.json").write_text(json.dumps(data, indent=2))
    return data


def _plot_sensitivity(data):
    # json round trip turned nan into None; plotting handles None already.
    for s in data["sweeps"]:
        s["pred"] = [np.nan if v is None else v for v in s["pred"]]
    plot_sensitivity(data, REPO / "results" / "sweeps")


# ---------------------------------------------------------------------------
# Paper figure set
# ---------------------------------------------------------------------------

def build_paper_set():
    paper = REPO / "results" / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    def copy(src_stem: str, dst_stem: str):
        for ext in ("pdf", "png"):
            shutil.copy2(REPO / "results" / f"{src_stem}.{ext}",
                         paper / f"{dst_stem}.{ext}")

    # 1. Orbit timeline (as generated in Phase 1).
    copy("nominal/fig1_orbit_timeline", "fig1_timeline")

    # 2. S1 solution, polished next to the stage-1 analytic center.
    p = Params.from_yaml(REPO / "configs" / "s1_uniform.yaml")
    tl = OrbitTimeline(p)
    d = REPO / "results" / "s1_uniform"
    sol1 = dict(np.load(d / f"solution_N{N_FINEST}.npz"))
    pol = dict(np.load(d / f"solution_N{N_FINEST}_polished.npz"))
    baseline = simulate_baseline(p, tl)
    plot_s1_solution_paper(p, tl, pol, sol1, baseline, paper)

    # 3. Switching functions (s1_apogee: zero crossings set the block).
    copy("s1_apogee/fig4_switching_functions", "fig3_switching")

    # 4. Costate jumps vs bound multipliers (s1_uniform).
    plot_costate_jumps(p, tl, pol, sol1, paper)

    # 5. DoD-floor sweep (three regimes).
    copy("sweeps/fig5_dod_sweep", "fig5_dod_sweep")

    # 6. Combined sensitivity figure.
    copy("sweeps/fig7_sensitivity", "fig6_sensitivity")

    # Candidate: snapped vs midpoint mesh error.
    copy("sweeps/fig6_mesh_error", "candidate_mesh_error")


# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------

def write_summary():
    cases = ["nominal", "s1_uniform", "s1_apogee", "s2_uniform", "s2_apogee"]
    comp = {c: json.loads((REPO / "results" / c / "comparison.json")
                          .read_text()) for c in cases}
    base = Params.from_yaml(REPO / "configs" / "nominal.yaml")
    tl = OrbitTimeline(base)
    lines = [
        "# Results summary (key numbers and figure paths)",
        "",
        "MATRIX per-orbit operations as a state-constrained periodic optimal",
        "control problem: direct collocation (CasADi/IPOPT, window-snapped",
        "trapezoidal mesh, exact for this model) compared against",
        "Pontryagin's Maximum Principle via multiplier-based costates.",
        "",
        "## Orbit and model",
        "",
        f"- Period {tl.T_h:.3f} h, e = {tl.e:.4f}; ROI (r > 6 R_E)"
        f" [{tl.t_roi_entry_h:.3f}, {tl.t_roi_exit_h:.3f}] h"
        f" = {tl.roi_duration_h:.3f} h ({100*tl.roi_duration_h/tl.T_h:.2f}%)",
        f"- Eclipse {base.eclipse_duration_h:.3f} h centered on apogee;"
        " contact 1 h centered on perigee (wraps)",
        f"- Loads: P_in {base.P_in} W, P0 day/eclipse {base.P0_day}/"
        f"{base.P0_eclipse} W, P_s {base.P_s} W, P_c {base.P_c} W;"
        f" rates r_s {base.r_s:.3f}, r_hk {base.r_hk:.3f},"
        f" r_c {base.r_c:.1f} Mbit/h",
        "",
        "## Headline results (per orbit, finest mesh N = 2880)",
        "",
        "| case | binding constraint | baseline sci [Mbit] |"
        " optimal sci [Mbit] | gain | eclipse sci [h] |",
        "|---|---|---|---|---|---|",
    ]
    binding = {
        "nominal": "E_max only (dump)",
        "s1_uniform": "E_min (DoD 0.36, EOL)",
        "s1_apogee": "E_min (apogee-weighted w)",
        "s2_uniform": "downlink (0.5 h contact)",
        "s2_apogee": "downlink (apogee-weighted w)",
    }
    for c in cases:
        k = comp[c]
        lines.append(
            f"| {c} | {binding[c]} | {k['baseline_science_Mbit']:.1f} |"
            f" {k['optimal_science_Mbit']:.1f} | {k['gain_percent']:+.1f}% |"
            f" {k['optimal_eclipse_science_h']:.3f} |")
    lines += [
        "",
        "Baseline infeasibility without a dump control: the FDR policy must",
        "shed 2198.3 Wh/orbit (77.9% of available solar); see",
        "results/nominal/audit.md.",
        "",
        "## Battery (S1) framing",
        "",
        "- DoD-floor regimes (results/sweeps/fig5_dod_sweep): infeasible"
        " below 0.310 (BOL) / 0.350 (EOL); E_min binds on [0.310, 0.329]"
        " (BOL) / [0.350, 0.372] (EOL); slack above.",
        "- S1 representative point: EOL capacity 734.4 Wh, floor 0.36,"
        " eclipse ration 1.995 h of 4.356 h (energy budget, exact).",
        "",
        "## PMP verification highlights",
        "",
        "- (a) KKT consistency (validates multiplier extraction): bang-bang"
        " agreement 100% everywhere; costate jump identity"
        " delta p = mu_upper - mu_lower to 1e-13; p(0) = p(T) to 1e-15;"
        " theta_h = 0 identically (energy free in sunlight).",
        "- (b) Hand-derived predictions: p_E = r_s/P_s = 6.1317 Mbit/Wh on"
        " the S1 constrained arc, both junction jumps equal to the bound"
        " multiplier point masses (eclipse entry +, E_min touch -);"
        " switch thresholds w = P_s p_E/r_s = 0.9954 (S1 apogee) and"
        " w = -p_D = 0.683 at r = 8.26 R_E (S2 apogee); no sustained E_min"
        " arc possible in eclipse (best dE/dt = -59 W).",
        "- Full tables: results/checkpoint3.md, results/<case>/pmp_checks.md.",
        "",
        "## Sensitivity (results/sweeps/fig7_sensitivity)",
        "",
        "- Battery capacity (DoD 0.36): infeasible below 713.9 Wh; E_min"
        " binds on [713.9, 758.6] Wh; EOL (734.4) sits inside the binding"
        " band, BOL (829.4) is slack.",
        "- Eclipse duration (EOL, DoD 0.36): infeasible beyond 4.48 h; the"
        " FDR worst case 4.356 h is 0.13 h from infeasibility.",
        "- Downlink rate (1 h contact): full-ROI science needs"
        " r_c >= 534.2 Mbit/h; FDR 912.2 has 1.7x margin.",
        "- Contact duration (FDR rate): full-ROI science needs 0.586 h;"
        " the 1 h window has 1.7x margin, S2's 0.5 h is below it.",
        "",
        "## Paper figure set (results/paper/)",
        "",
        "| figure | file | source |",
        "|---|---|---|",
        "| 1 orbit timeline | fig1_timeline | Phase 1 |",
        "| 2 S1 solution, polished vs analytic center | fig2_s1_solution |"
        " Phase 4 render |",
        "| 3 switching functions (s1_apogee) | fig3_switching | Phase 3 |",
        "| 4 costate jumps vs bound multipliers | fig4_costate_jumps |"
        " Phase 4 render |",
        "| 5 DoD-floor sweep | fig5_dod_sweep | Phase 2 |",
        "| 6 combined sensitivity | fig6_sensitivity | Phase 4 |",
        "| candidate: mesh error | candidate_mesh_error | Phase 2 |",
        "",
        "## Reproduce",
        "",
        "    .venv/bin/python -m pytest",
        "    .venv/bin/python scripts/run_phase1.py --config configs/nominal.yaml",
        "    .venv/bin/python scripts/run_phase2.py",
        "    .venv/bin/python scripts/run_phase3.py",
        "    .venv/bin/python scripts/run_phase4.py",
        "",
        "Docs: docs/model.md (parameter decisions), docs/direct_method.md",
        "(transcription, duals, degeneracy, polish), docs/pmp_derivation.md",
        "(derivation, normality, the two-group check rationale).",
    ]
    (REPO / "results" / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweeps", action="store_true")
    args = ap.parse_args()
    if args.skip_sweeps:
        data = json.loads(
            (REPO / "results" / "sweeps" / "sensitivity.json").read_text())
    else:
        print("[sweeps] sensitivity ...")
        data = run_sensitivity()
    _plot_sensitivity(data)
    print("[paper] building figure set ...")
    build_paper_set()
    write_summary()
    print("[ok] wrote results/SUMMARY.md and results/paper/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 1: orbit timeline, FDR baseline policy, and the constraint audit.

Usage:
    .venv/bin/python scripts/run_phase1.py --config configs/nominal.yaml

Writes to results/<case_name>/:
    fig1_orbit_timeline.{pdf,png}
    fig2_baseline_policy.{pdf,png}
    audit.json       every number computed below
    audit.md         human-readable audit report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matrix_ocp.baseline import (
    audit,
    battery_binding_interval,
    eclipse_energy_need,
    simulate_baseline,
)
from matrix_ocp.orbit import OrbitTimeline
from matrix_ocp.params import Params
from matrix_ocp.plotting import plot_baseline, plot_timeline

REPO = Path(__file__).resolve().parents[1]


def stress_case_quantification(p: Params, tl: OrbitTimeline, traj: dict) -> dict:
    """Numbers behind the proposed stress cases (battery side and data side)."""
    out: dict = {}

    # ---- Battery side: capacity x DoD-floor grid --------------------------
    # The battery enters the eclipse full (huge sunlit surplus), so whether
    # E_min binds is decided by usable energy vs the eclipse draw.
    capacities = {"BOL_829.44Wh": 829.44, "EOL_734.4Wh": 734.4}
    floors = [0.60, 0.30]
    grid = []
    need0 = eclipse_energy_need(p, 0.0)
    need1 = eclipse_energy_need(p, 1.0)
    for cap_name, cap in capacities.items():
        for f in floors:
            usable = f * cap
            if usable < need0:
                verdict = "INFEASIBLE (cannot survive eclipse even with science off)"
            elif usable < need1:
                verdict = "BINDING (E_min rations eclipse science)"
            else:
                verdict = "slack"
            grid.append({
                "capacity": cap_name,
                "dod_floor": f,
                "usable_Wh": usable,
                "eclipse_need_no_science_Wh": need0,
                "eclipse_need_full_science_Wh": need1,
                "margin_full_science_Wh": usable - need1,
                "verdict": verdict,
            })
    out["battery_grid"] = grid
    out["binding_dod_interval_BOL"] = battery_binding_interval(p, 829.44)
    out["binding_dod_interval_EOL"] = battery_binding_interval(p, 734.4)

    # Proposed S1: EOL capacity, DoD floor 0.36 (inside the EOL binding band).
    cap_eol = 734.4
    f_s1 = 0.36
    usable_s1 = f_s1 * cap_eol
    science_budget_Wh = usable_s1 - need0
    out["S1_proposal"] = {
        "capacity_Wh": cap_eol,
        "dod_floor": f_s1,
        "usable_Wh": usable_s1,
        "eclipse_science_energy_budget_Wh": science_budget_Wh,
        "eclipse_science_hours_affordable": science_budget_Wh / p.P_s,
        "eclipse_duration_h": p.eclipse_duration_h,
    }

    # ---- Data side: contact-length sweep ----------------------------------
    # Periodicity requires (science + housekeeping) generated per orbit to fit
    # through the contact window: r_s*T_sci + r_hk*T <= r_c * T_contact.
    roi_h = tl.roi_duration_h
    hk_Mbit = p.r_hk * tl.T_h
    sweep = []
    for tc in [1.0, 0.75, 0.5]:
        cap_Mbit = p.r_c * tc
        max_sci_h = min(roi_h, (cap_Mbit - hk_Mbit) / p.r_s)
        sweep.append({
            "contact_h": tc,
            "downlink_capacity_Mbit": cap_Mbit,
            "generation_full_roi_science_Mbit": p.r_s * roi_h + hk_Mbit,
            "max_science_h_supported": max_sci_h,
            "binding": bool(max_sci_h < roi_h - 1e-9),
            "baseline_science_h": traj["science_h"],
        })
    out["contact_sweep"] = sweep

    # Optional data stress: FGM burst mode held through the ROI (3072 bps vs 6).
    r_s_burst = p.r_s + (3072.0 - 6.0) * 3600.0 / 1e6  # Mbit/h, FDR Table 27
    out["burst_mode_variant"] = {
        "r_s_burst_Mbit_h": r_s_burst,
        "generation_full_roi_Mbit": r_s_burst * roi_h + hk_Mbit,
        "downlink_capacity_1h_Mbit": p.r_c * 1.0,
        "max_science_h_at_1h_contact": min(roi_h, (p.r_c * 1.0 - hk_Mbit) / r_s_burst),
    }
    return out


def upper_bound_comparison(p: Params, tl: OrbitTimeline, traj: dict) -> dict:
    """Baseline vs the simple candidate optimum (science through all of ROI).

    At nominal parameters all constraints are slack except E_max (handled by the
    dump), so science everywhere in the ROI is feasible; this is the number the
    Phase-2 optimizer should reproduce.
    """
    roi_h = tl.roi_duration_h
    full_roi_Mbit = p.r_s * roi_h
    base_Mbit = traj["science_Mbit"]
    return {
        "baseline_science_h": traj["science_h"],
        "baseline_science_Mbit": base_Mbit,
        "full_roi_science_h": roi_h,
        "full_roi_science_Mbit": full_roi_Mbit,
        "gain_Mbit": full_roi_Mbit - base_Mbit,
        "gain_percent": 100.0 * (full_roi_Mbit - base_Mbit) / base_Mbit,
        "extra_battery_cost_Wh": p.P_s * p.eclipse_duration_h,
    }


def write_markdown(report: dict, path: Path):
    """Render the audit dict as a readable summary."""
    a = report["audit"]
    nd = a["no_dump_feasibility"]
    ca = a["constraint_activity_baseline"]
    ub = report["upper_bound_comparison"]
    sq = report["stress_quantification"]

    lines = [
        f"# Phase 1 audit: case `{a['case_name']}`",
        "",
        "## Timeline",
        "",
        f"- Period T = {a['timeline']['period_h']:.4f} h, e = {a['timeline']['eccentricity']:.4f}",
        f"- ROI (r > 6 R_E): [{a['timeline']['roi_entry_h']:.3f}, {a['timeline']['roi_exit_h']:.3f}] h"
        f" -> {a['timeline']['roi_duration_h']:.3f} h ({100*a['timeline']['roi_fraction']:.2f}% of orbit;"
        " FDR Table 46 says 79.73%)",
        f"- Eclipse: [{a['timeline']['eclipse_window_h'][0]:.3f}, {a['timeline']['eclipse_window_h'][1]:.3f}] h"
        " (inside the ROI)",
        f"- Contact: {a['timeline']['contact_duration_h']:.2f} h centered on perigee",
        "",
        "## (1) Feasibility without the dump control u_h",
        "",
        f"- Solar available per orbit: {nd['solar_available_Wh_per_orbit']:.1f} Wh;"
        f" total load per orbit: {nd['total_load_Wh_per_orbit']:.1f} Wh",
        f"- Surplus that must be dumped: {nd['surplus_Wh_per_orbit']:.1f} Wh/orbit"
        f" ({100*nd['surplus_fraction_of_available']:.1f}% of available solar;"
        f" mean u_h needed = {nd['mean_u_h_required']:.3f})",
        f"- At full harvest the usable battery band fills in"
        f" {nd['hours_to_fill_usable_band_at_full_harvest']:.2f} h of the"
        " ~19.6 h sunlit arc -> **infeasible without u_h**",
        "",
        "## (2) Active constraints over the baseline periodic orbit",
        "",
        f"- E = E_max active {100*ca['E_max_active_fraction_of_orbit']:.1f}% of the orbit (dump arc)",
        f"- min E = {ca['min_E_Wh']:.1f} Wh vs E_min = {ca['E_min_Wh']:.1f} Wh ->"
        f" margin {ca['E_min_margin_Wh']:.1f} Wh (E_min {'ACTIVE' if ca['E_min_active'] else 'never active'})",
        f"- max D = {ca['max_D_Mbit']:.1f} Mbit vs D_max = {ca['D_max_Mbit']:.0f} Mbit"
        f" ({100*ca['max_D_Mbit']/ca['D_max_Mbit']:.1f}% of SSR; D_max"
        f" {'ACTIVE' if ca['D_max_active'] else 'never active'})",
        f"- Downlink: generated {ca['data_generated_Mbit']:.1f} Mbit vs capacity"
        f" {ca['downlink_capacity_Mbit']:.1f} Mbit ({100*ca['downlink_utilization']:.1f}% used;"
        f" {'ACTIVE' if ca['downlink_active'] else 'not binding'})",
        "",
        "## (3) Science-battery coupling",
        "",
        f"- Science payload adds {a['science_battery_coupling']['P_s_W']:.1f} W against a"
        f" {a['science_battery_coupling']['P0_eclipse_W']:.1f} W eclipse bus load",
        f"- Full-eclipse science costs {a['science_battery_coupling']['full_eclipse_science_cost_Wh']:.1f} Wh;"
        f" eclipse draw is {a['science_battery_coupling']['eclipse_need_no_science_Wh']:.1f} Wh (science off)"
        f" to {a['science_battery_coupling']['eclipse_need_full_science_Wh']:.1f} Wh (science on)",
        "",
        "| capacity | DoD floor | usable [Wh] | margin w/ full eclipse science [Wh] | verdict |",
        "|---|---|---|---|---|",
    ]
    for g in sq["battery_grid"]:
        lines.append(
            f"| {g['capacity']} | {g['dod_floor']:.2f} | {g['usable_Wh']:.1f} |"
            f" {g['margin_full_science_Wh']:.1f} | {g['verdict']} |"
        )
    b_bol = sq["binding_dod_interval_BOL"]["dod_floor_binding_interval"]
    b_eol = sq["binding_dod_interval_EOL"]["dod_floor_binding_interval"]
    s1 = sq["S1_proposal"]
    lines += [
        "",
        f"- E_min binds only for DoD floors in [{b_bol[0]:.3f}, {b_bol[1]:.3f}] at BOL"
        f" and [{b_eol[0]:.3f}, {b_eol[1]:.3f}] at EOL; below those -> infeasible.",
        f"- Proposed S1 (EOL {s1['capacity_Wh']:.1f} Wh, floor {s1['dod_floor']:.2f}):"
        f" eclipse science budget {s1['eclipse_science_energy_budget_Wh']:.1f} Wh ->"
        f" {s1['eclipse_science_hours_affordable']:.2f} h of the {s1['eclipse_duration_h']:.2f} h"
        " eclipse affordable: the optimizer must ration.",
        "",
        "## (4) Baseline vs simple upper bound (science through all of ROI)",
        "",
        f"- Baseline science: {ub['baseline_science_h']:.2f} h = {ub['baseline_science_Mbit']:.1f} Mbit/orbit",
        f"- Science through all of ROI: {ub['full_roi_science_h']:.2f} h ="
        f" {ub['full_roi_science_Mbit']:.1f} Mbit/orbit"
        f" (+{ub['gain_percent']:.1f}%), at an extra battery cost of only"
        f" {ub['extra_battery_cost_Wh']:.1f} Wh",
        "",
        "## (5) Contact-length sweep (data-side stress)",
        "",
        "| contact [h] | capacity [Mbit] | generation @ full ROI science [Mbit] | max science [h] | binding? |",
        "|---|---|---|---|---|",
    ]
    for s in sq["contact_sweep"]:
        lines.append(
            f"| {s['contact_h']:.2f} | {s['downlink_capacity_Mbit']:.1f} |"
            f" {s['generation_full_roi_science_Mbit']:.1f} | {s['max_science_h_supported']:.2f} |"
            f" {'YES' if s['binding'] else 'no'} |"
        )
    bm = sq["burst_mode_variant"]
    lines += [
        "",
        f"- Burst-FGM variant (3072 bps through ROI): generation"
        f" {bm['generation_full_roi_Mbit']:.1f} Mbit vs 1 h capacity"
        f" {bm['downlink_capacity_1h_Mbit']:.1f} Mbit -> max science"
        f" {bm['max_science_h_at_1h_contact']:.2f} h.",
        "",
        "## Baseline simulation checks",
        "",
        f"- Periodicity error of converged cycle: dE = {report['trajectory_tallies']['periodicity_error_E_Wh']:.2e} Wh,"
        f" dD = {report['trajectory_tallies']['periodicity_error_D_Mbit']:.2e} Mbit",
        f"- Energy dumped: {report['trajectory_tallies']['dumped_Wh']:.1f} Wh/orbit;"
        f" downlinked: {report['trajectory_tallies']['downlinked_Mbit']:.1f} Mbit/orbit",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "nominal.yaml"))
    args = ap.parse_args()

    p = Params.from_yaml(args.config)
    tl = OrbitTimeline(p)
    outdir = REPO / "results" / p.case_name
    outdir.mkdir(parents=True, exist_ok=True)

    traj = simulate_baseline(p, tl)
    plot_timeline(p, tl, outdir)
    plot_baseline(p, tl, traj, outdir)

    report = {
        "audit": audit(p, tl, traj),
        "upper_bound_comparison": upper_bound_comparison(p, tl, traj),
        "stress_quantification": stress_case_quantification(p, tl, traj),
        "trajectory_tallies": {
            k: traj[k] for k in [
                "science_h", "science_Mbit", "housekeeping_Mbit",
                "downlinked_Mbit", "solar_available_Wh", "solar_harvested_Wh",
                "dumped_Wh", "load_Wh", "min_E_Wh", "max_D_Mbit",
                "periodicity_error_E_Wh", "periodicity_error_D_Mbit",
            ]
        },
    }
    (outdir / "audit.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, outdir / "audit.md")
    print((outdir / "audit.md").read_text())
    print(f"[ok] wrote {outdir}/audit.json, audit.md, fig1, fig2")


if __name__ == "__main__":
    main()

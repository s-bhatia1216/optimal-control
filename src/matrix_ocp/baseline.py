"""FDR baseline policy simulation and the Phase-1 constraint audit.

The baseline is the fixed mode logic from the FDR (Tables 13 and 14):

    u_s = 1  inside the ROI, except in eclipse (Eclipse mode turns ESA/EPD off)
    u_c = 1  during the ground contact window, while there is data to send
    u_h      harvest as much as possible without exceeding E_max
             (the PPT power tracker throttles the array at full charge)

The simulation integrates the two-state dynamics with explicit Euler at a small
fixed step (loads are piecewise constant, so the only integration subtlety is
clipping at E_max and D = 0; the step is small enough that the clipping error is
negligible for an audit). It runs several orbits from a guess and keeps the last
one, which has converged to the periodic cycle.

The audit answers Checkpoint-1 questions: feasibility without the dump control,
which constraints are active, and how strongly science couples to the battery.
"""

from __future__ import annotations

import numpy as np

from .orbit import OrbitTimeline
from .params import Params


def simulate_baseline(p: Params, tl: OrbitTimeline) -> dict:
    """Simulate the FDR baseline policy to its periodic cycle.

    Returns a dict of arrays over the final orbit: time grid, states E and D,
    effective controls, window indicators, plus per-orbit energy/data tallies.
    """
    dt_h = p.dt_s / 3600.0
    n_per_orbit = int(round(tl.T_h / dt_h))
    dt_h = tl.T_h / n_per_orbit  # make the grid divide the period exactly

    n_total = n_per_orbit * p.n_orbits
    # States at step edges; controls on step intervals (evaluated at midpoints).
    E = np.empty(n_total + 1)
    D = np.empty(n_total + 1)
    E[0] = p.E_max   # start full; the cycle converges within one orbit
    D[0] = 0.0

    t_edges = np.arange(n_total + 1) * dt_h
    t_mid = t_edges[:-1] + 0.5 * dt_h

    # Window indicators and exogenous inputs at interval midpoints.
    roi = tl.in_roi(t_mid)
    ecl = tl.in_eclipse(t_mid)
    con = tl.in_contact(t_mid)
    P_in = tl.P_in(t_mid)
    P_0 = tl.P_0(t_mid)

    u_s = np.empty(n_total)
    u_c = np.empty(n_total)
    u_h = np.empty(n_total)

    for k in range(n_total):
        # --- FDR mode logic -------------------------------------------------
        us = 1.0 if (roi[k] and not ecl[k]) else 0.0

        # Transmit during contact while there is backlog. When the buffer is
        # empty, pass through the incoming rate (the radio sends fill data per
        # FDR Table 34; what matters to the model is that D stays at 0).
        if con[k]:
            inflow = p.r_hk + p.r_s * us
            uc = min(1.0, (D[k] / dt_h + inflow) / p.r_c)
        else:
            uc = 0.0

        # Harvest fully unless that would overfill the battery; then throttle
        # to exactly hold E_max (this is what the shunt/PPT does physically).
        load = P_0[k] + p.P_s * us + p.P_c * uc
        if P_in[k] > 0.0:
            uh = min(1.0, ((p.E_max - E[k]) / dt_h + load) / P_in[k])
            uh = max(uh, 0.0)
        else:
            uh = 0.0

        # --- Euler step -----------------------------------------------------
        E[k + 1] = E[k] + dt_h * (uh * P_in[k] - load)
        D[k + 1] = D[k] + dt_h * (p.r_hk + p.r_s * us - p.r_c * uc)
        # Guard against residual clipping error (order machine epsilon).
        E[k + 1] = min(E[k + 1], p.E_max)
        D[k + 1] = max(D[k + 1], 0.0)

        u_s[k], u_c[k], u_h[k] = us, uc, uh

    # Keep the final orbit (converged cycle) and re-base time to [0, T].
    s = slice(n_total - n_per_orbit, n_total)
    s_edge = slice(n_total - n_per_orbit, n_total + 1)
    out = {
        "t_edges_h": t_edges[s_edge] - t_edges[n_total - n_per_orbit],
        "t_mid_h": t_mid[s] - t_edges[n_total - n_per_orbit],
        "E_Wh": E[s_edge].copy(),
        "D_Mbit": D[s_edge].copy(),
        "u_s": u_s[s].copy(),
        "u_c": u_c[s].copy(),
        "u_h": u_h[s].copy(),
        "roi": roi[s].copy(),
        "eclipse": ecl[s].copy(),
        "contact": con[s].copy(),
        "P_in_W": P_in[s].copy(),
        "P_0_W": P_0[s].copy(),
        "dt_h": dt_h,
        "periodicity_error_E_Wh": abs(E[n_total] - E[n_total - n_per_orbit]),
        "periodicity_error_D_Mbit": abs(D[n_total] - D[n_total - n_per_orbit]),
    }
    out.update(_tallies(p, out))
    return out


def _tallies(p: Params, traj: dict) -> dict:
    """Per-orbit integrals of the converged cycle (all via the same grid)."""
    dt = traj["dt_h"]
    science_Mbit = float(np.sum(p.r_s * traj["u_s"]) * dt)
    science_h = float(np.sum(traj["u_s"]) * dt)
    downlinked_Mbit = float(np.sum(p.r_c * traj["u_c"]) * dt)
    solar_available_Wh = float(np.sum(traj["P_in_W"]) * dt)
    solar_harvested_Wh = float(np.sum(traj["u_h"] * traj["P_in_W"]) * dt)
    load_Wh = float(
        np.sum(traj["P_0_W"] + p.P_s * traj["u_s"] + p.P_c * traj["u_c"]) * dt
    )
    return {
        "science_h": science_h,
        "science_Mbit": science_Mbit,
        "housekeeping_Mbit": p.r_hk * traj["t_edges_h"][-1],
        "downlinked_Mbit": downlinked_Mbit,
        "solar_available_Wh": solar_available_Wh,
        "solar_harvested_Wh": solar_harvested_Wh,
        "dumped_Wh": solar_available_Wh - solar_harvested_Wh,
        "load_Wh": load_Wh,
        "min_E_Wh": float(np.min(traj["E_Wh"])),
        "max_D_Mbit": float(np.max(traj["D_Mbit"])),
    }


# ---------------------------------------------------------------------------
# Constraint audit
# ---------------------------------------------------------------------------

def eclipse_energy_need(p: Params, science_frac: float) -> float:
    """Battery energy [Wh] drawn over the worst eclipse.

    ``science_frac`` is the fraction of the eclipse spent with full science on
    (0 = FDR Eclipse mode, 1 = science through the whole eclipse). The dump and
    transmit controls do not matter here: P_in = 0 in eclipse and contact is at
    perigee, outside the eclipse.
    """
    return (p.P0_eclipse + science_frac * p.P_s) * p.eclipse_duration_h


def battery_binding_interval(p: Params, capacity_Wh: float) -> dict:
    """For a given capacity, which DoD floors make E_min slack / binding / infeasible.

    The battery starts the eclipse full (sunlit surplus is large, see audit), so
    the E >= E_min constraint over the eclipse reduces to
        usable = dod_floor * capacity  vs  eclipse draw.
    E_min is BINDING (rations eclipse science, comment 2 of the feedback) when
        need(science=0) < usable < need(science=1),
    infeasible below that interval, slack above it.
    """
    need0 = eclipse_energy_need(p, 0.0)
    need1 = eclipse_energy_need(p, 1.0)
    return {
        "capacity_Wh": capacity_Wh,
        "eclipse_need_no_science_Wh": need0,
        "eclipse_need_full_science_Wh": need1,
        "dod_floor_infeasible_below": need0 / capacity_Wh,
        "dod_floor_binding_interval": [need0 / capacity_Wh, need1 / capacity_Wh],
    }


def audit(p: Params, tl: OrbitTimeline, traj: dict) -> dict:
    """The Checkpoint-1 constraint audit for one parameter set."""
    T = tl.T_h

    # (a) Feasibility without the dump control: with u_h = 1 the net energy
    # over one orbit must be zero for a periodic solution. It is hugely
    # positive, so E would exceed E_max on the first sunlit arc: infeasible.
    surplus_Wh = traj["solar_available_Wh"] - traj["load_Wh"]
    # Time to fill the whole usable band at full harvest after eclipse exit
    # (net charge rate with science on, no transmit):
    net_charge_W = p.P_in - p.P0_day - p.P_s
    time_to_fill_h = p.E_usable / net_charge_W if net_charge_W > 0 else float("inf")

    # (b) Active constraints over the baseline periodic orbit.
    on_Emax = float(np.mean(np.isclose(traj["E_Wh"][:-1], p.E_max, atol=1e-6)))
    downlink_capacity_Mbit = p.r_c * p.contact_duration_h
    generated_Mbit = traj["science_Mbit"] + traj["housekeeping_Mbit"]

    # (c) Science-battery coupling at several capacity / DoD-floor variants.
    # (Capacity values are passed in by the caller through p / p_eol.)
    return {
        "case_name": p.case_name,
        "timeline": tl.summary(),
        "no_dump_feasibility": {
            "solar_available_Wh_per_orbit": traj["solar_available_Wh"],
            "total_load_Wh_per_orbit": traj["load_Wh"],
            "surplus_Wh_per_orbit": surplus_Wh,
            "surplus_fraction_of_available": surplus_Wh / traj["solar_available_Wh"],
            "mean_u_h_required": traj["load_Wh"] / traj["solar_available_Wh"],
            "hours_to_fill_usable_band_at_full_harvest": time_to_fill_h,
            "feasible_without_dump": bool(abs(surplus_Wh) < 1e-6),
        },
        "constraint_activity_baseline": {
            "E_max_active_fraction_of_orbit": on_Emax,
            "min_E_Wh": traj["min_E_Wh"],
            "E_min_Wh": p.E_min,
            "E_min_margin_Wh": traj["min_E_Wh"] - p.E_min,
            "E_min_active": bool(traj["min_E_Wh"] <= p.E_min + 1e-6),
            "max_D_Mbit": traj["max_D_Mbit"],
            "D_max_Mbit": p.D_max,
            "D_max_margin_Mbit": p.D_max - traj["max_D_Mbit"],
            "D_max_active": bool(traj["max_D_Mbit"] >= p.D_max - 1e-6),
            "data_generated_Mbit": generated_Mbit,
            "downlink_capacity_Mbit": downlink_capacity_Mbit,
            "downlink_utilization": generated_Mbit / downlink_capacity_Mbit,
            "downlink_active": bool(generated_Mbit >= downlink_capacity_Mbit - 1e-6),
        },
        "science_battery_coupling": {
            "P_s_W": p.P_s,
            "P0_eclipse_W": p.P0_eclipse,
            "full_eclipse_science_cost_Wh": p.P_s * p.eclipse_duration_h,
            "eclipse_need_no_science_Wh": eclipse_energy_need(p, 0.0),
            "eclipse_need_full_science_Wh": eclipse_energy_need(p, 1.0),
        },
    }

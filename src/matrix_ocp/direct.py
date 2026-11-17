"""Direct transcription of the MATRIX per-orbit OCP (trapezoidal collocation).

Discretization (details and dual conventions in docs/direct_method.md):

  - A mesh of N uniform intervals over [0, T], SNAPPED to the window edge
    times (ROI, eclipse, contact boundaries are inserted as extra nodes), so
    every interval lies purely inside or outside each window. States E, D
    live at the nodes; controls u_s, u_c, u_h are piecewise constant on
    intervals.
  - Exogenous inputs (P_in, P_0, contact fraction c, weight w) are AVERAGED
    over each interval (OrbitTimeline.interval_inputs). The dynamics are
    control-affine and INDEPENDENT of the states, so with interval-averaged
    inputs the one-step defect is the exact integral of the continuous
    dynamics for piecewise-constant controls; window edges inside an interval
    appear as fractional indicator values rather than O(h) quantization. The
    only remaining discretization restriction is that controls cannot switch
    inside an interval, which the mesh-refinement study quantifies.
  - Because objective and constraints are all linear, the transcribed NLP is
    a linear program. IPOPT (an interior-point method) converges to the
    analytic center of the optimal face, so on degenerate/singular arcs it
    returns smeared interior control values rather than an arbitrary vertex;
    we report those arcs rather than regularizing them away (per the
    professor's comment 2 and the checkpoint-1 decisions).

Costate extraction (validated in tests/test_lqr_multipliers.py on an LQR with
known Riccati costate):

    defect_k :  x_{k+1} - x_k - h f_k = 0   (equality, as written below)
    IPOPT dual lambda_k  =  -p_classic(t_{k+1/2})  =  +p_Liberzon(t_{k+1/2})

with NO timestep scaling. We store p_E, p_D in the Liberzon convention
(H = <p, f> + p0 L, p0 = -1, H maximized), the one used in docs/pmp_derivation.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from .orbit import OrbitTimeline
from .params import Params

# Validated in tests/test_lqr_multipliers.py: p_Liberzon = +1 * ipopt_dual.
LIBERZON_COSTATE_SIGN = +1


def solve_ocp(p: Params, tl: OrbitTimeline, N: int, max_iter: int = 3000) -> dict:
    """Solve the transcribed OCP; return trajectories, duals, and diagnostics.

    Returns a dict with status != "ok" (and no arrays) if IPOPT fails, which
    is how infeasible stress-sweep points are detected.
    """
    T = tl.T_h
    # Uniform base mesh + window edges (merging near-duplicates).
    t_edges = np.unique(np.concatenate(
        [np.linspace(0.0, T, N + 1), tl.window_edge_times()]))
    t_edges = t_edges[np.concatenate([[True], np.diff(t_edges) > 1e-9])]
    n_int = len(t_edges) - 1
    h = np.diff(t_edges)                      # per-interval widths [h]
    h_dm = ca.DM(h)
    t_mid = t_edges[:-1] + 0.5 * h

    # Exogenous inputs, averaged per interval (exact; see module docstring).
    inp = tl.interval_inputs(t_edges)
    P_in = ca.DM(inp["P_in_W"])
    P_0 = ca.DM(inp["P_0_W"])
    c_win = inp["contact_f"]              # numpy, bounds u_c (fractional at edges)
    w = inp["w_f"]                         # numpy, used in objective

    opti = ca.Opti()
    E = opti.variable(n_int + 1)
    D = opti.variable(n_int + 1)
    u_s = opti.variable(n_int)
    u_c = opti.variable(n_int)
    u_h = opti.variable(n_int)

    # --- dynamics (exact per interval; see module docstring) ---------------
    dE = u_h * P_in - P_0 - p.P_s * u_s - p.P_c * u_c      # [W] = [Wh/h]
    dD = p.r_hk + p.r_s * u_s - p.r_c * u_c                 # [Mbit/h]
    c_defE = (E[1:] - E[:-1] - h_dm * dE) == 0
    c_defD = (D[1:] - D[:-1] - h_dm * dD) == 0
    opti.subject_to(c_defE)
    opti.subject_to(c_defD)

    # --- periodic boundary conditions (initial values free) ----------------
    c_perE = (E[0] - E[n_int]) == 0
    c_perD = (D[0] - D[n_int]) == 0
    opti.subject_to(c_perE)
    opti.subject_to(c_perD)

    # --- path constraints (one-sided, so each gets its own dual) -----------
    c_Emax = E <= p.E_max
    c_Emin = E >= p.E_min
    c_Dmax = D <= p.D_max
    c_Dmin = D >= 0.0
    for con in (c_Emax, c_Emin, c_Dmax, c_Dmin):
        opti.subject_to(con)

    # Control boxes; u_c is additionally capped by the contact indicator.
    opti.subject_to(opti.bounded(0.0, u_s, 1.0))
    opti.subject_to(opti.bounded(0.0, u_h, 1.0))
    opti.subject_to(u_c >= 0.0)
    opti.subject_to(u_c <= ca.DM(c_win))

    # --- objective: maximize weighted science  ->  minimize -J -------------
    J = p.r_s * ca.dot(ca.DM(w * h), u_s)
    opti.minimize(-J)

    # Warm start near the baseline shape to help IPOPT.
    opti.set_initial(E, p.E_max * np.ones(n_int + 1))
    opti.set_initial(D, 100.0 * np.ones(n_int + 1))
    opti.set_initial(u_s, inp["roi_f"])
    opti.set_initial(u_c, 0.5 * c_win)
    opti.set_initial(u_h, 0.3 * np.ones(n_int))

    opti.solver(
        "ipopt",
        {"print_time": 0},
        {"print_level": 0, "sb": "yes", "max_iter": max_iter, "tol": 1e-9},
    )
    try:
        sol = opti.solve()
    except RuntimeError:
        return {"status": opti.return_status(), "N": N, "case": p.case_name}

    val = lambda v: np.atleast_1d(np.array(sol.value(v), dtype=float)).flatten()
    out = {
        "status": "ok",
        "return_status": sol.stats()["return_status"],
        "N": N,
        "n_intervals": n_int,
        "h_h": h,
        "case": p.case_name,
        "w_mode": p.w_mode,
        "t_edges_h": t_edges,
        "t_mid_h": t_mid,
        "E_Wh": val(E),
        "D_Mbit": val(D),
        "u_s": val(u_s),
        "u_c": val(u_c),
        "u_h": val(u_h),
        "w": np.asarray(w, dtype=float),
        "roi": inp["roi_f"],
        "eclipse": inp["eclipse_f"],
        "contact": c_win,
        "P_in_W": np.asarray(inp["P_in_W"], dtype=float),
        "P_0_W": np.asarray(inp["P_0_W"], dtype=float),
        "J_weighted_Mbit": float(sol.value(J)),
        # Costates in the Liberzon convention, defined at interval midpoints.
        "p_E": LIBERZON_COSTATE_SIGN * val(opti.dual(c_defE)),
        "p_D": LIBERZON_COSTATE_SIGN * val(opti.dual(c_defD)),
        # Node-wise duals of the state bounds (complementarity diagnostics).
        "mu_Emax": val(opti.dual(c_Emax)),
        "mu_Emin": val(opti.dual(c_Emin)),
        "mu_Dmax": val(opti.dual(c_Dmax)),
        "mu_Dmin": val(opti.dual(c_Dmin)),
        # Duals of the periodicity constraints (transversality check, Phase 3).
        "lam_perE": float(val(opti.dual(c_perE))[0]),
        "lam_perD": float(val(opti.dual(c_perD))[0]),
    }
    out.update(solution_metrics(p, tl, out))
    return out


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _arcs_from_mask(t_edges: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """Contiguous [t_in, t_out] intervals where a node mask is true."""
    arcs = []
    in_arc = False
    for k, m in enumerate(mask):
        if m and not in_arc:
            t0, in_arc = t_edges[k], True
        elif not m and in_arc:
            arcs.append([float(t0), float(t_edges[k - 1])])
            in_arc = False
    if in_arc:
        arcs.append([float(t0), float(t_edges[len(mask) - 1])])
    return arcs


def switch_times(t_mid: np.ndarray, u: np.ndarray, thresh: float = 0.5) -> list[float]:
    """Times where a control crosses ``thresh`` (linear interp on midpoints)."""
    out = []
    s = u - thresh
    for k in range(len(u) - 1):
        if s[k] == 0.0:
            continue
        if s[k] * s[k + 1] < 0.0:
            frac = s[k] / (s[k] - s[k + 1])
            out.append(float(t_mid[k] + frac * (t_mid[k + 1] - t_mid[k])))
    return out


def solution_metrics(p: Params, tl: OrbitTimeline, sol: dict, atol_E: float = 1e-3,
                     atol_D: float = 1e-3) -> dict:
    """Scalar diagnostics used in the refinement study and case comparison."""
    h = np.asarray(sol["h_h"])
    us, ecl, roi = sol["u_s"], sol["eclipse"], sol["roi"]
    E, D = sol["E_Wh"], sol["D_Mbit"]
    sci_roi_Mbit = float(p.r_s * np.sum(h * us * roi))
    return {
        "science_roi_h": float(np.sum(h * us * roi)),
        "science_roi_Mbit": sci_roi_Mbit,
        "science_offroi_h": float(np.sum(h * us * (1 - roi))),
        "science_eclipse_h": float(np.sum(h * us * ecl)),
        "downlinked_Mbit": float(p.r_c * np.sum(h * sol["u_c"])),
        "dumped_Wh": float(np.sum(h * (1 - sol["u_h"]) * sol["P_in_W"])),
        "min_E_Wh": float(np.min(E)),
        "max_D_Mbit": float(np.max(D)),
        "Emax_arcs_h": _arcs_from_mask(sol["t_edges_h"], E >= p.E_max - atol_E),
        "Emin_active_t_h": [float(t) for t in sol["t_edges_h"][E <= p.E_min + atol_E]],
        "Dmin_arcs_h": _arcs_from_mask(sol["t_edges_h"], D <= atol_D),
        "u_s_switch_times_h": switch_times(sol["t_mid_h"], sol["u_s"]),
        "periodicity_gap_E_Wh": float(abs(E[-1] - E[0])),
        "periodicity_gap_D_Mbit": float(abs(D[-1] - D[0])),
    }

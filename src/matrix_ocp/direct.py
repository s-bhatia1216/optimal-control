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
  - mesh="midpoint" keeps the naive scheme (uniform mesh, inputs sampled at
    interval midpoints) for the refinement comparison figure; it carries an
    O(h) window-quantization error that the snapped mesh removes.
  - Science is allowed only inside the ROI (u_s <= ROI indicator), matching
    the FDR operations concept where ESA/EPD run only in the magnetotail.
    Off-ROI science has zero weight and zero binding cost in the nominal and
    S1 cases, so this bound does not change the optimal value; it removes a
    degenerate direction from the optimal face.
  - Because objective and constraints are all linear, the transcribed NLP is
    a linear program. IPOPT (an interior-point method) converges to the
    analytic center of the optimal face, so on degenerate/singular arcs it
    returns smeared interior control values rather than an arbitrary vertex;
    those arcs are reported as-is rather than regularized away, since the
    degeneracy is real problem structure.
  - polish_solution() runs a second-stage lexicographic solve (hold the
    science value, maximize stored energy, then minimize stored data) that
    picks a flight-like member of the optimal face FOR PLOTTING ONLY; all
    PMP checks use the first-stage duals (valid by LP complementary
    slackness, verified numerically per case).

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


def _transcribe(p: Params, t_edges: np.ndarray, inp: dict) -> dict:
    """Build the transcribed LP on a given mesh with given per-interval inputs.

    Returns the Opti object plus handles to variables, constraints, and the
    science objective expression J, so callers can add stage-specific
    constraints/objectives (solve_ocp, polish_solution).
    """
    n_int = len(t_edges) - 1
    h = np.diff(t_edges)                      # per-interval widths [h]
    h_dm = ca.DM(h)
    c_win = inp["contact_f"]              # numpy, bounds u_c (fractional at edges)
    roi_f = inp["roi_f"]                   # numpy, bounds u_s (science in ROI only)
    w = inp["w_f"]                          # numpy, used in objective

    opti = ca.Opti()
    E = opti.variable(n_int + 1)
    D = opti.variable(n_int + 1)
    u_s = opti.variable(n_int)
    u_c = opti.variable(n_int)
    u_h = opti.variable(n_int)

    # --- dynamics (exact per interval; see module docstring) ---------------
    dE = u_h * ca.DM(inp["P_in_W"]) - ca.DM(inp["P_0_W"]) \
        - p.P_s * u_s - p.P_c * u_c                          # [W] = [Wh/h]
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

    # Control boxes; u_c is capped by the contact indicator and u_s by the
    # ROI indicator (instruments on only in the magnetotail, FDR Tables 13/14).
    opti.subject_to(u_s >= 0.0)
    opti.subject_to(u_s <= ca.DM(roi_f))
    opti.subject_to(opti.bounded(0.0, u_h, 1.0))
    opti.subject_to(u_c >= 0.0)
    opti.subject_to(u_c <= ca.DM(c_win))

    # Weighted science value (maximized by solve_ocp, held by the polish).
    J = p.r_s * ca.dot(ca.DM(w * h), u_s)

    # Warm start near the baseline shape to help IPOPT.
    opti.set_initial(E, p.E_max * np.ones(n_int + 1))
    opti.set_initial(D, 100.0 * np.ones(n_int + 1))
    opti.set_initial(u_s, roi_f)
    opti.set_initial(u_c, 0.5 * c_win)
    opti.set_initial(u_h, 0.3 * np.ones(n_int))

    return {
        "opti": opti, "E": E, "D": D, "u_s": u_s, "u_c": u_c, "u_h": u_h,
        "J": J, "h": h, "n_int": n_int,
        "c_defE": c_defE, "c_defD": c_defD, "c_perE": c_perE, "c_perD": c_perD,
        "c_Emax": c_Emax, "c_Emin": c_Emin, "c_Dmax": c_Dmax, "c_Dmin": c_Dmin,
    }


def _ipopt_solve(opti: ca.Opti, max_iter: int):
    opti.solver(
        "ipopt",
        {"print_time": 0},
        {"print_level": 0, "sb": "yes", "max_iter": max_iter, "tol": 1e-9},
    )
    return opti.solve()


def solve_ocp(p: Params, tl: OrbitTimeline, N: int, max_iter: int = 3000,
              mesh: str = "snapped") -> dict:
    """Solve the transcribed OCP; return trajectories, duals, and diagnostics.

    mesh="snapped" (default): window-edge-snapped mesh with interval-averaged
    inputs (exact transcription). mesh="midpoint": uniform mesh with inputs
    sampled at interval midpoints (the naive O(h) scheme, kept only for the
    refinement comparison figure).

    Returns a dict with status != "ok" (and no arrays) if IPOPT fails, which
    is how infeasible stress-sweep points are detected.
    """
    T = tl.T_h
    if mesh == "snapped":
        # Uniform base mesh + window edges (merging near-duplicates).
        t_edges = np.unique(np.concatenate(
            [np.linspace(0.0, T, N + 1), tl.window_edge_times()]))
        t_edges = t_edges[np.concatenate([[True], np.diff(t_edges) > 1e-9])]
        inp = tl.interval_inputs(t_edges)
    elif mesh == "midpoint":
        t_edges = np.linspace(0.0, T, N + 1)
        inp = tl.midpoint_inputs(t_edges)
    else:
        raise ValueError(f"unknown mesh mode {mesh!r}")
    n_int = len(t_edges) - 1
    h = np.diff(t_edges)
    t_mid = t_edges[:-1] + 0.5 * h

    tr = _transcribe(p, t_edges, inp)
    opti, J = tr["opti"], tr["J"]
    opti.minimize(-J)
    try:
        sol = _ipopt_solve(opti, max_iter)
    except RuntimeError:
        return {"status": opti.return_status(), "N": N, "case": p.case_name}

    val = lambda v: np.atleast_1d(np.array(sol.value(v), dtype=float)).flatten()
    out = {
        "status": "ok",
        "return_status": sol.stats()["return_status"],
        "N": N,
        "n_intervals": n_int,
        "mesh": mesh,
        "h_h": h,
        "case": p.case_name,
        "w_mode": p.w_mode,
        "t_edges_h": t_edges,
        "t_mid_h": t_mid,
        "E_Wh": val(tr["E"]),
        "D_Mbit": val(tr["D"]),
        "u_s": val(tr["u_s"]),
        "u_c": val(tr["u_c"]),
        "u_h": val(tr["u_h"]),
        "w": np.asarray(inp["w_f"], dtype=float),
        "roi": np.asarray(inp["roi_f"], dtype=float),
        "eclipse": np.asarray(inp["eclipse_f"], dtype=float),
        "contact": np.asarray(inp["contact_f"], dtype=float),
        "P_in_W": np.asarray(inp["P_in_W"], dtype=float),
        "P_0_W": np.asarray(inp["P_0_W"], dtype=float),
        "J_weighted_Mbit": float(sol.value(J)),
        # Costates in the Liberzon convention, defined at interval midpoints.
        "p_E": LIBERZON_COSTATE_SIGN * val(opti.dual(tr["c_defE"])),
        "p_D": LIBERZON_COSTATE_SIGN * val(opti.dual(tr["c_defD"])),
        # Node-wise duals of the state bounds (complementarity diagnostics).
        "mu_Emax": val(opti.dual(tr["c_Emax"])),
        "mu_Emin": val(opti.dual(tr["c_Emin"])),
        "mu_Dmax": val(opti.dual(tr["c_Dmax"])),
        "mu_Dmin": val(opti.dual(tr["c_Dmin"])),
        # Duals of the periodicity constraints (transversality check, Phase 3).
        "lam_perE": float(val(opti.dual(tr["c_perE"]))[0]),
        "lam_perD": float(val(opti.dual(tr["c_perD"]))[0]),
    }
    out.update(solution_metrics(p, tl, out))
    return out


def polish_solution(p: Params, tl: OrbitTimeline, sol1: dict,
                    max_iter: int = 3000) -> dict:
    """Second-stage lexicographic solve that picks a clean, flight-like member
    of the stage-1 optimal face, FOR PLOTTING ONLY.

    Stage 2a: hold the weighted science value at its stage-1 optimum and
    maximize the stored-energy integral, so E rides E_max instead of drifting
    (resolves the harvest-timing degeneracy). Stage 2b: additionally hold the
    stored-energy integral and minimize the stored-data integral (resolves the
    free D offset; the buffer empties at contact like the flight rule).

    The returned dict carries the polished primal but keeps the STAGE-1 duals
    (p_E, p_D, mu_*, lam_per*) untouched: for a linear program, any optimal
    dual pairs with any optimal primal under complementary slackness, so the
    PMP checks remain valid along the polished trajectory. That pairing is
    verified numerically in "cs_violation" (max |mu * slack| per bound).
    """
    t_edges = np.asarray(sol1["t_edges_h"])
    h = np.diff(t_edges)
    inp = {k: sol1[src] for k, src in [
        ("roi_f", "roi"), ("eclipse_f", "eclipse"), ("contact_f", "contact"),
        ("w_f", "w"), ("P_in_W", "P_in_W"), ("P_0_W", "P_0_W")]}
    J_star = sol1["J_weighted_Mbit"]
    tol_J = 1e-6 * max(1.0, abs(J_star))
    # Trapezoid weights for integrals of the (piecewise-linear) states.
    wq = np.zeros(len(t_edges))
    wq[:-1] += 0.5 * h
    wq[1:] += 0.5 * h
    wq_dm = ca.DM(wq)

    def _warm(tr):
        for key, var in [("E_Wh", "E"), ("D_Mbit", "D"), ("u_s", "u_s"),
                         ("u_c", "u_c"), ("u_h", "u_h")]:
            tr["opti"].set_initial(tr[var], sol1[key])

    # Stage 2a: max integral of E subject to J >= J*.
    tr = _transcribe(p, t_edges, inp)
    tr["opti"].subject_to(tr["J"] >= J_star - tol_J)
    tr["opti"].minimize(-ca.dot(wq_dm, tr["E"]))
    _warm(tr)
    sol_a = _ipopt_solve(tr["opti"], max_iter)
    intE_star = float(sol_a.value(ca.dot(wq_dm, tr["E"])))
    tol_E = 1e-6 * max(1.0, abs(intE_star))

    # Stage 2b: min integral of D subject to J >= J* and int(E) >= int(E)*.
    tr = _transcribe(p, t_edges, inp)
    tr["opti"].subject_to(tr["J"] >= J_star - tol_J)
    tr["opti"].subject_to(ca.dot(wq_dm, tr["E"]) >= intE_star - tol_E)
    tr["opti"].minimize(ca.dot(wq_dm, tr["D"]))
    _warm(tr)
    sol_b = _ipopt_solve(tr["opti"], max_iter)

    val = lambda v: np.atleast_1d(np.array(sol_b.value(v), dtype=float)).flatten()
    out = dict(sol1)
    out.update({
        "polished": True,
        "E_Wh": val(tr["E"]),
        "D_Mbit": val(tr["D"]),
        "u_s": val(tr["u_s"]),
        "u_c": val(tr["u_c"]),
        "u_h": val(tr["u_h"]),
        "J_weighted_Mbit": float(sol_b.value(tr["J"])),
    })
    out.update(solution_metrics(p, tl, out))
    # Complementary slackness of (stage-1 duals, polished primal): for an LP
    # these pair exactly; the residuals below are the numerical check.
    cs = {
        "Emax_Mbit": float(np.max(np.abs(sol1["mu_Emax"] * (p.E_max - out["E_Wh"])))),
        "Emin_Mbit": float(np.max(np.abs(sol1["mu_Emin"] * (out["E_Wh"] - p.E_min)))),
        "Dmax_Mbit": float(np.max(np.abs(sol1["mu_Dmax"] * (p.D_max - out["D_Mbit"])))),
        "Dmin_Mbit": float(np.max(np.abs(sol1["mu_Dmin"] * out["D_Mbit"]))),
    }
    out["cs_violation"] = cs
    out["cs_max_violation"] = max(cs.values())
    out["J_gap_vs_stage1_Mbit"] = float(out["J_weighted_Mbit"] - J_star)
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

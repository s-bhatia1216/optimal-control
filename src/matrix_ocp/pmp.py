"""Numerical PMP consistency checks along the direct solutions.

Everything here evaluates the first-order conditions derived in
docs/pmp_derivation.md along a STAGE-1 direct solution (the one whose IPOPT
duals are the costates; see docs/direct_method.md Sections 2 and 4):

  - switching functions theta_s, theta_c, theta_h per interval, and the
    sign/bang agreement of each control with its switching function;
  - piecewise constancy of the costates and the jump identity
    delta p(node) = combination of the state-bound duals at that node
    (the discrete counterpart of the state-constrained costate jump);
  - the boundary-arc control u_h = (P_0 + P_s u_s + P_c u_c)/P_in on
    E = E_max arcs (checked on the polished trajectory, which rides E_max);
  - periodic transversality p(0) = p(T), and against the duals of the
    periodicity constraints;
  - continuity of the Hamiltonian across control switches, after removing
    the explicit time dependence (window edges, apogee-weighted w(t)).

Convention throughout: Liberzon (2012) / the course notes Sections 16-18,
H = <p, f> + p0 L with p0 = -1 and H maximized pointwise in u.
"""

from __future__ import annotations

import numpy as np

from .params import Params


def switching_functions(p: Params, sol: dict) -> dict:
    """theta_i = dH/du_i per interval [Mbit/h], H in the maximize convention.

    H = p_E (u_h P_in - P_0 - P_s u_s - P_c u_c)
      + p_D (r_hk + r_s u_s - r_c u_c) + w r_s u_s          (p0 = -1)
    """
    pE, pD, w = sol["p_E"], sol["p_D"], sol["w"]
    return {
        "theta_s": p.r_s * (w + pD) - p.P_s * pE,
        "theta_c": -(p.P_c * pE + p.r_c * pD),
        "theta_h": sol["P_in_W"] * pE,
    }


def hamiltonian(p: Params, sol: dict) -> np.ndarray:
    """H per interval along the solution (maximize convention, p0 = -1)."""
    pE, pD = sol["p_E"], sol["p_D"]
    fE = sol["u_h"] * sol["P_in_W"] - sol["P_0_W"] \
        - p.P_s * sol["u_s"] - p.P_c * sol["u_c"]
    fD = p.r_hk + p.r_s * sol["u_s"] - p.r_c * sol["u_c"]
    return pE * fE + pD * fD + sol["w"] * p.r_s * sol["u_s"]


def bang_agreement(theta: np.ndarray, u: np.ndarray, ub: np.ndarray,
                   tol_theta: float, tol_u: float = 1e-5) -> dict:
    """Check u against the bang-bang law u = ub if theta > 0, 0 if theta < 0.

    Intervals with |theta| <= tol_theta are classified singular (the law is
    silent there); intervals with ub = 0 are excluded (the control is fixed
    by its own bound, no PMP content). Returns counts and the worst
    violation of the law on the decided intervals.
    """
    free = ub > tol_u
    on = free & (theta > tol_theta)
    off = free & (theta < -tol_theta)
    sing = free & ~on & ~off
    viol_on = float(np.max(ub[on] - u[on])) if np.any(on) else 0.0
    viol_off = float(np.max(u[off])) if np.any(off) else 0.0
    n_dec = int(np.sum(on) + np.sum(off))
    n_ok = int(np.sum(ub[on] - u[on] <= tol_u) + np.sum(u[off] <= tol_u))
    return {
        "n_free": int(np.sum(free)),
        "n_on": int(np.sum(on)),
        "n_off": int(np.sum(off)),
        "n_singular": int(np.sum(sing)),
        "n_decided": n_dec,
        "n_decided_ok": n_ok,
        "agreement": (n_ok / n_dec) if n_dec else None,
        "worst_on_violation": viol_on,
        "worst_off_violation": viol_off,
        "max_abs_theta_singular": float(np.max(np.abs(theta[sing]))) if np.any(sing) else 0.0,
    }


def costate_jump_check(sol: dict) -> dict:
    """Verify delta p(node) = s_max * mu_max(node) + s_min * mu_min(node).

    This is the KKT stationarity of the transcribed LP with respect to the
    interior state nodes: between two defects the costate can change only by
    the bound duals at the shared node. The signs s in {-1, +1} depend on the
    dual-sign convention of the NLP layer, so they are FIT (the combination
    with the smallest worst-case residual) and reported; the content of the
    check is that one fixed sign pair explains every jump to solver
    tolerance. Jumps above ``jump_tol`` are reported as junctions.
    """
    out = {}
    for state, mu_hi, mu_lo in [("E", "mu_Emax", "mu_Emin"),
                                ("D", "mu_Dmax", "mu_Dmin")]:
        dp = np.diff(sol[f"p_{state}"])              # jump at nodes 1..n-1
        hi = sol[mu_hi][1:-1]
        lo = sol[mu_lo][1:-1]
        best = None
        for s_hi in (+1.0, -1.0):
            for s_lo in (+1.0, -1.0):
                r = float(np.max(np.abs(dp - s_hi * hi - s_lo * lo)))
                if best is None or r < best[0]:
                    best = (r, s_hi, s_lo)
        resid, s_hi, s_lo = best
        jumps = []
        t_nodes = sol["t_edges_h"][1:-1]
        big = np.abs(dp) > 1e-3
        for i in np.flatnonzero(big):
            jumps.append({
                "t_h": float(t_nodes[i]),
                "delta_p": float(dp[i]),
                "mu_upper": float(hi[i]),
                "mu_lower": float(lo[i]),
            })
        out[state] = {
            "sign_upper": s_hi, "sign_lower": s_lo,
            "max_residual": resid,
            "junctions": jumps,
            "n_junctions": len(jumps),
        }
    return out


def boundary_arc_check(p: Params, sol_pol: dict, atol_E: float = 1e-3) -> dict:
    """On sunlit E = E_max arcs the dynamics force the interior control
    u_h = (P_0 + P_s u_s + P_c u_c) / P_in. Checked on the POLISHED
    trajectory, which rides E_max (the stage-1 analytic center drifts)."""
    E = sol_pol["E_Wh"]
    on_arc = (E[:-1] >= p.E_max - atol_E) & (E[1:] >= p.E_max - atol_E) \
        & (sol_pol["P_in_W"] > 1.0)
    if not np.any(on_arc):
        return {"n_intervals": 0, "max_residual": None, "arc_hours": 0.0}
    u_h_pred = (sol_pol["P_0_W"] + p.P_s * sol_pol["u_s"]
                + p.P_c * sol_pol["u_c"]) / np.where(sol_pol["P_in_W"] > 0,
                                                     sol_pol["P_in_W"], np.inf)
    resid = np.abs(sol_pol["u_h"] - u_h_pred)[on_arc]
    interior = (sol_pol["u_h"][on_arc] > 1e-4) & (sol_pol["u_h"][on_arc] < 1 - 1e-4)
    return {
        "n_intervals": int(np.sum(on_arc)),
        "arc_hours": float(np.sum(sol_pol["h_h"][on_arc])),
        "max_residual": float(np.max(resid)),
        "fraction_interior": float(np.mean(interior)),
    }


def transversality_check(sol: dict, meta: dict) -> dict:
    """Periodic BCs x(0) = x(T) require p(0) = p(T); the duals of the
    periodicity constraints should match the endpoint costates."""
    out = {}
    for state, lam_key in [("E", "lam_perE"), ("D", "lam_perD")]:
        pp = sol[f"p_{state}"]
        out[state] = {
            "p_first": float(pp[0]),
            "p_last": float(pp[-1]),
            "gap": float(abs(pp[0] - pp[-1])),
            "lam_per": float(meta[lam_key]),
            "lam_matches_p0": float(min(abs(meta[lam_key] - pp[0]),
                                        abs(meta[lam_key] + pp[0]))),
        }
    return out


def hamiltonian_continuity_check(p: Params, sol: dict,
                                 edge_tol: float = 1e-9) -> dict:
    """H along the solution should be continuous across control switches and
    constant where the problem is autonomous; it may jump only where the
    exogenous inputs jump (window edges) or a state-constraint junction
    moves the costate. We check the jump of H across every interior node
    after removing the explicit time dependence (w, P_in, P_0 differences),
    excluding nodes where the inputs or the costates jump."""
    H = hamiltonian(p, sol)
    dH = np.diff(H)
    # Explicit time dependence between neighbor intervals.
    us_bar = 0.5 * (sol["u_s"][1:] + sol["u_s"][:-1])
    uh_bar = 0.5 * (sol["u_h"][1:] + sol["u_h"][:-1])
    pE_bar = 0.5 * (sol["p_E"][1:] + sol["p_E"][:-1])
    expl = p.r_s * us_bar * np.diff(sol["w"]) \
        + pE_bar * (uh_bar * np.diff(sol["P_in_W"]) - np.diff(sol["P_0_W"]))
    window_edge = (np.abs(np.diff(sol["P_in_W"])) > edge_tol) \
        | (np.abs(np.diff(sol["contact"])) > edge_tol) \
        | (np.abs(np.diff(sol["roi"])) > edge_tol)
    costate_jump = (np.abs(np.diff(sol["p_E"])) > 1e-6) \
        | (np.abs(np.diff(sol["p_D"])) > 1e-6)
    plain = ~window_edge & ~costate_jump
    resid = np.abs(dH - expl)[plain]
    return {
        "n_plain_nodes": int(np.sum(plain)),
        "max_residual_plain": float(np.max(resid)) if np.any(resid) else 0.0,
        "n_excluded_window": int(np.sum(window_edge)),
        "n_excluded_costate": int(np.sum(costate_jump)),
    }


def run_all_checks(p: Params, sol: dict, sol_pol: dict, meta: dict,
                   theta_tol_frac: float = 1e-5) -> dict:
    """All PMP checks for one case. theta tolerances scale with each
    switching function's dynamic range (LP duals carry ~1e-9 solver noise)."""
    th = switching_functions(p, sol)
    scale = {k: max(1.0, float(np.max(np.abs(v)))) for k, v in th.items()}
    checks = {
        "theta_scales": scale,
        "bang_s": bang_agreement(th["theta_s"], sol["u_s"], sol["roi"],
                                 theta_tol_frac * scale["theta_s"]),
        "bang_c": bang_agreement(th["theta_c"], sol["u_c"], sol["contact"],
                                 theta_tol_frac * scale["theta_c"]),
        "bang_h": bang_agreement(th["theta_h"], sol["u_h"],
                                 np.ones_like(sol["u_h"]),
                                 theta_tol_frac * scale["theta_h"]),
        "costate_jumps": costate_jump_check(sol),
        "boundary_arc": boundary_arc_check(p, sol_pol),
        "transversality": transversality_check(sol, meta),
        "hamiltonian": hamiltonian_continuity_check(p, sol),
        # No sustained E_min arc can exist in eclipse: the best achievable
        # dE/dt there (all loads off, no sun) is still negative.
        "eclipse_best_dEdt_W": -p.P0_eclipse,
    }
    return checks

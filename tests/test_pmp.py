"""PMP consistency checks as regressions (small mesh, fast).

These assert the Phase-3 first-order-condition checks numerically on
s1_uniform, the case that exercises both state-constraint junction types
(E_max arc exit at eclipse entry, E_min touch at eclipse exit) and a
singular arc. Full-size results live in results/<case>/pmp_checks.md.
"""

from pathlib import Path

import numpy as np
import pytest

from matrix_ocp.direct import polish_solution, solve_ocp
from matrix_ocp.orbit import OrbitTimeline
from matrix_ocp.params import Params
from matrix_ocp.pmp import run_all_checks

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def s1_checks():
    p = Params.from_yaml(CONFIGS / "s1_uniform.yaml")
    tl = OrbitTimeline(p)
    sol = solve_ocp(p, tl, N=360)
    assert sol["status"] == "ok"
    pol = polish_solution(p, tl, sol)
    return p, sol, run_all_checks(p, sol, pol, meta=sol)


def test_bang_bang_agreement(s1_checks):
    p, sol, c = s1_checks
    assert c["bang_s"]["agreement"] == 1.0
    assert c["bang_s"]["n_singular"] > 0          # the eclipse ration arc
    assert c["bang_s"]["max_abs_theta_singular"] < 1e-6
    # Energy is free in sunlight (dump): theta_h vanishes identically.
    assert c["bang_h"]["n_decided"] == 0
    assert c["bang_h"]["max_abs_theta_singular"] < 1e-6


def test_costate_jump_identity(s1_checks):
    p, sol, c = s1_checks
    jE = c["costate_jumps"]["E"]
    # delta p = mu_upper - mu_lower at every interior node, to solver tol.
    assert (jE["sign_upper"], jE["sign_lower"]) == (1.0, -1.0)
    assert jE["max_residual"] < 1e-9
    # Exactly two junctions: E_max exit at eclipse entry, E_min touch at
    # eclipse exit, each jump equal to the bound multiplier point mass.
    assert jE["n_junctions"] == 2
    j_entry, j_touch = jE["junctions"]
    rsps = p.r_s / p.P_s
    assert j_entry["delta_p"] == pytest.approx(+rsps, rel=1e-3)
    assert j_entry["mu_upper"] == pytest.approx(+rsps, rel=1e-3)
    assert j_touch["delta_p"] == pytest.approx(-rsps, rel=1e-3)
    assert j_touch["mu_lower"] == pytest.approx(+rsps, rel=1e-3)


def test_boundary_arc_interior_control(s1_checks):
    p, sol, c = s1_checks
    ba = c["boundary_arc"]
    assert ba["n_intervals"] > 0
    assert ba["max_residual"] < 1e-2
    assert ba["fraction_interior"] == 1.0


def test_periodic_transversality(s1_checks):
    p, sol, c = s1_checks
    for state in ("E", "D"):
        t = c["transversality"][state]
        assert t["gap"] < 1e-9
        assert t["lam_matches_p0"] < 1e-9

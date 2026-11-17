"""Regression checks for the direct method on the MATRIX cases.

The expected values are the hand-derivable optima (docs/direct_method.md):
the mesh is snapped to the window edges and the dynamics are state-independent
and affine, so the transcription is exact and these targets are hit at any N.
"""

from pathlib import Path

import numpy as np
import pytest

from matrix_ocp.direct import solve_ocp
from matrix_ocp.orbit import OrbitTimeline
from matrix_ocp.params import Params

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _solve(case: str, N: int = 360):
    p = Params.from_yaml(CONFIGS / f"{case}.yaml")
    tl = OrbitTimeline(p)
    sol = solve_ocp(p, tl, N)
    assert sol["status"] == "ok"
    assert sol["return_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    return p, tl, sol


def test_nominal_optimum_is_full_roi_science():
    p, tl, sol = _solve("nominal")
    # All constraints slack except E_max (dump): science fills the ROI.
    assert sol["science_roi_h"] == pytest.approx(tl.roi_duration_h, abs=1e-4)
    assert sol["science_roi_Mbit"] == pytest.approx(p.r_s * tl.roi_duration_h, abs=0.01)
    assert sol["min_E_Wh"] >= p.E_min - 1e-6
    assert sol["periodicity_gap_E_Wh"] < 1e-6
    assert sol["periodicity_gap_D_Mbit"] < 1e-6


def test_s1_ration_matches_energy_budget():
    p, tl, sol = _solve("s1_uniform")
    # Affordable eclipse science: (usable - eclipse base draw) / P_s.
    budget_h = (p.dod_floor * p.capacity_Wh
                - p.P0_eclipse * p.eclipse_duration_h) / p.P_s
    assert sol["science_eclipse_h"] == pytest.approx(budget_h, abs=1e-3)
    # The battery floor is reached (touch at eclipse exit).
    assert sol["min_E_Wh"] == pytest.approx(p.E_min, abs=1e-4)
    # PMP prediction: on the rationed (singular) arc p_E = r_s / P_s.
    full_ecl = sol["eclipse"] > 0.999
    assert np.mean(sol["p_E"][full_ecl]) == pytest.approx(p.r_s / p.P_s, rel=1e-3)


def test_s2_science_capped_by_downlink():
    p, tl, sol = _solve("s2_uniform")
    cap_Mbit = p.r_c * p.contact_duration_h
    max_sci_h = (cap_Mbit - p.r_hk * tl.T_h) / p.r_s
    assert sol["science_roi_h"] == pytest.approx(max_sci_h, abs=1e-3)
    assert sol["downlinked_Mbit"] == pytest.approx(cap_Mbit, abs=0.05)
    # Data shadow price: with uniform w the rationed science is singular and
    # p_D sits at -w = -1 (marginal stored Mbit costs one science Mbit).
    assert np.median(sol["p_D"]) == pytest.approx(-1.0, abs=1e-3)


def test_apogee_weight_breaks_s1_tie():
    """With w = r/r_a the rationed eclipse science concentrates at apogee."""
    p, tl, sol = _solve("s1_apogee")
    ecl = sol["eclipse"] > 0.999
    t = sol["t_mid_h"][ecl]
    us = sol["u_s"][ecl]
    t_apo = tl.t_eclipse_center_h
    near = np.abs(t - t_apo) < 0.8   # inner +/- 0.8 h of the eclipse
    far = np.abs(t - t_apo) > 1.6
    assert us[near].mean() > 0.95    # on near apogee
    assert us[far].mean() < 0.05     # off at the eclipse edges

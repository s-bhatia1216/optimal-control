"""Orbit timeline checks against FDR-quoted values (Tables 46, 47, 20)."""

from pathlib import Path

import numpy as np
import pytest

from matrix_ocp.orbit import OrbitTimeline, solve_kepler
from matrix_ocp.params import Params

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nominal.yaml"


@pytest.fixture(scope="module")
def tl():
    return OrbitTimeline(Params.from_yaml(CONFIG))


def test_kepler_solver_residual():
    e = 0.825
    M = np.linspace(0, 2 * np.pi, 1001)
    E = solve_kepler(M, e)
    assert np.max(np.abs(E - e * np.sin(E) - np.mod(M, 2 * np.pi))) < 1e-10


def test_period_matches_fdr(tl):
    # FDR Table 46: Group 1 period is 24 h.
    assert tl.T_h == pytest.approx(24.0, rel=1e-3)


def test_radius_endpoints(tl):
    p = tl.p
    assert float(tl.radius_RE(0.0)) == pytest.approx(p.r_p_RE, rel=1e-9)
    assert float(tl.radius_RE(tl.T_h / 2)) == pytest.approx(p.r_a_RE, rel=1e-9)


def test_roi_fraction_matches_fdr(tl):
    # FDR Tables 46 and 20: 79.73% of the orbit is beyond 6 R_E.
    assert tl.roi_duration_h / tl.T_h == pytest.approx(0.7973, abs=2e-3)
    # Indicator must agree with the analytic crossing times.
    t = np.linspace(0, tl.T_h, 200001)
    frac = np.mean(tl.in_roi(t))
    assert frac == pytest.approx(tl.roi_duration_h / tl.T_h, abs=1e-4)


def test_window_durations(tl):
    p = tl.p
    t = np.linspace(0, tl.T_h, 400001)[:-1]
    dt = tl.T_h / (len(t))
    assert np.sum(tl.in_eclipse(t)) * dt == pytest.approx(p.eclipse_duration_h, abs=2e-3)
    assert np.sum(tl.in_contact(t)) * dt == pytest.approx(p.contact_duration_h, abs=2e-3)


def test_eclipse_inside_roi_and_contact_outside(tl):
    lo, hi = tl.t_eclipse_center_h - 1e-6, tl.t_eclipse_center_h + 1e-6
    assert tl.in_roi(np.array([lo, hi])).all()
    # Eclipse bounds within ROI bounds (nominal geometry).
    assert tl.t_eclipse_center_h - 0.5 * tl.p.eclipse_duration_h > tl.t_roi_entry_h
    # Contact (at perigee) is outside the ROI.
    assert not tl.in_roi(0.0)

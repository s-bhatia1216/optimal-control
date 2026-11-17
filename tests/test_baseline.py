"""Baseline policy simulation checks: periodicity, bounds, balances."""

from pathlib import Path

import numpy as np
import pytest

from matrix_ocp.baseline import eclipse_energy_need, simulate_baseline
from matrix_ocp.orbit import OrbitTimeline
from matrix_ocp.params import Params

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nominal.yaml"


@pytest.fixture(scope="module")
def setup():
    p = Params.from_yaml(CONFIG)
    tl = OrbitTimeline(p)
    traj = simulate_baseline(p, tl)
    return p, tl, traj


def test_periodicity(setup):
    p, tl, traj = setup
    assert traj["periodicity_error_E_Wh"] < 1e-6
    assert traj["periodicity_error_D_Mbit"] < 1e-6
    assert abs(traj["E_Wh"][-1] - traj["E_Wh"][0]) < 1e-6
    assert abs(traj["D_Mbit"][-1] - traj["D_Mbit"][0]) < 1e-6


def test_bounds_satisfied(setup):
    p, tl, traj = setup
    assert np.all(traj["E_Wh"] <= p.E_max + 1e-9)
    assert np.all(traj["E_Wh"] >= p.E_min - 1e-9)  # slack at nominal
    assert np.all(traj["D_Mbit"] >= -1e-9)
    assert np.all(traj["D_Mbit"] <= p.D_max + 1e-9)
    assert np.all((traj["u_s"] >= 0) & (traj["u_s"] <= 1))
    assert np.all((traj["u_c"] >= 0) & (traj["u_c"] <= 1))
    assert np.all((traj["u_h"] >= 0) & (traj["u_h"] <= 1 + 1e-12))


def test_energy_balance(setup):
    """E(T) - E(0) must equal the integral of dE/dt (harvest minus loads)."""
    p, tl, traj = setup
    dt = traj["dt_h"]
    net = (traj["u_h"] * traj["P_in_W"] - traj["P_0_W"]
           - p.P_s * traj["u_s"] - p.P_c * traj["u_c"])
    assert float(np.sum(net) * dt) == pytest.approx(
        traj["E_Wh"][-1] - traj["E_Wh"][0], abs=1e-6)
    # Equivalently: harvested energy equals total load over the periodic orbit.
    assert traj["solar_harvested_Wh"] == pytest.approx(traj["load_Wh"], abs=1e-6)


def test_data_balance(setup):
    """Over the periodic orbit, downlinked = science + housekeeping generated."""
    p, tl, traj = setup
    generated = traj["science_Mbit"] + traj["housekeeping_Mbit"]
    assert traj["downlinked_Mbit"] == pytest.approx(generated, abs=1e-6)


def test_min_energy_matches_hand_calc(setup):
    """Baseline min E should be E_max minus the no-science eclipse draw.

    The battery is full at eclipse entry (sunlit surplus >> loads) and the FDR
    baseline turns science off in eclipse, so the hand calculation is exact.
    """
    p, tl, traj = setup
    expected = p.E_max - eclipse_energy_need(p, 0.0)
    assert traj["min_E_Wh"] == pytest.approx(expected, abs=0.5)


def test_science_window_matches_roi_minus_eclipse(setup):
    p, tl, traj = setup
    expected_h = tl.roi_duration_h - p.eclipse_duration_h
    assert traj["science_h"] == pytest.approx(expected_h, abs=2 * traj["dt_h"])

"""Problem parameters for the MATRIX per-orbit optimal control problem.

All raw numbers live in YAML config files (configs/) with units in their key
names and FDR citations in comments. This module converts them once into the
working units used everywhere else in the project:

    time    : hours  [h]
    energy  : watt-hours [Wh]   (so power in W times time in h is already Wh)
    data    : megabits [Mbit]   (so rates are Mbit/h)

Those units keep every state and control within a few orders of magnitude of 1,
which is what the direct-method solver (IPOPT, Phase 2) wants to see.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

# Conversion: bits/second -> Mbit/hour.  (x bps) * 3600 s/h / 1e6 bit/Mbit
BPS_TO_MBIT_PER_H = 3600.0 / 1e6


@dataclass(frozen=True)
class Params:
    """All model parameters in working units (h, Wh, Mbit).

    The battery bounds follow the FDR convention: a depth-of-discharge (DoD)
    floor of ``dod_floor`` means the battery may discharge at most that fraction
    of capacity, so E_min = (1 - dod_floor) * capacity and E_max = capacity.
    """

    case_name: str

    # --- orbit geometry (used by orbit.OrbitTimeline) ---
    r_p_RE: float          # perigee radius [R_E]
    r_a_RE: float          # apogee radius [R_E]
    roi_radius_RE: float   # science region is r > this [R_E]
    R_E_km: float
    mu_km3_s2: float

    # --- windows ---
    eclipse_duration_h: float
    eclipse_center_frac: float  # fraction of period past perigee (0.5 = apogee)
    contact_duration_h: float
    contact_center_frac: float  # 0.0 = centered on perigee

    # --- power [W]; W * h = Wh in working units ---
    P_in: float        # solar array output when sunlit
    P0_day: float      # uncontrollable bus load, sunlit
    P0_eclipse: float  # uncontrollable bus load, eclipse (heaters on)
    P_s: float         # controllable science payload power (ESA + EPD)
    P_c: float         # controllable transmit power (power amp + radio TX delta)

    # --- battery [Wh] ---
    capacity_Wh: float
    dod_floor: float

    # --- data [Mbit, Mbit/h] ---
    r_s: float         # science generation rate at u_s = 1
    r_hk: float        # always-on housekeeping generation rate (0 if disabled)
    r_c: float         # downlink rate at u_c = 1
    D_max: float       # solid-state recorder size

    # --- simulation settings ---
    dt_s: float
    n_orbits: int

    # ------------------------------------------------------------------
    @property
    def E_max(self) -> float:
        """Battery upper bound [Wh]: full charge."""
        return self.capacity_Wh

    @property
    def E_min(self) -> float:
        """Battery lower bound [Wh] from the DoD floor."""
        return (1.0 - self.dod_floor) * self.capacity_Wh

    @property
    def E_usable(self) -> float:
        """Energy available between full charge and the DoD floor [Wh]."""
        return self.dod_floor * self.capacity_Wh

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Params":
        """Load a config file and convert everything to working units."""
        raw = yaml.safe_load(Path(path).read_text())

        orbit = raw["orbit"]
        win = raw["windows"]
        pw = raw["power_W"]
        bat = raw["battery"]
        dat = raw["data"]
        sim = raw.get("simulation", {})

        capacity = (
            bat["capacity_eol_Wh"] if bat.get("use_eol_capacity", False)
            else bat["capacity_Wh"]
        )
        r_hk = dat["r_hk_bps"] * BPS_TO_MBIT_PER_H
        if not dat.get("include_housekeeping", True):
            r_hk = 0.0

        return cls(
            case_name=raw["case_name"],
            r_p_RE=orbit["r_p_RE"],
            r_a_RE=orbit["r_a_RE"],
            roi_radius_RE=orbit["roi_radius_RE"],
            R_E_km=orbit["R_E_km"],
            mu_km3_s2=orbit["mu_km3_s2"],
            eclipse_duration_h=win["eclipse_duration_h"],
            eclipse_center_frac=win["eclipse_center_frac"],
            contact_duration_h=win["contact_duration_h"],
            contact_center_frac=win["contact_center_frac"],
            P_in=pw["P_in"],
            P0_day=pw["P0_day"],
            P0_eclipse=pw["P0_eclipse"],
            P_s=pw["P_s"],
            P_c=pw["P_c"],
            capacity_Wh=capacity,
            dod_floor=bat["dod_floor"],
            r_s=dat["r_s_bps"] * BPS_TO_MBIT_PER_H,
            r_hk=r_hk,
            # Storage: decimal GB -> bits -> Mbit (4 GB = 32e9 bit = 32000 Mbit).
            r_c=dat["r_c_bps"] * BPS_TO_MBIT_PER_H,
            D_max=dat["storage_GB"] * 8e9 / 1e6,
            dt_s=float(sim.get("dt_s", 10.0)),
            n_orbits=int(sim.get("n_orbits", 3)),
        )

    def replace(self, **changes) -> "Params":
        """Return a copy with some fields changed (used for audit sweeps)."""
        return dataclasses.replace(self, **changes)

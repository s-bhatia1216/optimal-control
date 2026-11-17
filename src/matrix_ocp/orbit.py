"""Two-body Kepler timeline for one MATRIX orbit.

Everything about the orbit that the control problem needs is a function of time
over one period: the radius r(t), and four indicator windows built from it or
from the config:

    ROI      : r(t) > 6 R_E            (science region, FDR Sec 1.1)
    eclipse  : fixed-length interval centered on a configurable orbit fraction
               (default apogee, the tail-season worst case, FDR Table 47)
    contact  : fixed-length interval centered on perigee (FDR Sec 5.7.10)
    sunlit   : complement of eclipse

Time convention: t = 0 at PERIGEE, t in [0, T). The contact window therefore
wraps around the period boundary ([0, 0.5 h] and [T - 0.5 h, T] for a 1 h
window); all window tests use circular (mod-T) distance so the wrap is handled
uniformly. The problem has periodic boundary conditions, so the choice of t = 0
is only a plotting convention.

No STK: r(t) comes from Kepler's equation M = E - e sin E solved by Newton's
method, which is exact for the two-body problem.
"""

from __future__ import annotations

import numpy as np

from .params import Params


def solve_kepler(M: np.ndarray, e: float, tol: float = 1e-13, max_iter: int = 60) -> np.ndarray:
    """Solve Kepler's equation E - e*sin(E) = M for the eccentric anomaly E.

    Newton's method with the standard starter E0 = M + e*sin(M); converges in a
    handful of iterations even for e ~ 0.83 (this orbit).
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    M = np.mod(M, 2.0 * np.pi)
    E = M + e * np.sin(M)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


class OrbitTimeline:
    """Precomputes the orbit geometry and exposes window/indicator functions.

    All public time arguments are hours; indicator methods accept scalars or
    arrays and are periodic in the orbit period ``T_h``.
    """

    def __init__(self, p: Params):
        self.p = p
        R_E = p.R_E_km

        # Geometry: semi-major axis and eccentricity from perigee/apogee radii.
        self.a_km = 0.5 * (p.r_p_RE + p.r_a_RE) * R_E
        self.e = (p.r_a_RE - p.r_p_RE) / (p.r_a_RE + p.r_p_RE)

        # Period from Kepler's third law. The FDR (Table 46) states T = 24 h for
        # Group 1; we recompute it and a test asserts they agree to < 0.1%.
        T_s = 2.0 * np.pi * np.sqrt(self.a_km**3 / p.mu_km3_s2)
        self.T_h = T_s / 3600.0

        # ROI entry/exit: solve r = roi_radius in closed form.
        # r = a(1 - e cos E)  =>  cos E* = (1 - r/a)/e, then M* = E* - e sin E*
        # and t* = (M*/2pi) T. By symmetry the ROI is (t_entry, T - t_entry).
        r_roi_km = p.roi_radius_RE * R_E
        cosE = (1.0 - r_roi_km / self.a_km) / self.e
        E_star = np.arccos(np.clip(cosE, -1.0, 1.0))
        M_star = E_star - self.e * np.sin(E_star)
        self.t_roi_entry_h = float(M_star / (2.0 * np.pi) * self.T_h)
        self.t_roi_exit_h = self.T_h - self.t_roi_entry_h
        self.roi_duration_h = self.t_roi_exit_h - self.t_roi_entry_h

        # Fixed windows from the config (centers as orbit fractions).
        self.t_eclipse_center_h = p.eclipse_center_frac * self.T_h
        self.t_contact_center_h = p.contact_center_frac * self.T_h

    # ------------------------------------------------------------------
    def radius_RE(self, t_h: np.ndarray) -> np.ndarray:
        """Orbit radius r(t) in Earth radii; t in hours, periodic."""
        M = 2.0 * np.pi * np.mod(np.asarray(t_h, dtype=float), self.T_h) / self.T_h
        E = solve_kepler(M, self.e)
        r_km = self.a_km * (1.0 - self.e * np.cos(E))
        out = r_km / self.p.R_E_km
        return float(out[0]) if np.ndim(t_h) == 0 else out

    def _circ_dist_h(self, t_h, center_h):
        """Shortest periodic distance |t - center| on the orbit circle [h]."""
        t = np.mod(np.asarray(t_h, dtype=float), self.T_h)
        d = np.abs(np.mod(t - center_h + 0.5 * self.T_h, self.T_h) - 0.5 * self.T_h)
        return d

    # --- indicator functions (return float arrays of 0.0 / 1.0) -----------
    def in_roi(self, t_h) -> np.ndarray:
        d = self._circ_dist_h(t_h, 0.5 * self.T_h)  # distance from apogee
        # ROI is symmetric about apogee: within (T/2 - t_entry) of apogee.
        return (d <= (0.5 * self.T_h - self.t_roi_entry_h)).astype(float)

    def in_eclipse(self, t_h) -> np.ndarray:
        d = self._circ_dist_h(t_h, self.t_eclipse_center_h)
        return (d <= 0.5 * self.p.eclipse_duration_h).astype(float)

    def in_contact(self, t_h) -> np.ndarray:
        d = self._circ_dist_h(t_h, self.t_contact_center_h)
        return (d <= 0.5 * self.p.contact_duration_h).astype(float)

    def sunlit(self, t_h) -> np.ndarray:
        return 1.0 - self.in_eclipse(t_h)

    # --- time-varying model inputs ----------------------------------------
    def P_in(self, t_h) -> np.ndarray:
        """Available solar power [W]: full array output when sunlit, else 0."""
        return self.p.P_in * self.sunlit(t_h)

    def P_0(self, t_h) -> np.ndarray:
        """Uncontrollable bus load [W]: higher in eclipse (heaters on)."""
        ecl = self.in_eclipse(t_h)
        return self.p.P0_day * (1.0 - ecl) + self.p.P0_eclipse * ecl

    def w(self, t_h) -> np.ndarray:
        """Science value weight (pluggable; selected by ``Params.w_mode``).

        "uniform": the ROI indicator (all ROI science equally valuable).
        "apogee":  indicator * r(t)/r_a, so deep-tail science is worth more.
                   Used to break the tie on singular arcs, where uniform w
                   makes the placement of rationed science non-unique.
        """
        roi = self.in_roi(t_h)
        if self.p.w_mode == "apogee":
            return roi * self.radius_RE(t_h) / self.p.r_a_RE
        return roi

    # --- interval-averaged inputs for the direct method --------------------
    # The dynamics are control-affine and state-independent, so propagating
    # states with per-interval AVERAGES of the exogenous inputs is exact for
    # piecewise-constant controls. Averaging (instead of midpoint sampling)
    # removes the O(h) window-quantization error: e.g. the grid's downlink
    # capacity sum(c_k h) r_c equals the true r_c * contact_duration exactly.

    def _window_segments(self, center_h: float, halfwidth_h: float):
        """A periodic window as 1-2 plain [lo, hi] segments inside [0, T]."""
        T = self.T_h
        lo, hi = center_h - halfwidth_h, center_h + halfwidth_h
        if lo < 0.0:
            return [(0.0, hi), (T + lo, T)]
        if hi > T:
            return [(lo, T), (0.0, hi - T)]
        return [(lo, hi)]

    def _avg_over_intervals(self, t_edges: np.ndarray, segments) -> np.ndarray:
        """Exact per-interval overlap fraction with a union of segments."""
        a, b = t_edges[:-1], t_edges[1:]
        out = np.zeros(len(a))
        for (lo, hi) in segments:
            out += np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None)
        return out / (b - a)

    def window_edge_times(self) -> np.ndarray:
        """All window boundary times in (0, T): ROI, eclipse, contact edges.

        The direct method inserts these into its mesh so that every interval
        lies purely inside or outside each window; with state-independent
        affine dynamics the transcription is then exact for piecewise-constant
        controls (no O(h) window quantization, no intra-interval sun/eclipse
        mixing).
        """
        times = [self.t_roi_entry_h, self.t_roi_exit_h]
        for center, half in [
            (self.t_eclipse_center_h, 0.5 * self.p.eclipse_duration_h),
            (self.t_contact_center_h, 0.5 * self.p.contact_duration_h),
        ]:
            for lo, hi in self._window_segments(center, half):
                times += [lo, hi]
        t = np.array(times)
        return np.unique(t[(t > 1e-9) & (t < self.T_h - 1e-9)])

    def interval_inputs(self, t_edges: np.ndarray, w_subsamples: int = 64) -> dict:
        """Exogenous inputs averaged over each interval of a mesh.

        Indicators (ROI, eclipse, contact) use exact segment overlap; the
        science weight w (smooth inside the ROI for the apogee mode) uses
        midpoint-rule subsampling with ``w_subsamples`` points per interval.
        """
        p = self.p
        roi_f = self._avg_over_intervals(
            t_edges, self._window_segments(0.5 * self.T_h,
                                           0.5 * self.T_h - self.t_roi_entry_h))
        ecl_f = self._avg_over_intervals(
            t_edges, self._window_segments(self.t_eclipse_center_h,
                                           0.5 * p.eclipse_duration_h))
        con_f = self._avg_over_intervals(
            t_edges, self._window_segments(self.t_contact_center_h,
                                           0.5 * p.contact_duration_h))
        # Subsampled average of w(t); supports non-uniform meshes. On a mesh
        # snapped to the window edges this is exact up to quadrature error of
        # the smooth r(t)/r_a factor (midpoint rule, 64 points per interval).
        a = t_edges[:-1]
        widths = np.diff(t_edges)
        offs = (np.arange(w_subsamples) + 0.5) / w_subsamples
        t_sub = a[:, None] + offs[None, :] * widths[:, None]
        w_f = self.w(t_sub.ravel()).reshape(len(a), -1).mean(axis=1)
        return {
            "roi_f": roi_f,
            "eclipse_f": ecl_f,
            "contact_f": con_f,
            "w_f": w_f,
            "P_in_W": p.P_in * (1.0 - ecl_f),
            "P_0_W": p.P0_day * (1.0 - ecl_f) + p.P0_eclipse * ecl_f,
        }

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Key timeline numbers for the audit report."""
        return {
            "period_h": self.T_h,
            "eccentricity": self.e,
            "semi_major_axis_km": self.a_km,
            "roi_entry_h": self.t_roi_entry_h,
            "roi_exit_h": self.t_roi_exit_h,
            "roi_duration_h": self.roi_duration_h,
            "roi_fraction": self.roi_duration_h / self.T_h,
            "eclipse_window_h": [
                self.t_eclipse_center_h - 0.5 * self.p.eclipse_duration_h,
                self.t_eclipse_center_h + 0.5 * self.p.eclipse_duration_h,
            ],
            "contact_duration_h": self.p.contact_duration_h,
            "sunlit_duration_h": self.T_h - self.p.eclipse_duration_h,
            "science_window_baseline_h": self.roi_duration_h - self.p.eclipse_duration_h,
        }

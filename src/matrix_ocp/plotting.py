"""Shared plotting style and the Phase-1 figures.

Style notes (following the course of action in the dataviz guidance):
  - colorblind-safe categorical palette, hues assigned in fixed order;
  - window shading uses light tints (semantic, not series, colors);
  - one y-axis per panel, recessive grid, axes labeled with units;
  - every figure saved as both vector PDF and PNG into results/<case>/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .orbit import OrbitTimeline
from .params import Params

# Categorical palette (validated colorblind-safe set, fixed assignment order).
C_BLUE = "#2a78d6"    # battery energy E
C_ORANGE = "#eb6834"  # science control u_s
C_AQUA = "#1baf7a"    # transmit control u_c
C_YELLOW = "#eda100"  # harvest control u_h
C_MAGENTA = "#e87ba4"
C_VIOLET = "#6250d6"  # stored data D
C_RED = "#e34948"     # bound violations / E_min line

# Window shading (semantic tints, kept light so series stay dominant).
SHADE_ROI = dict(color="#2a78d6", alpha=0.10, lw=0)
SHADE_ECLIPSE = dict(color="#52514e", alpha=0.22, lw=0)
SHADE_CONTACT = dict(color="#1baf7a", alpha=0.25, lw=0)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "legend.frameon": False,
})


def _shade_windows(ax, tl: OrbitTimeline, roi=True, eclipse=True, contact=True):
    """Shade ROI / eclipse / contact windows on an axis (handles the wrap)."""
    T = tl.T_h
    if roi:
        ax.axvspan(tl.t_roi_entry_h, tl.t_roi_exit_h, **SHADE_ROI)
    if eclipse:
        c = tl.t_eclipse_center_h
        half = 0.5 * tl.p.eclipse_duration_h
        ax.axvspan(max(c - half, 0.0), min(c + half, T), **SHADE_ECLIPSE)
    if contact:
        c = tl.t_contact_center_h
        half = 0.5 * tl.p.contact_duration_h
        lo, hi = c - half, c + half
        # contact is centered on perigee (t = 0), so it wraps mod T
        if lo < 0:
            ax.axvspan(0.0, hi, **SHADE_CONTACT)
            ax.axvspan(T + lo, T, **SHADE_CONTACT)
        elif hi > T:
            ax.axvspan(lo, T, **SHADE_CONTACT)
            ax.axvspan(0.0, hi - T, **SHADE_CONTACT)
        else:
            ax.axvspan(lo, hi, **SHADE_CONTACT)


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)


def plot_timeline(p: Params, tl: OrbitTimeline, outdir: Path):
    """Figure 1: r(t) with ROI / eclipse / contact windows, and model inputs."""
    t = np.linspace(0.0, tl.T_h, 2001)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    ax1.plot(t, tl.radius_RE(t), color=C_BLUE, lw=2)
    ax1.axhline(p.roi_radius_RE, color="#52514e", lw=1, ls="--")
    ax1.text(0.35, p.roi_radius_RE + 0.25, f"ROI boundary ({p.roi_radius_RE:.0f} $R_E$)",
             color="#52514e", fontsize=8.5)
    _shade_windows(ax1, tl)
    ax1.set_ylabel("orbit radius $r$ [$R_E$]")
    ax1.set_title(
        f"MATRIX Group-1 orbit timeline (T = {tl.T_h:.2f} h, e = {tl.e:.3f});"
        " blue: ROI, gray: eclipse, green: contact"
    )
    # Direct labels for the shaded windows (legend-free semantics).
    ax1.text(tl.t_eclipse_center_h, 1.3, "eclipse", ha="center", fontsize=8.5,
             color="#30302e")
    ax1.text(tl.t_roi_entry_h + 1.2, 11.0, "ROI ($r>6\\,R_E$)", fontsize=8.5,
             color=C_BLUE)
    ax1.text(1.15, 10.4, "ground\ncontact", fontsize=8.5, color="#117a55")
    ax1.annotate("", xy=(0.4, 9.6), xytext=(1.5, 10.2),
                 arrowprops=dict(arrowstyle="->", color="#117a55", lw=0.9))

    ax2.plot(t, tl.P_in(t), color=C_YELLOW, lw=2, label="$P_{in}(t)$ available solar")
    ax2.plot(t, tl.P_0(t), color=C_RED, lw=2, label="$P_0(t)$ uncontrollable load")
    _shade_windows(ax2, tl, roi=False)
    ax2.set_ylabel("power [W]")
    ax2.set_xlabel("time since perigee [h]")
    ax2.set_xlim(0, tl.T_h)
    ax2.legend(loc="center left", fontsize=8.5)

    _save(fig, outdir, "fig1_orbit_timeline")


def plot_baseline(p: Params, tl: OrbitTimeline, traj: dict, outdir: Path):
    """Figure 2: baseline states and controls over the converged periodic orbit."""
    te, tm = traj["t_edges_h"], traj["t_mid_h"]
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.0]})
    ax_E, ax_D, ax_u, ax_P = axes
    for ax in axes:
        _shade_windows(ax, tl, roi=False)

    # Battery energy with bounds.
    ax_E.plot(te, traj["E_Wh"], color=C_BLUE, lw=2)
    ax_E.axhline(p.E_max, color="#52514e", lw=1, ls="--")
    ax_E.axhline(p.E_min, color=C_RED, lw=1, ls="--")
    ax_E.text(0.3, p.E_max - 40, "$E_{max}$", fontsize=8.5, color="#52514e")
    ax_E.text(0.3, p.E_min + 15, "$E_{min}$ (DoD floor)", fontsize=8.5, color=C_RED)
    ax_E.set_ylabel("battery $E$ [Wh]")
    ax_E.set_title(f"FDR baseline policy, case '{p.case_name}' "
                   "(gray: eclipse, green: contact)")

    # Stored data.
    ax_D.plot(te, traj["D_Mbit"], color=C_VIOLET, lw=2)
    ax_D.set_ylabel("stored data $D$ [Mbit]")

    # Controls (step plots; piecewise constant).
    ax_u.step(tm, traj["u_s"], where="mid", color=C_ORANGE, lw=1.8, label="$u_s$ science")
    ax_u.step(tm, traj["u_c"], where="mid", color=C_AQUA, lw=1.8, label="$u_c$ transmit")
    ax_u.step(tm, traj["u_h"], where="mid", color=C_YELLOW, lw=1.8, label="$u_h$ harvest")
    ax_u.set_ylabel("controls [-]")
    ax_u.set_ylim(-0.05, 1.1)
    ax_u.legend(loc="center right", ncol=1, fontsize=8.5)

    # Net electrical power into the battery.
    net = traj["u_h"] * traj["P_in_W"] - traj["P_0_W"] \
        - p.P_s * traj["u_s"] - p.P_c * traj["u_c"]
    ax_P.plot(tm, net, color=C_MAGENTA, lw=1.8)
    ax_P.axhline(0.0, color="#52514e", lw=0.8)
    ax_P.set_ylabel("net power $\\dot E$ [W]")
    ax_P.set_xlabel("time since perigee [h]")
    ax_P.set_xlim(0, tl.T_h)

    _save(fig, outdir, "fig2_baseline_policy")

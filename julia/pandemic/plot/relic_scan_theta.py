#! /usr/bin/env python3
"""
Contours of the coupling y in the (m_N, sin^2 2theta) plane for which
Omega h^2 = 0.12, from the scan in src/run_relic_scan_theta_z.jl
(tmp/relic_scan_theta/m_N_*keV.csv), in the style of Fig. 3 of the paper.

Only roots on the freeze-in branch are drawn: for each (m_N, y) the smallest
converged sin^2 2theta at which Omega h^2 rises through the target. Contour
lines are broken where no such root was found.

Run from julia/pandemic:
    conda run -n pandemic python plot/relic_scan_theta.py [scan_dir] [output_path_without_extension]
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SCAN_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tmp", "relic_scan_theta")
XRAY_FILE = os.path.join(ROOT, "..", "..", "xray_constraints", "overall_constraint.dat")
DW_FILE = os.path.join(ROOT, "data", "dw", "0612182_dw_fig_4.dat")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures", "relic_contours_y")

OMEGA_TARGET = 0.12
M_A_OVER_M_N = 2.5


def load_scan():
    rows = []
    for path in sorted(glob.glob(os.path.join(SCAN_DIR, "m_N_*keV.csv"))):
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        rows.extend(np.atleast_1d(data).tolist())
    if not rows:
        raise SystemExit(f"no scan results in {SCAN_DIR}")
    names = ["m_N", "y", "sin2_2theta", "omega_h2", "converged", "plateau_ok", "retcode", "branch", "slope", "n_failed"]
    return {n: np.array([r[i] for r in rows]) for i, n in enumerate(names)}


def freeze_in_roots(scan):
    """(m_N [keV], y) -> smallest converged sin^2 2theta with dOmega/dtheta > 0."""
    ok = (scan["converged"] == True) & (scan["plateau_ok"] == True) & (scan["slope"] > 0)  # noqa: E712
    roots = {}
    for m, y, s in zip(scan["m_N"][ok], scan["y"][ok], scan["sin2_2theta"][ok]):
        key = (round(m * 1e6, 10), y)
        roots[key] = min(s, roots.get(key, np.inf))
    return roots


def dodelson_widrow_line(m_keV):
    """sin^2 2theta for which DW production alone gives Omega h^2 = 0.12
    (Omega h^2 = 0.11 C_e(m) (sin 2theta m / 2 / 0.1 eV)^2, hep-ph/0612182)."""
    table = np.loadtxt(DW_FILE, comments="#")
    m_GeV = m_keV * 1e-6
    c_e = np.exp(np.interp(np.log(m_GeV), np.log(table[:, 0]), np.log(table[:, 1])))
    return 4 * OMEGA_TARGET / (0.11 * c_e * (m_GeV * 1e10)**2)


def main():
    scan = load_scan()
    roots = freeze_in_roots(scan)
    masses = np.array(sorted({k[0] for k in roots}))
    ys = np.array(sorted({k[1] for k in roots}))

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    m_lo = min(masses.min(), 1e0)
    m_hi = max(masses.max(), 3e2)
    s_lo, s_hi = 1e-18, 1e-8

    # Constraints for orientation, as in Fig. 3.
    m_band = np.logspace(np.log10(m_lo), np.log10(m_hi), 400)
    s_dw = dodelson_widrow_line(m_band)
    ax.fill_between(m_band, s_dw, s_hi, color="tab:blue", alpha=0.25, lw=0)
    ax.plot(m_band, s_dw, color="tab:blue", ls=":", lw=1)
    if os.path.exists(XRAY_FILE):
        xray = np.loadtxt(XRAY_FILE)
        m_x, s_x = xray[:, 0] * 1e6, xray[:, 1]
        sel = (m_x >= m_lo) & (m_x <= m_hi)
        ax.fill_between(m_x[sel], np.minimum(s_x[sel], s_hi), s_hi, color="0.6", alpha=0.6, lw=0)
        ax.plot(m_x[sel], s_x[sel], color="k", lw=1)

    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(ys)))
    for y, c in zip(ys, colors):
        m_line = np.array([m for m in masses if (m, y) in roots])
        s_line = np.array([roots[(m, y)] for m in m_line])
        if len(m_line) == 0:
            continue
        # Break the line where a mass in the grid has no root for this y.
        idx = np.searchsorted(masses, m_line)
        segments = np.split(np.arange(len(m_line)), np.where(np.diff(idx) > 1)[0] + 1)
        for seg in segments:
            ax.plot(m_line[seg], s_line[seg], color=c, lw=1.3, marker="o", ms=2.5)
        exp10 = np.log10(y)
        label = rf"$10^{{{exp10:.0f}}}$" if abs(exp10 - round(exp10)) < 1e-6 else rf"$10^{{{exp10:.1f}}}$"
        ax.text(m_line[0] * 0.93, s_line[0], label, color=c, fontsize=7, ha="right", va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(m_lo * 0.6, m_hi)
    ax.set_ylim(s_lo, s_hi)
    ax.set_xlabel(r"$m_N$ [keV]")
    ax.set_ylabel(r"$\sin^2(2\theta)$")
    ax.text(0.97, 0.97, rf"$m_{{A'}} = {M_A_OVER_M_N}\,m_N$", transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    handles = [
        Line2D([], [], color="0.3", lw=1.3, marker="o", ms=2.5, label=r"$\Omega h^2 = 0.12$ at fixed $y$"),
        Line2D([], [], color="tab:blue", ls=":", label="Dodelson-Widrow"),
        Line2D([], [], color="k", lw=1, label="X-rays"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=200)
    print(f"wrote {OUT}.pdf/.png with {len(roots)} points, {len(masses)} masses, {len(ys)} values of y")


if __name__ == "__main__":
    main()

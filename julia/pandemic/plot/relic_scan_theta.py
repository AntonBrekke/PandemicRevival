#! /usr/bin/env python3
"""
Contours of the coupling y (= g in the paper) in the (m_N, sin^2 2theta) plane
for which Omega h^2 = 0.12, from the scan in src/run_relic_scan_theta_z.jl
(tmp/relic_scan_theta/m_N_*keV.csv). Styled like the money plot of the paper
(code/sterile_res/plotter_3.py): Dodelson-Widrow band, X-ray limits and
projections, and the contours labelled on the plot.

Only roots on the freeze-in branch are drawn: for each (m_N, y) the smallest
converged sin^2 2theta at which Omega h^2 rises through the target. The lines
are monotone (PCHIP) interpolations in log-log between the scanned masses and
are broken where no root was found for a mass in the grid.

Run from julia/pandemic:
    conda run -n pandemic python plot/relic_scan_theta.py [scan_dir] [output_path_without_extension] [--debug]

--debug marks the scanned roots on the lines and writes <output>_debug.pdf/.png,
so the production figure is not overwritten.
"""

import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
XRAY_DIR = os.path.join(ROOT, "..", "..", "xray_constraints")
DW_DIR = os.path.join(ROOT, "data", "dw")

OMEGA_TARGET = 0.12
M_A_OVER_M_N = 2.5

COLUMNWIDTH = 418.25368  # pt, \showthe\textwidth in LaTeX, as for the paper figure
M_LIM = (1.0, 300.0)     # keV
S_LIM = (1e-18, 1e-8)

C_DW = "#83781B"
C_DW_BAND = "#EAE299"
C_OVERPROD = "#92CBE2"   # skyblue at alpha 0.8 on white
C_OVERPROD_TEXT = "#155D7A"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan_dir", nargs="?", default=os.path.join(ROOT, "tmp", "relic_scan_theta"))
    p.add_argument("out", nargs="?", default=os.path.join(ROOT, "figures", "relic_contours_y"),
                   help="output path without extension")
    p.add_argument("--debug", action="store_true", help="mark the scanned roots on the contour lines")
    return p.parse_args()


def set_style():
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
    plt.rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.titlesize": 10, "font.size": 10})


def get_figsize(columnwidth, wf=1.0, hf=(5.**0.5 - 1.0) / 2.0):
    """[width, height] in inches for a fraction wf of the LaTeX column width
    [pt] and aspect ratio hf (golden ratio by default)."""
    fig_width = columnwidth * wf / 72.27
    return [fig_width, fig_width * hf]


def load_scan(scan_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(scan_dir, "m_N_*keV.csv"))):
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        rows.extend(np.atleast_1d(data).tolist())
    if not rows:
        raise SystemExit(f"no scan results in {scan_dir}")
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


def y_label(y, first):
    exp10 = np.log10(y)
    exp_str = f"{exp10:.0f}" if abs(exp10 - round(exp10)) < 1e-6 else f"{exp10:.1f}"
    return (r"$g = " if first else "$") + rf"10^{{{exp_str}}}$"


def plot_dodelson_widrow(ax):
    """Band of Omega h^2 = 0.12 from DW production (fig. 5 of hep-ph/0612182),
    with the overproduction region above it."""
    def load(name, omega_ref):
        d = np.loadtxt(os.path.join(DW_DIR, name), skiprows=2)
        return 1e6 * d[:, 0], (OMEGA_TARGET / omega_ref) * d[:, 1] * (1e-6 / d[:, 0])**2

    m_mid, s_mid = load("0612182_dw_fig_5.dat", 0.11)
    m_up, s_up = load("0612182_dw_fig_5_up.dat", 0.105)
    m_low, s_low = load("0612182_dw_fig_5_low.dat", 0.105)
    ax.plot(m_mid, s_mid, color=C_DW, ls="--", zorder=1)
    ax.plot(m_low, s_low, color=C_DW, ls=":", zorder=1)
    ax.plot(m_up, s_up, color=C_DW, ls=":", zorder=1)
    ax.fill(np.concatenate((m_low, m_up[::-1])), np.concatenate((s_low, s_up[::-1])),
            color=C_DW_BAND, lw=0, zorder=0)
    for m, s in ((m_low, s_low), (m_up, s_up)):
        ax.fill_between(m, s, 1.0, color=C_OVERPROD, lw=0, zorder=-1)
    ax.text(10**1.5, 10**-10.25, "Dodelson-Widrow", color=C_DW, rotation=-22, ha="center")
    ax.text(10**1.6, 10**-10.25, "overproduction", color=C_OVERPROD_TEXT, rotation=-24)


def plot_xrays(ax):
    xray = np.loadtxt(os.path.join(XRAY_DIR, "overall_constraint.dat"))
    m, s = 1e6 * xray[:, 0], xray[:, 1]
    ax.fill_between(m, s, 1.0, color="white", lw=0, zorder=-3)
    ax.fill_between(m, s, 1.0, color="black", alpha=0.25, lw=0, zorder=-3)
    ax.plot(m, s, color="black", lw=1.3, zorder=-2)
    ax.text(10**1.45, 1e-13, "X-rays", color="black")

    for name, ls in (("Athena_projection_2103.13242.dat", "-."),
                     ("eROSITA_projection_2103.13241.dat", "--"),
                     ("eXTP_projection_2001.07014.dat", ":")):
        proj = np.loadtxt(os.path.join(XRAY_DIR, name), skiprows=2)
        ax.plot(1e6 * proj[:, 0], proj[:, 1], color="black", lw=1.3, ls=ls, zorder=1)
    ax.text(10**0.3, 10**-10.69, "eROSITA", color="black", rotation=-45)
    ax.text(10**0.95, 10**-13.35, "Athena", color="black", rotation=-15)
    ax.text(10**1.04, 1e-15, "eXTP", color="black")


def plot_contours(ax, roots, debug):
    masses = np.array(sorted({k[0] for k in roots}))
    ys = np.array(sorted({k[1] for k in roots}))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(ys)))
    for i, (y, c) in enumerate(zip(ys, colors)):
        m_line = np.array([m for m in masses if (m, y) in roots])
        s_line = np.array([roots[(m, y)] for m in m_line])
        if len(m_line) == 0:
            continue
        # Break the line where a mass in the grid has no root for this y.
        idx = np.searchsorted(masses, m_line)
        segments = np.split(np.arange(len(m_line)), np.where(np.diff(idx) > 1)[0] + 1)
        for seg in segments:
            lm, ls = np.log(m_line[seg]), np.log(s_line[seg])
            if len(seg) > 1:
                lm_fine = np.linspace(lm[0], lm[-1], 200)
                ax.plot(np.exp(lm_fine), np.exp(PchipInterpolator(lm, ls)(lm_fine)), color=c, lw=1.0, zorder=-1)
            if debug or len(seg) == 1:
                ax.plot(m_line[seg], s_line[seg], ls="none", marker="o", ms=2.5, color=c, zorder=2)
        # Label at the left edge, just above the start of the line, as in the paper figure.
        ax.text(M_LIM[0] * 1.05, s_line[0] * 1.6, y_label(y, first=(i == 0)), color=c,
                ha="left", va="bottom", zorder=-1)


def main():
    args = parse_args()
    roots = freeze_in_roots(load_scan(args.scan_dir))
    out = args.out + ("_debug" if args.debug else "")

    set_style()
    fig = plt.figure(figsize=get_figsize(COLUMNWIDTH, wf=1.0, hf=0.9), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(axis="both", which="both", direction="in", width=0.5)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    plot_dodelson_widrow(ax)
    plot_xrays(ax)
    plot_contours(ax, roots, args.debug)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*M_LIM)
    ax.set_ylim(*S_LIM)
    # Label every decade, as in the paper figure.
    for axis, (lo, hi) in ((ax.xaxis, M_LIM), (ax.yaxis, S_LIM)):
        decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
        axis.set_major_locator(FixedLocator(10**decades))
        axis.set_minor_locator(FixedLocator([k * 10**d for d in decades for k in range(2, 10)]))
        axis.set_major_formatter(LogFormatterMathtext())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$m_{N_1}\;\;[\mathrm{keV}]$")
    ax.set_ylabel(r"$\sin^2 (2 \theta_1)$")
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=1, edgecolor="0.8")
    ax.text(0.97, 0.96, rf"$m_{{A'}} = {M_A_OVER_M_N}\,m_{{N_1}}$", transform=ax.transAxes,
            ha="right", va="top", bbox=props)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out + ".pdf")
    if shutil.which("dvipng"):
        fig.savefig(out + ".png", dpi=300)
    elif shutil.which("pdftoppm"):  # usetex needs dvipng for raster output
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", out + ".pdf", out], check=True)
    print(f"wrote {out}.pdf/.png with {len(roots)} points, "
          f"{len({k[0] for k in roots})} masses, {len({k[1] for k in roots})} values of y")


if __name__ == "__main__":
    main()

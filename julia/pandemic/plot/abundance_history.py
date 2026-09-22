#! /usr/bin/env python3
"""
Benchmark-point figure in the style of the paper (Fig. "evolution", made by
code/sterile_test/pandemic_rate_combined.py) for histories written by
src/run_abundance_history.jl:
    top left:     abundances m n/s of N_1 + N_2 and A'
    bottom left:  temperature ratio T_N/T_nu
    right:        Hubble rate and collision rates |C|/n
The history starts at the Dodelson-Widrow initial condition, where the dark
sector is taken to be thermalized, so the figure starts there.

Run from julia/pandemic, for one benchmark point or for every history in a
directory:
    conda run -n pandemic python plot/abundance_history.py [history.csv | history_dir] [figure_dir] [--debug]

--debug marks the points of the ODE solution on every curve and writes
<name>_debug.pdf, so the production figure is not overwritten.
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

M_A_OVER_M_N = 2.5
MY_RELIC = 4.354e-10 * 1e6  # keV, omega_d0 * rho_crit0_h2 / s0, see constant_functions.jl
NAME = re.compile(r"m_N_(?P<m>[^_]+)keV_y_(?P<y>[^_]+)_sin22th_(?P<s>[^_]+)\.csv$")

COLUMNWIDTH = 426.39256  # pt, \showthe\textwidth in LaTeX, as for the paper figure

C_N = "#7bc043"       # green
C_A = "#f37736"       # orange
C_RELIC = "0.55"
C_TEMP = "0.4"
C_H = "crimson"
C_A_N2NU = "#1aa7ec"  # sky blue
C_AA_NN = "#4adede"   # turquoise
C_TOTAL = "0.3"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("history", nargs="?", default=os.path.join(ROOT, "tmp", "abundance_history"),
                   help="history CSV of one benchmark point, or a directory of them")
    p.add_argument("out_dir", nargs="?", default=os.path.join(ROOT, "figures", "abundance_history"))
    p.add_argument("--debug", action="store_true", help="mark the points of the ODE solution")
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


def latex_num(v):
    """4.151e-15 -> 4.151\\cdot 10^{-15}"""
    mant, exp = f"{v:.3e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp = int(exp)
    if exp == 0:
        return mant
    return (f"{mant}\\cdot " if mant != "1" else "") + f"10^{{{exp}}}"


def load_history(path):
    """Columns of `transform_sol_z` by name, see src/run_abundance_history.jl."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {k: d[k] for k in d.dtype.names}


def style_axis(ax):
    ax.tick_params(axis="both", which="both", direction="in", width=0.5)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def decade_ticks(axis, lo, hi, label_every=1, skip=()):
    """A tick at every decade in [lo, hi], labelled every `label_every` decades
    except for the decades in `skip`."""
    decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1).astype(int)
    labelled = {d for d in decades if d % label_every == 0 and d not in skip}
    axis.set_major_locator(FixedLocator(10.0**decades))
    axis.set_minor_locator(FixedLocator([k * 10.0**d for d in decades for k in range(2, 10)]))
    axis.set_major_formatter(FuncFormatter(
        lambda v, _: rf"$10^{{{round(np.log10(v))}}}$" if round(np.log10(v)) in labelled else ""))
    axis.set_minor_formatter(NullFormatter())


def positive(x, y):
    """Points with finite, positive y (log axes)."""
    ok = np.isfinite(y) & (y > 0)
    return x[ok], y[ok]


def curve(ax, x, y, debug, **kw):
    x, y = positive(x, y)
    ax.plot(x, y, **kw)
    if debug:
        ax.plot(x, y, ls="none", marker="o", ms=1.5, color=kw.get("color"), zorder=kw.get("zorder", 2))
    return x, y


def label_curves(ax, curves, xlim, ylim):
    """Label curves in their own colour, just above or below them, where the
    label's box is furthest (in log y) from the other curves and from the
    labels already placed. `curves` is a list of (x, y, text, color); curves
    with text None are only obstacles."""
    lx = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)
    ly_lo, ly_hi = np.log10(ylim[0]), np.log10(ylim[1])
    half_w = int(0.06 * len(lx))              # half width of a label, in grid points
    height = 0.07 * (ly_hi - ly_lo)          # height of a label, in decades
    pad = 0.015 * (ly_hi - ly_lo)

    def on_grid(x, y):
        if len(x) < 2:
            return np.full(lx.shape, np.nan)
        return np.interp(lx, np.log10(x), np.log10(y), left=np.nan, right=np.nan)

    def distance(v, lo, hi):
        return np.where(v > hi, v - hi, np.where(v < lo, lo - v, 0.0))

    grids = [on_grid(x, y) for x, y, _, _ in curves]
    placed = []  # (k, lo, hi) of the labels so far
    for i, (_, _, text, color) in enumerate(curves):
        if text is None:
            continue
        best = (0.0, None, None)
        for k in range(2 * half_w, len(lx) - 2 * half_w):
            sl = slice(k - half_w, k + half_w + 1)
            own = grids[i][sl]
            if np.isnan(own).any():
                continue
            others = np.concatenate([g[sl] for j, g in enumerate(grids) if j != i] or [np.array([])])
            others = others[~np.isnan(others)]
            for lo in (own.max() + pad, own.min() - pad - height):  # above, below
                hi = lo + height
                if lo < ly_lo or hi > ly_hi:
                    continue
                clearance = np.min(distance(others, lo, hi), initial=np.inf)
                for kp, lop, hip in placed:
                    if abs(k - kp) <= 2 * half_w:
                        clearance = min(clearance, max(lop - hi, lo - hip, 0.0))
                clearance = min(clearance, 3 * height)  # beyond this, prefer the first spot
                if clearance > best[0]:
                    best = (clearance, k, lo)
        if best[1] is not None:
            k, lo = best[1], best[2]
            placed.append((k, lo, lo + height))
            ax.text(10**lx[k], 10**(lo + 0.5 * height), text, color=color, ha="center", va="center")


def plot_history(path, out_dir, debug=False):
    match = NAME.search(os.path.basename(path))
    if match is None:
        print(f"skipping {path}: unexpected file name")
        return None
    m_keV, y, s = (float(match[k]) for k in ("m", "y", "s"))
    m_N = m_keV * 1e-6  # GeV
    h = load_history(path)
    x = h["x_nu"]
    xlim = (x[0], x[-1])

    fig = plt.figure(figsize=get_figsize(COLUMNWIDTH, wf=1.5, hf=0.5), dpi=150)
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0.0, wspace=0.0)
    ax_n = fig.add_subplot(grid[0, 0])
    ax_t = fig.add_subplot(grid[1, 0], sharex=ax_n)
    ax_r = fig.add_subplot(grid[:, 1])
    for ax in (ax_n, ax_t, ax_r):
        style_axis(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
    # The panels touch: drop the rate panel's x label at the seam if it is on the edge.
    for ax in (ax_n, ax_t):
        decade_ticks(ax.xaxis, *xlim)
    seam = np.log10(xlim[0])
    decade_ticks(ax_r.xaxis, *xlim, skip={round(seam)} if abs(seam - round(seam)) < 0.15 else set())
    plt.setp(ax_n.get_xticklabels(), visible=False)

    # Abundances m n/s [keV]; N1 and N2 have equal masses.
    my_N = 1e6 * m_N * (h["y_N1"] + h["y_N2"])
    my_A = 1e6 * M_A_OVER_M_N * m_N * h["y_A"]
    n_lim = (MY_RELIC * 1e-15, MY_RELIC * 1e2)
    ax_n.axhline(MY_RELIC, color=C_RELIC, ls="-.", zorder=-2)
    ax_n.text(xlim[0] * 2, MY_RELIC * 3, r"$\Omega_{N} h^2 = 0.12$", color=C_RELIC, va="bottom")
    xN, yN = curve(ax_n, x, my_N, debug, color=C_N, zorder=-1)
    xA, yA = curve(ax_n, x, my_A, debug, color=C_A, zorder=-1)
    ax_n.set_ylim(*n_lim)
    decade_ticks(ax_n.yaxis, *n_lim, label_every=2)
    label_curves(ax_n, [(xN, yN, r"$N_1 + N_2$", C_N), (xA, yA, r"$A'$", C_A),
                        (np.array(xlim), np.full(2, MY_RELIC), None, None)], xlim, n_lim)
    ax_n.set_ylabel(r"$m\, n / s\;\;[\mathrm{keV}]$")

    # Temperature ratio.
    ratio = x / h["x_N"]
    curve(ax_t, x, ratio, debug, color=C_TEMP, zorder=-1)
    t_lim = (np.nanmin(ratio) * 0.5, np.nanmax(ratio) * 2)
    ax_t.set_ylim(*t_lim)
    decade_ticks(ax_t.yaxis, *t_lim)
    ax_t.set_xlabel(r"$m_{N_1} / T_\nu$")
    ax_t.set_ylabel(r"$T_N/T_\nu$")

    # Rates [keV]: collision terms per dark-sector number density n = s Y_n.
    n = h["ent"] * h["y_n"]
    rates = [
        (1e6 * h["hubble"], r"$H$", C_H, "-"),
        (1e6 * np.abs(h["coll_A_N2nu"]) / n, r"$A' \leftrightarrow N_2 \nu$", C_A_N2NU, "-"),
        (1e6 * np.abs(h["coll_AA_NN"]) / n, r"$A'A' \leftrightarrow NN$", C_AA_NN, "-"),
        (1e6 * np.abs(h["coll_n"]) / n, r"total", C_TOTAL, "--"),
    ]
    r_max = max(np.nanmax(r[0]) for r in rates)
    r_lim = (np.nanmin(rates[0][0]) * 1e-4, r_max * 1e2)
    labelled = []
    for r, text, color, ls in rates:
        xr, yr = curve(ax_r, x, r, debug, color=color, ls=ls, zorder=-1, lw=1.0 if ls == "-" else 0.8)
        labelled.append((xr, yr, text, color))
    ax_r.set_ylim(*r_lim)
    decade_ticks(ax_r.yaxis, *r_lim, label_every=2)
    label_curves(ax_r, labelled, xlim, r_lim)
    ax_r.yaxis.set_label_position("right")
    ax_r.yaxis.set_ticks_position("both")
    ax_r.tick_params(axis="y", which="both", labelleft=False, labelright=True)
    ax_r.set_xlabel(r"$m_{N_1} / T_\nu$")
    ax_r.set_ylabel(r"$\mathrm{Rate}\;\;[\mathrm{keV}]$")

    fig.suptitle(fr"$m_{{N_1}}={m_keV:g}\ \mathrm{{keV}},\ m_{{A'}}={M_A_OVER_M_N}\, m_{{N_1}},\ "
                 fr"g={latex_num(y)},\ \sin^2(2\theta_1)={latex_num(s)}$")
    fig.tight_layout()
    stem = os.path.join(out_dir, os.path.basename(path)[:-len(".csv")] + ("_debug" if debug else ""))
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return stem + ".pdf"


def main():
    args = parse_args()
    if os.path.isdir(args.history):
        paths = sorted(glob.glob(os.path.join(args.history, "m_N_*keV_y_*_sin22th_*.csv")))
        if not paths:
            raise SystemExit(f"no abundance histories in {args.history}")
    else:
        paths = [args.history]

    set_style()
    os.makedirs(args.out_dir, exist_ok=True)
    for path in paths:
        out = plot_history(path, args.out_dir, args.debug)
        if out is not None:
            print(out)


if __name__ == "__main__":
    main()

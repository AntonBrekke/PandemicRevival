#! /usr/bin/env python3
"""
Contours of the coupling y (= g in the paper) in the (m_N, sin^2 2theta) plane
for which Omega h^2 = 0.12, from a scan directory written by plot/combined_scan_dir.py
(tmp/combined_scan: roots.csv and lyman_alpha.csv). Styled like the money plot of the paper
(code/sterile_res/plotter_3.py): Dodelson-Widrow band, the X-ray limit and the contours
labelled on the plot. The X-ray region is one line: the minimum over the current limits
(PandemicRevival/xray_constraints/plot_xray.py), rescaled by the N1 fraction of the dark matter
at each root (`scaled_xray_limit`), since only N1 decays to nu gamma. It is drawn over the
constraint bounds below. Projected sensitivities (eROSITA, Athena, eXTP, rescaled the same way)
only with --xray-proj, which writes <out>_xproj.pdf; --bp marks benchmark points; --mA sets
m_A'/m_N (sigma/m and the label) for scans other than 2.5.

The roots are those of the freeze-in branch (the root searches only follow it). The
lines are monotone (PCHIP) interpolations in log-log between the scanned masses and
are broken where no root was found for a mass in the grid. Only whole decades
of y are drawn, and only those at y >= LABEL_Y_MIN are labelled; a finer
sub-decade grid, if scanned, still feeds the constraint boundaries below,
which are what it is needed for.

Three constraints are shaded, each bounding the region of small m_N:

  Ly-alpha    free-streaming length outside the kinetically coupled phase,
              lambda_fs > 0.24 Mpc, and sound horizon r_s > 0.34 Mpc, remapped
              from the WDM limit m_WDM > 1.9 keV as in Bringmann et al.
              (2206.10630); see src/lyman_alpha.jl.
  self-int.   sigma/m > 1 cm^2/g for the N1/N2 mixture of each point (N2
              fraction and splitting from lyman_alpha.csv), with
              N1N1 -> N2N2, N2N2 -> N1N1 and N1N2 -> N1N2; see
              plot/self_interaction.py.

On every line of constant y the crossing of each limit is interpolated (in
log) between neighbouring roots and the crossings are joined. For sigma/m the
interpolation is exact, since sigma/m ~ m^-3 at fixed y and m_A'/m_N.

Run from julia/pandemic:
    conda run -n pandemic python plot/relic_scan_theta.py [scan_dir] [output_path_without_extension] [--debug]

--debug marks the scanned roots on the lines (open markers where a root is
excluded by one of the constraints) and writes <output>_debug.pdf, so the
production figure is not overwritten. --all-points marks every scanned root
instead, sub-decade couplings included (small dots coloured by g, on a colour
bar; open where excluded), and writes <output>_debug_all.pdf.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_interaction import sigma_over_m

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
XRAY_DIR = os.path.join(ROOT, "..", "..", "xray_constraints")
DW_DIR = os.path.join(ROOT, "data", "dw")

OMEGA_TARGET = 0.12
M_A_OVER_M_N = 2.5   # default of --mA

# Projected X-ray sensitivities (--xray-proj), as in Fig. 3 of 2206.10630: file in XRAY_DIR: label
XRAY_PROJECTIONS = {
    "eROSITA_projection_2103.13241.dat": "eROSITA",
    "Athena_projection_2103.13242.dat": "Athena",
    "eXTP_projection_2001.07014.dat": "eXTP",
}

# The abundance the digitized Dodelson-Widrow curves of hep-ph/0612182 Fig. 5
# were drawn for, the same for the central line and both band edges. Combining
# data/dw/0612182_dw_fig_4.dat (C_e) with 0612182_dw_fig_5.dat through the
# paper's Eq. (4.10), Omega h^2 = 0.11 C_e (theta m / 0.1 eV)^2, gives
# 0.1100 +- 0.0002 over 0.1 keV to 1 MeV, which is what the .dat headers say.
# Not the 0.105 of the paper's Eq. (4.11): that is the measured WMAP value the
# paper compares to, not the abundance Fig. 5 is drawn at.
OMEGA_DW_REF = 0.11

COLUMNWIDTH = 418.25368  # pt, \showthe\textwidth in LaTeX, as for the paper figure
M_LIM = (1.0, 300.0)     # keV
S_LIM = (1e-18, 1e-8)

C_DW = "#83781B"
C_DW_BAND = "#EAE299"
C_OVERPROD = "#92CBE2"   # skyblue at alpha 0.8 on white
C_OVERPROD_TEXT = "#155D7A"

# Limits of the three constraints, and the colours they are drawn in.
LAMBDA_FS_MAX_MPC = 0.24
R_S_MAX_MPC = 0.34
SIGMA_M_MAX = 1.0        # cm^2/g
C_LYA = "#D95F02"
C_RS = "#C9184A"   # the shading is drawn at alpha, so the line/label carry the colour
C_SI = "#A300CC"
BOUND_LABEL_SIZE = 9

# Contours below this coupling are drawn but not labelled: they only enter the
# frame at m_N ~ 3 keV, inside the Dodelson-Widrow band, where there is no room.
LABEL_Y_MIN = 1e-6

# Drawing order, bottom to top. The constraint bounds sit below the X-ray
# region, so that its opaque white underlay covers them, and below the
# Omega h^2 contour lines, so those stay readable over the shading.
Z_BOUND = {"lya": -3.5, "r_s": -3.7, "si": -3.9}
Z_LABEL = 2.5   # all on-plot text, above every shaded region

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "scan_dir",
        nargs="?",
        default=os.path.join(ROOT, "tmp", "combined_scan")
    )
    p.add_argument(
        "out",
        nargs="?",
        default=os.path.join(ROOT, "figures", "relic_contours"),
        help="output path without extension"
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="mark the scanned roots on the contour lines"
    )
    p.add_argument(
        "--all-points",
        action="store_true",
        help="mark every scanned root, all y, and write <out>_debug_all.pdf"
    )
    p.add_argument(
        "--mA",
        type=float,
        default=M_A_OVER_M_N,
        help="m_A'/m_N of the scan (sigma/m and the label; default 2.5)"
    )
    p.add_argument(
        "--xray-proj",
        action="store_true",
        help="draw the projected X-ray reach (XRAY_PROJECTIONS) and write <out>_xproj.pdf"
    )
    p.add_argument(
        "--bp",
        default=None,
        help="CSV (name,m_keV,sin2_2theta,...) of benchmark points to mark as stars"
    )
    return p.parse_args()


def set_style():
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
    plt.rcParams.update({
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 10,
        "font.size": 10
    })


def get_figsize(columnwidth, wf=1.0, hf=(5.**0.5 - 1.0) / 2.0):
    """[width, height] in inches for a fraction wf of the LaTeX column width
    [pt] and aspect ratio hf (golden ratio by default)."""
    fig_width = columnwidth * wf / 72.27
    return [fig_width, fig_width * hf]


def read_csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))


def key(r):
    return (round(r["m_N"] * 1e6, 10), r["y"])


def load_roots(scan_dir):
    """(m_N [keV], y) -> sin^2 2theta of every root in <scan_dir>/roots.csv."""
    return {key(r): r["sin2_2theta"] for r in read_csv(os.path.join(scan_dir, "roots.csv"))
            if r["solver"] != "failed" and np.isfinite(r["sin2_2theta"])}


def load_lyman_alpha(scan_dir):
    """(m_N [keV], y) -> lambda_fs and r_s [Mpc], and -> (f_N2, delta_m), from <scan_dir>/lyman_alpha.csv."""
    data = read_csv(os.path.join(scan_dir, "lyman_alpha.csv"))
    lam = {key(r): r["lambda_fs_kd_Mpc"] for r in data if np.isfinite(r["lambda_fs_kd_Mpc"])}
    r_s = {key(r): r["r_s_Mpc"] for r in data if np.isfinite(r["r_s_Mpc"])}
    mix = {key(r): (r["f_N2"], r["delta_m"]) for r in data if np.isfinite(r["f_N2"]) and np.isfinite(r["delta_m"])}
    return lam, r_s, mix


def bound_crossings(roots, vals, limit):
    """For every y, the point (m_N, sin^2 2theta) on its Omega h^2 = 0.12 line
    where the length crosses `limit` downwards for the last time with growing
    m_N, interpolated linearly in log along the line (the lengths fall with
    m_N, so smaller masses are excluded). Lines that are excluded at all their
    masses give their largest mass, lines that are allowed everywhere none."""
    crossings = []
    for y in sorted({k[1] for k in roots}):
        pts = sorted(
            (k[0], roots[k], vals[k]) for k in roots if k[1] == y and k in vals
        )
        if not pts or all(p[2] <= limit for p in pts):
            continue
        if pts[-1][2] > limit:
            crossings.append(pts[-1][:2])
            continue
        for (m0, s0, l0), (m1, s1, l1) in reversed(list(zip(pts[:-1], pts[1:]))):
            if l0 > limit >= l1:
                w = np.log(l0 / limit) / np.log(l0 / l1)
                crossings.append((m0 * (m1 / m0)**w, s0 * (s1 / s0)**w))
                break
    return crossings


def place_label(m, s, m_pad=1.03, s_pad=1.6):
    """(m, s, va) for a label anchored at (m, s): moved just inside the axes,
    and hung below the anchor instead of above it when it had to be pulled
    down from the top, so that it is drawn inside the frame however far off it
    the curve it belongs to runs."""
    m = min(max(m, M_LIM[0] * m_pad), M_LIM[1] / m_pad)
    if s > S_LIM[1] / s_pad:
        return m, S_LIM[1] / 1.15, "top"
    return m, max(s, S_LIM[0] * s_pad), "bottom"


def extend_boundary(m_c, s_c, base_dex=0.5):
    """The boundary continued past its end points to beyond the top and bottom
    of the frame, so the shaded region closes on the frame edge instead of on
    the outermost line of constant y that happens to have been scanned.

    The continuation is a straight line in (log m_N, log sin^2 2theta) through
    the outermost crossing and the nearest one at least `base_dex` decades
    of sin^2 2theta further in (the farthest one if none is). Not through the
    two outermost ones: at small y the lines all run into the Dodelson-Widrow
    line and their crossings pile up at nearly the same sin^2 2theta with
    jittering m_N, so that the direction through two neighbours is noise (the
    Lyman-alpha boundary was sent off the left edge at the top, leaving the
    excluded low-mass corner white). Where the bound really does run out -- the
    self-interaction boundary leaves the frame on the left, because sigma/m
    stops excluding anything above m_N = 1 keV once y is small enough -- the
    extrapolation walks off the left edge on its own and nothing extra is
    shaded, which is why it is done this way rather than by holding m_N fixed.
    """
    def step(m0, s0, m1, s1, s_target):
        # (m, s) on the line through the two points, at s = s_target
        if s1 == s0:
            return m1, s_target
        w = np.log(s_target / s1) / np.log(s1 / s0)
        m = m1 * (m1 / m0)**w
        return float(np.clip(m, M_LIM[0] * 0.01, M_LIM[1] * 100)), s_target

    def inner(order):
        # index of the crossing that sets the direction at the end order[0]
        end = order[0]
        for i in order[1:]:
            if abs(np.log10(s_c[i] / s_c[end])) >= base_dex:
                return i
        return order[-1]

    idx = list(range(len(s_c)))
    i_lo, i_hi = inner(idx), inner(idx[::-1])
    lo = step(m_c[i_lo], s_c[i_lo], m_c[0], s_c[0], S_LIM[0] * 0.5)
    hi = step(m_c[i_hi], s_c[i_hi], m_c[-1], s_c[-1], S_LIM[1] * 2.0)
    return (np.concatenate(([lo[0]], m_c, [hi[0]])),
            np.concatenate(([lo[1]], s_c, [hi[1]])))


def plot_bound(
        ax,
        roots,
        vals,
        limit,
        color,
        label,
        zorder,
        frac=0.5
    ):
    """Shade the region of smaller m_N than the crossings of `limit` and label
    the boundary inside the shaded side, `frac` of the way up its visible part
    (0.5 is halfway; lower it to dodge a busy part of the figure)."""
    crossings = sorted(bound_crossings(roots, vals, limit), key=lambda c: c[1])
    if len(crossings) < 2:
        return False
    m_c = np.array([c[0] for c in crossings])
    s_c = np.array([c[1] for c in crossings])
    m_e, s_e = extend_boundary(m_c, s_c)
    m_lo = M_LIM[0] * 0.5   # off the left edge, so the shading is flush with it
    ax.fill(
        np.concatenate(([m_lo], m_e, [m_lo])),
        np.concatenate(([s_e[0]], s_e, [s_e[-1]])),
        color=color, alpha=0.25, lw=0,
        zorder=zorder
    )
    ax.plot(m_e, s_e, color=color, lw=1.3, zorder=zorder + 0.1)

    # Label offset into the excluded (left) side and below the anchor, so it
    # clears the boundary line instead of sitting on it. The text is above
    # everything, since the shading is drawn under the X-ray region.
    vis = np.nonzero((s_c > S_LIM[0]) & (s_c < S_LIM[1]))[0]
    if len(vis):
        mid = vis[min(int(frac * len(vis)), len(vis) - 1)]
    else:
        mid = int(np.argmax(m_c))
    m_t = float(np.clip(m_c[mid] * 0.88, M_LIM[0] * 1.06, M_LIM[1] / 1.06))
    s_t = float(np.clip(s_c[mid] * 0.70, S_LIM[0] * 4.0, S_LIM[1] / 2.0))
    ax.text(m_t, s_t, label, color=color, fontsize=BOUND_LABEL_SIZE,
            ha="right", va="top", zorder=Z_LABEL)
    return True


def plot_dodelson_widrow(ax):
    """Band of Omega h^2 = 0.12 from DW production (fig. 5 of hep-ph/0612182),
    with the overproduction region above it."""
    def load(name):
        d = np.loadtxt(os.path.join(DW_DIR, name), skiprows=2)
        return (
            1e6 * d[:, 0],
            (OMEGA_TARGET / OMEGA_DW_REF) * d[:, 1] * (1e-6 / d[:, 0])**2
        )

    m_mid, s_mid = load("0612182_dw_fig_5.dat")
    m_up, s_up = load("0612182_dw_fig_5_up.dat")
    m_low, s_low = load("0612182_dw_fig_5_low.dat")
    ax.plot(m_mid, s_mid, color=C_DW, ls="--", zorder=1)
    ax.plot(m_low, s_low, color=C_DW, ls=":", zorder=1)
    ax.plot(m_up, s_up, color=C_DW, ls=":", zorder=1)
    ax.fill(
        np.concatenate((m_low, m_up[::-1])),
        np.concatenate((s_low, s_up[::-1])),
        color=C_DW_BAND,
        lw=0,
        zorder=0
    )
    for m, s in ((m_low, s_low), (m_up, s_up)):
        ax.fill_between(m, s, 1.0, color=C_OVERPROD, lw=0, zorder=-1)
    ax.text(
        10**1.5, 10**-10.25,
        "Dodelson-Widrow",
        color=C_DW, rotation=-22, ha="center"
    )
    ax.text(
        10**1.6, 10**-10.25,
        "overproduction",
        color=C_OVERPROD_TEXT,
        rotation=-24
    )


def load_xray_limit():
    """(m_N [keV], sin^2 2theta) of the combined X-ray limit, which assumes that all the DM decays."""
    xray = np.loadtxt(os.path.join(XRAY_DIR, "overall_constraint.dat"))
    return 1e6 * xray[:, 0], xray[:, 1]


def f_N1_field(roots, mix):
    """
    f_N1(m [keV], sin^2 2theta) interpolated from the roots: at each scanned mass linear in
    log sin^2 2theta between the roots (constant beyond the first and last), and linear in
    log m between scanned masses (constant beyond them).
    """
    cols = {}
    for k in roots:
        if k in mix:
            cols.setdefault(k[0], []).append((np.log(roots[k]), 1.0 - mix[k][0]))
    ms = np.array(sorted(cols))
    tabs = [np.array(sorted(cols[m])).T for m in ms]

    def f(m, s2):
        lm, ls = np.log(m), np.log(s2)
        col = lambda i: np.interp(ls, tabs[i][0], tabs[i][1])
        if m <= ms[0]:
            return col(0)
        if m >= ms[-1]:
            return col(len(ms) - 1)
        i = np.searchsorted(ms, m) - 1
        t = (lm - np.log(ms[i])) / (np.log(ms[i + 1]) - np.log(ms[i]))
        return (1 - t) * col(i) + t * col(i + 1)
    return f


def scaled_xray_limit(roots, mix, limit=None):
    """
    The X-ray boundary in the relic plane. Only N1 mixes with nu and decays to nu gamma, so a root
    is excluded if f_N1 sin^2 2theta exceeds the limit L(m), which assumes that all the DM decays.
    f_N1 depends on the root (from 1 for DW-produced N1 at small g to 1/2 where N1 and N2 are in
    chemical equilibrium), so the boundary at each mass is the s with s f_N1(m, s) = L(m): s lies in
    [L, 2L] and s f_N1(m, s) grows with s, found by bisection in log s. `limit` = (m [keV], L)
    replaces the combined current limit (e.g. a projection).
    """
    m, L = load_xray_limit() if limit is None else limit
    f = f_N1_field(roots, mix)
    s_b = np.empty_like(L)
    for i in range(len(m)):
        lo, hi = np.log(L[i]), np.log(2 * L[i])
        for _ in range(40):
            mid = (lo + hi) / 2
            if mid + np.log(f(m[i], np.exp(mid))) > np.log(L[i]):
                hi = mid
            else:
                lo = mid
        s_b[i] = np.exp((lo + hi) / 2)
    return m, s_b


def load_xray_projections():
    """{label: (m_N [keV], sin^2 2theta)} of the projected sensitivities (all DM decaying)."""
    out = {}
    for f, label in XRAY_PROJECTIONS.items():
        d = np.loadtxt(os.path.join(XRAY_DIR, f), comments="#")
        out[label] = (1e6 * d[:, 0], d[:, 1])
    return out


def plot_xray_projections(ax, roots, mix):
    """The projected reach, rescaled by f_N1 as the current limit (`scaled_xray_limit`)."""
    for (label, lim), ls in zip(load_xray_projections().items(), ("--", "-.", ":")):
        m, s = scaled_xray_limit(roots, mix, lim)
        ax.plot(m, s, color="black", lw=1.0, ls=ls, zorder=-2, label=label)
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.8, edgecolor="0.8",
              title="projected X-ray reach", title_fontsize=7)


def plot_benchmarks(ax, path):
    """Red stars at the benchmark points of `path` (CSV: name, m_keV, sin2_2theta, ...)."""
    import csv
    for r in csv.DictReader(open(path)):
        m, s = float(r["m_keV"]), float(r["sin2_2theta"])
        ax.plot(m, s, marker="*", ms=9, color="red", mec="darkred", mew=0.5, zorder=4)
        ax.annotate(r["name"], (m, s), xytext=(4, 3), textcoords="offset points", fontsize=7,
                    color="darkred", zorder=4)


def plot_xrays(ax, m, s):
    ax.fill_between(m, s, 1.0, color="white", lw=0, zorder=-3)
    ax.fill_between(m, s, 1.0, color="black", alpha=0.25, lw=0, zorder=-3)
    ax.plot(m, s, color="black", lw=1.3, zorder=-2)
    ax.text(10**1.45, 1e-13, "X-rays", color="black")


def plot_contours(ax, roots, excluded, debug):
    """The Omega h^2 = 0.12 lines, drawn for whole decades of y only."""
    masses = np.array(sorted({k[0] for k in roots}))
    ys = np.array(sorted({k[1] for k in roots}))
    decade_ys = np.array(
        [y for y in ys if abs(np.log10(y) - round(np.log10(y))) < 1e-6]
    )
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(decade_ys)))
    first = [True]   # the first line that gets a label carries the "g =" prefix
    for y, c in zip(decade_ys, colors):
        m_line = np.array([m for m in masses if (m, y) in roots])
        s_line = np.array([roots[(m, y)] for m in m_line])
        if len(m_line) == 0:
            continue
        # Break the line where a mass in the grid has no root for this y.
        idx = np.searchsorted(masses, m_line)
        segments = np.split(
            np.arange(len(m_line)),
            np.where(np.diff(idx) > 1)[0] + 1
        )
        for seg in segments:
            lm, ls = np.log(m_line[seg]), np.log(s_line[seg])
            if len(seg) > 1:
                lm_fine = np.linspace(lm[0], lm[-1], 200)
                ax.plot(
                    np.exp(lm_fine),
                    np.exp(PchipInterpolator(lm, ls)(lm_fine)),
                    color=c,
                    lw=1.0,
                    zorder=-1
                )
            if debug:
                # Open markers where a root is excluded by one of the
                # constraints. Only in --debug: a segment of a single root
                # used to be drawn as a marker here, since it cannot be drawn
                # as a line, but in the production figure that puts a lone dot
                # in the middle of nowhere. The one case in the current scan
                # is m_N = 3.72 keV at g = 1e-8, whose neighbours in the mass
                # grid both failed to converge, and which is excluded anyway
                # (lambda_fs = 0.47 Mpc). Isolated roots are now visible only
                # in --debug; the fix is to make those masses converge.
                ex = np.array(
                    [excluded.get((m, y), False) for m in m_line[seg]]
                )
                ax.plot(
                    m_line[seg][~ex],
                    s_line[seg][~ex],
                    ls="none",
                    marker="o", ms=2.5,
                    color=c,
                    zorder=2
                )
                ax.plot(
                    m_line[seg][ex],
                    s_line[seg][ex],
                    ls="none",
                    marker="o", ms=2.5,
                    color=c,
                    mfc="white",
                    zorder=2
                )
        # Label at the left edge, just above the start of the line, as in the
        # paper figure -- but anchored to the first point that is on the plot,
        # since the smallest couplings leave the top of the frame well to the
        # right of m_N = 1 keV. The couplings below LABEL_Y_MIN only appear in
        # the top-right corner, crowded against the Dodelson-Widrow band, and
        # are left unlabelled; their lines are still drawn.
        inside = np.nonzero(s_line < S_LIM[1] / 10)[0]
        if len(inside) and y >= LABEL_Y_MIN * (1 - 1e-6):
            j = inside[0]
            # The 2.6 clears the line itself: the anchor is offset to the right
            # as well, and the contours rise with m_N, so a smaller lift left
            # the text sitting on its own curve.
            m_t, s_t, va = place_label(
                m_line[j] * 1.14,
                s_line[j] * 2.6,
                m_pad=1.12
            )
            # Above the bands: the smallest couplings only enter the frame
            # inside the Dodelson-Widrow band, which the lines run under.
            ax.text(
                m_t, s_t,
                y_label(y, first=first[0]),
                color=c,
                ha="left", va=va,
                zorder=Z_LABEL
            )
            first[0] = False


def plot_all_points(fig, ax, roots, excluded):
    """Every root of the scan, all y, coloured by log10 g (--all-points)."""
    from matplotlib.colors import Normalize
    keys = sorted(roots)
    m = np.array([k[0] for k in keys])
    s = np.array([roots[k] for k in keys])
    lg = np.log10([k[1] for k in keys])
    ex = np.array([excluded.get(k, False) for k in keys])
    norm = Normalize(lg.min(), lg.max())
    cmap = plt.cm.viridis
    ax.scatter(m[~ex], s[~ex], s=3, c=cmap(norm(lg[~ex])), lw=0, zorder=3)
    ax.scatter(m[ex], s[ex], s=4, facecolors="white", edgecolors=cmap(norm(lg[ex])), lw=0.4, zorder=3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, pad=0.02, label=r"$\log_{10} g$")


def y_label(y, first):
    exp10 = np.log10(y)
    if abs(exp10 - round(exp10)) < 1e-6:
        exp_str = f"{exp10:.0f}"
    else:
        exp_str = f"{exp10:.1f}"
    return (r"$g = " if first else "$") + rf"10^{{{exp_str}}}$"


def main():
    args = parse_args()
    roots = load_roots(args.scan_dir)
    lam, r_s, mix = load_lyman_alpha(args.scan_dir)
    sig_m = {k: sigma_over_m(k[0] * 1e-6, k[1], args.mA, *mix[k]) for k in roots if k in mix}
    out = args.out + ("_debug_all" if args.all_points else "_debug" if args.debug else
                      "_xproj" if args.xray_proj else "")

    set_style()
    fig = plt.figure(figsize=get_figsize(COLUMNWIDTH, wf=1.0, hf=0.9), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(axis="both", which="both", direction="in", width=0.5)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    plot_dodelson_widrow(ax)
    m_x, s_x = scaled_xray_limit(roots, mix)
    plot_xrays(ax, m_x, s_x)
    # Both Lyman-alpha bounds are labelled by the length that sets them.
    # Both bounds are Lyman-alpha; the length in brackets says which one, as in
    # Bringmann et al. (2206.10630). Ly-alpha (lambda_fs) is labelled low on
    # its boundary, clear of the ragged X-ray curve.
    n_lya = plot_bound(
        ax, roots, lam, LAMBDA_FS_MAX_MPC, C_LYA,
        r"Ly-$\alpha$ ($\lambda_\mathrm{fs}$)", Z_BOUND["lya"], frac=0.28
    )
    # The r_s label sits on the upper part of its boundary, above the
    # lambda_fs line. Both label positions depend on the data: recheck them
    # (and those of Ly-alpha (lambda_fs) and self-int.) when the bounds move.
    plot_bound(
        ax, roots, r_s, R_S_MAX_MPC, C_RS,
        r"Ly-$\alpha$ ($r_s$)", Z_BOUND["r_s"], frac=0.45
    )
    plot_bound(ax, roots, sig_m, SIGMA_M_MAX, C_SI, r"self-int.", Z_BOUND["si"], frac=0.55)

    excluded = {
        k: (
            roots[k] > np.exp(np.interp(np.log(k[0]), np.log(m_x), np.log(s_x), left=np.inf, right=np.inf))
            or lam.get(k, 0.0) > LAMBDA_FS_MAX_MPC
            or r_s.get(k, 0.0) > R_S_MAX_MPC
            or sig_m.get(k, 0.0) > SIGMA_M_MAX
        ) for k in roots
    }
    plot_contours(ax, roots, excluded, args.debug and not args.all_points)
    if args.all_points:
        plot_all_points(fig, ax, roots, excluded)
    if args.xray_proj:
        plot_xray_projections(ax, roots, mix)
    if args.bp:
        plot_benchmarks(ax, args.bp)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*M_LIM)
    ax.set_ylim(*S_LIM)
    # Label every decade, as in the paper figure.
    for axis, (lo, hi) in ((ax.xaxis, M_LIM), (ax.yaxis, S_LIM)):
        decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
        axis.set_major_locator(FixedLocator(10**decades))
        axis.set_minor_locator(
            FixedLocator([k * 10**d for d in decades for k in range(2, 10)])
        )
        axis.set_major_formatter(LogFormatterMathtext())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$m_{N_1}\;\;[\mathrm{keV}]$")
    ax.set_ylabel(r"$\sin^2 (2 \theta_1)$")
    props = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8,
        linewidth=1,
        edgecolor="0.8"
    )
    ax.text(
        0.97, 0.96, 
        rf"$m_{{A'}} = {args.mA:g}\,m_{{N_1}}$",
        transform=ax.transAxes,
        ha="right", va="top", bbox=props
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out + ".pdf")
    print(f"wrote {out}.pdf with {len(roots)} points, "
          f"{len({k[0] for k in roots})} masses, {len({k[1] for k in roots})} values of y, "
          f"Lyman-alpha lengths for {sum(k in lam for k in roots)} of them"
          + ("" if n_lya else " (no lambda_fs bound drawn)"))


if __name__ == "__main__":
    main()

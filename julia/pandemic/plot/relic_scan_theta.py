#! /usr/bin/env python3
"""
Contours of the coupling y (= g in the paper) in the (m_N, sin^2 2theta) plane
for which Omega h^2 = 0.12, from the scan in src/run_relic_scan_theta_z.jl
(tmp/relic_scan_theta/m_N_*keV.csv). Styled like the money plot of the paper
(code/sterile_res/plotter_3.py): Dodelson-Widrow band, current X-ray limits,
and the contours labelled on the plot. The X-ray region is drawn over the
constraint bounds below, and the projected sensitivities (Athena, eROSITA,
eXTP) are deliberately left out to keep the figure readable.

Only roots on the freeze-in branch are drawn: for each (m_N, y) the smallest
converged sin^2 2theta at which Omega h^2 rises through the target. The lines
are monotone (PCHIP) interpolations in log-log between the scanned masses and
are broken where no root was found for a mass in the grid. Only whole decades
of y are drawn, and only those at y >= LABEL_Y_MIN are labelled; a finer
sub-decade grid, if scanned, still feeds the constraint boundaries below,
which are what it is needed for.

Three constraints are shaded, each bounding the region of small m_N:

  Ly-alpha    free-streaming length after kinetic decoupling, lambda_fs >
              0.24 Mpc, and sound horizon r_s > 0.34 Mpc, remapped from the
              WDM limits as in Bringmann et al. (2206.10630). Needs
              <scan_dir>/lyman_alpha.csv from src/run_lyman_alpha.jl; without
              it the two are silently skipped.
  self-int.   sigma/m > 1 cm^2/g from the draft's (A6), see
              plot/self_interaction.py.

On every line of constant y the crossing of each limit is interpolated (in
log) between neighbouring roots and the crossings are joined. For sigma/m the
interpolation is exact, since sigma/m ~ m^-3 at fixed y and m_A'/m_N.

Environment: LYA_VARIANT=old plots the lengths as the old Python code computed
them (its amplitude and its stray factor 1e3), LYA_VARIANT=nokd the
free-streaming length without kinetic decoupling; SI_VARIANT=old the
self-interaction cross section of plotter_3.py.

Run from julia/pandemic:
    conda run -n pandemic python plot/relic_scan_theta.py [scan_dir] [output_path_without_extension] [--debug]

--debug marks the scanned roots on the lines (open markers where a root is
excluded by one of the constraints) and writes <output>_debug.pdf, so the
production figure is not overwritten.
"""

import argparse
import glob
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
M_A_OVER_M_N = 2.5

COLUMNWIDTH = 418.25368  # pt, \showthe\textwidth in LaTeX, as for the paper figure
M_LIM = (1.0, 300.0)     # keV
S_LIM = (1e-18, 1e-8)

C_DW = "#83781B"
C_DW_BAND = "#EAE299"
C_OVERPROD = "#92CBE2"   # skyblue at alpha 0.8 on white
C_OVERPROD_TEXT = "#155D7A"

# Limits of the three constraints, and the colours they are drawn in.
# LAMBDA_FS_MAX_MPC and R_S_MAX_MPC are LYA_LAMBDA_FS_MAX_MPC and
# LYA_R_S_MAX_MPC of src/lyman_alpha.jl and src/kinetic_decoupling.jl.
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

SI_VARIANT = os.environ.get("SI_VARIANT", "A6")
LYA_VARIANT = os.environ.get("LYA_VARIANT", "A6")
LYA_COLUMNS = {"A6": ("lambda_fs_kd_Mpc", "r_s_Mpc"),
               "old": ("lambda_fs_kd_old_Mpc", "r_s_old_Mpc"),
               "nokd": ("lambda_fs_Mpc", None)}[LYA_VARIANT]


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


def load_lyman_alpha(scan_dir):
    """(m_N [keV], y) -> lambda_fs and r_s [Mpc] of LYA_VARIANT from
    src/run_lyman_alpha.jl, or ({}, {}) if it has not been run."""
    path = os.path.join(scan_dir, "lyman_alpha.csv")
    if not os.path.exists(path):
        return {}, {}
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    col_l, col_r = LYA_COLUMNS
    key = lambda r: (round(r["m_N"] * 1e6, 10), r["y"])
    lam = {key(r): r[col_l] for r in data if np.isfinite(r[col_l])}
    r_s = {key(r): r[col_r] for r in data if np.isfinite(r[col_r])} if col_r else {}
    return lam, r_s


def bound_crossings(roots, vals, limit):
    """For every y, the point (m_N, sin^2 2theta) on its Omega h^2 = 0.12 line
    where the length crosses `limit` downwards for the last time with growing
    m_N, interpolated linearly in log along the line (the lengths fall with
    m_N, so smaller masses are excluded). Lines that are excluded at all their
    masses give their largest mass, lines that are allowed everywhere none."""
    crossings = []
    for y in sorted({k[1] for k in roots}):
        pts = sorted((k[0], roots[k], vals[k]) for k in roots if k[1] == y and k in vals)
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


def extend_boundary(m_c, s_c):
    """The boundary continued past its end points to beyond the top and bottom
    of the frame, so the shaded region closes on the frame edge instead of on
    the outermost line of constant y that happens to have been scanned.

    The continuation is a straight line in (log m_N, log sin^2 2theta) through
    the two outermost crossings. Where the bound really does run out -- the
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

    lo = step(m_c[1], s_c[1], m_c[0], s_c[0], S_LIM[0] * 0.5)
    hi = step(m_c[-2], s_c[-2], m_c[-1], s_c[-1], S_LIM[1] * 2.0)
    return (np.concatenate(([lo[0]], m_c, [hi[0]])),
            np.concatenate(([lo[1]], s_c, [hi[1]])))


def plot_bound(ax, roots, vals, limit, color, label, zorder, frac=0.5):
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
    ax.fill(np.concatenate(([m_lo], m_e, [m_lo])), np.concatenate(([s_e[0]], s_e, [s_e[-1]])),
            color=color, alpha=0.25, lw=0, zorder=zorder)
    ax.plot(m_e, s_e, color=color, lw=1.3, zorder=zorder + 0.1)

    # Label offset into the excluded (left) side and below the anchor, so it
    # clears the boundary line instead of sitting on it. The text is above
    # everything, since the shading is drawn under the X-ray region.
    vis = np.nonzero((s_c > S_LIM[0]) & (s_c < S_LIM[1]))[0]
    mid = vis[min(int(frac * len(vis)), len(vis) - 1)] if len(vis) else int(np.argmax(m_c))
    m_t = float(np.clip(m_c[mid] * 0.88, M_LIM[0] * 1.06, M_LIM[1] / 1.06))
    s_t = float(np.clip(s_c[mid] * 0.70, S_LIM[0] * 4.0, S_LIM[1] / 2.0))
    ax.text(m_t, s_t, label, color=color, fontsize=BOUND_LABEL_SIZE,
            ha="right", va="top", zorder=Z_LABEL)
    return True


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


def plot_contours(ax, roots, excluded, debug):
    """The Omega h^2 = 0.12 lines, drawn for whole decades of y only."""
    masses = np.array(sorted({k[0] for k in roots}))
    ys = np.array(sorted({k[1] for k in roots}))
    decade_ys = np.array([y for y in ys if abs(np.log10(y) - round(np.log10(y))) < 1e-6])
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(decade_ys)))
    first = [True]   # the first line that gets a label carries the "g =" prefix
    for y, c in zip(decade_ys, colors):
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
                # Open markers where a root is excluded by one of the constraints.
                ex = np.array([excluded.get((m, y), False) for m in m_line[seg]])
                ax.plot(m_line[seg][~ex], s_line[seg][~ex], ls="none", marker="o", ms=2.5, color=c, zorder=2)
                ax.plot(m_line[seg][ex], s_line[seg][ex], ls="none", marker="o", ms=2.5, color=c,
                        mfc="white", zorder=2)
        # Label at the left edge, just above the start of the line, as in the
        # paper figure -- but anchored to the first point that is on the plot,
        # since the smallest couplings leave the top of the frame well to the
        # right of m_N = 1 keV. The couplings below LABEL_Y_MIN only appear in
        # the top-right corner, crowded against the Dodelson-Widrow band, and
        # are left unlabelled; their lines are still drawn.
        inside = np.nonzero(s_line < S_LIM[1] / 10)[0]
        if len(inside) and y >= LABEL_Y_MIN * (1 - 1e-6):
            j = inside[0]
            m_t, s_t, va = place_label(m_line[j] * 1.14, s_line[j] * 1.6, m_pad=1.12)
            # Above the bands: the smallest couplings only enter the frame
            # inside the Dodelson-Widrow band, which the lines run under.
            ax.text(m_t, s_t, y_label(y, first=first[0]), color=c,
                    ha="left", va=va, zorder=Z_LABEL)
            first[0] = False


def y_label(y, first):
    exp10 = np.log10(y)
    exp_str = f"{exp10:.0f}" if abs(exp10 - round(exp10)) < 1e-6 else f"{exp10:.1f}"
    return (r"$g = " if first else "$") + rf"10^{{{exp_str}}}$"


def main():
    args = parse_args()
    roots = freeze_in_roots(load_scan(args.scan_dir))
    lam, r_s = load_lyman_alpha(args.scan_dir)
    sig_m = {k: sigma_over_m(k[0] * 1e-6, k[1], M_A_OVER_M_N, SI_VARIANT) for k in roots}
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
    # Both Lyman-alpha bounds are labelled by the length that sets them.
    # Ly-alpha is labelled low on its boundary, clear of the ragged X-ray curve.
    n_lya = plot_bound(ax, roots, lam, LAMBDA_FS_MAX_MPC, C_LYA, r"Ly-$\alpha$",
                       Z_BOUND["lya"], frac=0.28)
    plot_bound(ax, roots, r_s, R_S_MAX_MPC, C_RS, r"$r_s$", Z_BOUND["r_s"])
    plot_bound(ax, roots, sig_m, SIGMA_M_MAX, C_SI, r"self-int.", Z_BOUND["si"])

    excluded = {k: (lam.get(k, 0.0) > LAMBDA_FS_MAX_MPC or r_s.get(k, 0.0) > R_S_MAX_MPC
                    or sig_m[k] > SIGMA_M_MAX) for k in roots}
    plot_contours(ax, roots, excluded, args.debug)

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
    print(f"wrote {out}.pdf with {len(roots)} points, "
          f"{len({k[0] for k in roots})} masses, {len({k[1] for k in roots})} values of y, "
          f"Lyman-alpha lengths for {sum(k in lam for k in roots)} of them"
          + ("" if n_lya else " (no lambda_fs bound drawn)"))


if __name__ == "__main__":
    main()

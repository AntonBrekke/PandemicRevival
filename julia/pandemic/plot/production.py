"""
Production figures in the style of the draft's Fig. (evolution), from the benchmark CSVs of
src/run_benchmarks.jl: one PDF per point, for either solver.

    python plot/production.py [csv | dir] [fig_dir] [--min 1e-2]

Defaults: tmp/benchmarks -> figures/production. Against x = m_N/T_nu:
- top left: m_N n/s [keV] of each species, counting the N it decays into (A' -> 2 N,
  phi -> w_phi N; their sum is Omega h^2 in other units), with Omega h^2 = 0.12; N2 dashed;
- bottom left: T_N/T_nu;
- right: the reaction rates per DM particle [keV], as in the draft (C/n_d), and H: the
  forward rate of each reaction per n~ (its rate per first initial particle times
  n_first/n~), the same quantity for both solvers (the fluid's from the MB tables at its
  state). The fluid's per-process net terms are not used: in equilibrium they are
  residuals of near-cancellations. The net change of the DM is the abundance history.
  n~ = n_N1 + n_N2 + 2 n_A' + w_phi n_phi is the number of N after all decays. A reaction is
  drawn only if it ever runs: if int (rate/H) d ln x over the history reaches --min.
The first network point is the seed state (network_initial_state) and is left out: its
seeds of N2, A' and phi relax within the first step.
"""
import sys
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from benchmark import LABELS, RATE_COLOURS, SPECIES_COLOURS, SPECIES_LS, omega_parts

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 10, "font.size": 10})

COLUMNWIDTH = 426.39256   # pt, \textwidth of the draft
GEV_KEV = 1e6
C_H = "crimson"
FIRST = (("N1", "Y_N1"), ("N2", "Y_N2"), ("phi", "Y_phi"), ("A", "Y_A"))   # order matters: A before AA


def log_ticks(ax, axis, lo, hi, every=1, blank_at=None):
    """Major ticks at whole decades, labelled every `every`-th (not at `blank_at`); no minor ticks."""
    dec = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
    lab = [rf"$10^{{{int(e)}}}$" if int(e) % every == 0 and not (blank_at and np.isclose(10.0**e, blank_at)) else ""
           for e in dec]
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(10.0**dec))
    a.set_major_formatter(FixedFormatter(lab))
    a.set_minor_locator(FixedLocator([]))
    a.set_minor_formatter(NullFormatter())


def place_labels(ax, curves, fontsize=9, gap=0.012):
    """
    Writes each curve's label next to it (above, below, beside or diagonal to a point of the
    curve), where the label box crosses the fewest curves and labels and is farthest from them
    (log axes, in axes units). Call after the layout is final.
    curves: (x, y, text, colour); text None marks a curve that is only avoided.
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    renderer = ax.figure.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    grid = np.linspace(0, 1, 500)
    on_grid = []
    for x, y, _, _ in curves:
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (np.log10(x) - np.log10(x0)) / (np.log10(x1) - np.log10(x0))
            v = (np.log10(y) - np.log10(y0)) / (np.log10(y1) - np.log10(y0))
        ok = np.isfinite(v)
        if not ok.any():
            on_grid.append(np.full_like(grid, np.nan))
            continue
        g = np.interp(grid, u[ok], v[ok], left=np.nan, right=np.nan)
        g[np.interp(grid, u, ok.astype(float)) < 1] = np.nan   # keep the curve's gaps
        on_grid.append(g)
    placed = []   # label boxes (u_lo, u_hi, v_lo, v_hi)
    for k, (_, _, text, colour) in enumerate(curves):
        if text is None:
            continue
        t = ax.text(0, 0, text, color=colour, transform=ax.transAxes, ha="center", va="bottom", fontsize=fontsize)
        bb = t.get_window_extent(renderer)
        hw, h = 0.5 * bb.width / ax_box.width, bb.height / ax_box.height
        best, best_score = None, (-np.inf, -np.inf)
        for i in range(0, len(grid), 3):
            if not np.isfinite(on_grid[k][i]):
                continue
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    if du == dv == 0:
                        continue
                    u = grid[i] + du * (hw + gap)
                    lo = on_grid[k][i] + (gap if dv > 0 else -gap - h if dv < 0 else -h / 2)
                    hi = lo + h
                    if not (hw + 0.01 < u < 0.99 - hw and 0.01 < lo and hi < 0.99):
                        continue
                    span = np.abs(grid - u) <= hw
                    crossings, dist = 0, 0.15
                    for g in on_grid:   # the own curve included: it must not cross its label
                        seg = g[span][np.isfinite(g[span])]
                        if seg.size == 0:
                            continue
                        if seg.min() <= hi and seg.max() >= lo:
                            crossings += 1
                        else:
                            dist = min(dist, np.min(np.maximum(lo - seg, seg - hi)))
                    for (pl, ph, ql, qh) in placed:
                        if pl < u + hw and u - hw < ph:
                            sep = max(ql - hi, lo - qh)
                            crossings += sep < 0
                            dist = min(dist, max(sep, 0))
                    score = (-crossings, dist)
                    if score > best_score:
                        best, best_score = (u, lo), score
        if best is None:
            t.remove()
            continue
        u, lo = best
        t.set_position((u, lo))
        placed.append((u - hw, u + hw, lo, lo + h))


def sci_tex(v):
    m, e = f"{v:.2e}".split("e")
    return rf"{m}\times 10^{{{int(e)}}}" if m != "1.00" else rf"10^{{{int(e)}}}"


def plot(path, fig_dir, min_moved):
    title = open(path).readline().lstrip("# ").strip()
    solver = title.split()[0]
    m_keV = float(re.search(r"m_N = ([\d.eE+-]+) keV", title).group(1))
    g = float(re.search(r"g = ([\d.eE+-]+)", title).group(1))
    s2 = float(re.search(r"2theta = ([\d.eE+-]+)", title).group(1))
    mphi = re.search(r"_mphi([\d.]+)\.csv$", path)
    mphi = float(mphi.group(1)) if mphi else 100.0
    d = np.genfromtxt(path, delimiter=",", names=True, skip_header=1, deletechars="")
    om1, om2, omA, omphi, w_phi = omega_parts(d)
    # Omega h^2 = m_N Y s0/rho_crit0_h2: m_N Y [keV] = Omega h^2 * m_N / k, k from omega_parts
    keV_per_omega = m_keV * (om1[-1] / d["Y_N1"][-1]) ** -1
    if solver == "network":
        d, om1, om2, omA, omphi = (a[1:] for a in (d, om1, om2, omA, omphi))
    x, H = d["x"], d["H_GeV"] * GEV_KEV
    Yt = d["Y_N1"] + d["Y_N2"] + 2 * d["Y_A"] + w_phi * d["Y_phi"]

    w, h = COLUMNWIDTH / 72.27 * 1.5, COLUMNWIDTH / 72.27 * 0.75
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0.0, wspace=0.0)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    a3 = fig.add_subplot(gs[:, 1])
    for a in (a1, a2, a3):
        a.tick_params(axis="both", which="both", direction="in", width=0.5, top=True, right=True)
        for s in a.spines.values():
            s.set_linewidth(0.5)
    xlim = (x[0], x[-1])

    # abundances
    total = d["omega_h2"] * keV_per_omega
    top = max(3e2 * total.max(), 0.12 * keV_per_omega * 30)
    bottom = 1e-16 * total[-1]
    curves = []
    for o, key, lab in ((om1, "N1", r"$N_1$"), (om2, "N2", r"$N_2$"), (omA, "A", r"$A'$"), (omphi, "phi", r"$\phi$")):
        y = o * keV_per_omega
        if np.nanmax(y) <= 1e2 * bottom:   # never visible
            continue
        a1.loglog(x, y, color=SPECIES_COLOURS[key], ls=SPECIES_LS.get(key, "-"))
        curves.append((x, y, lab, SPECIES_COLOURS[key]))
    a1.loglog(xlim, [0.12 * keV_per_omega] * 2, color="0.55", ls="-.", zorder=-2)
    curves.append((np.array(xlim), np.array([0.12 * keV_per_omega] * 2), r"$\Omega h^2 = 0.12$", "0.55"))
    a1_curves = curves
    a1.set_xlim(*xlim)
    a1.set_ylabel(r"$m_N\, n / s\;\;\mathrm{[keV]}$")
    plt.setp(a1.get_xticklabels(), visible=False)
    log_ticks(a1, "y", bottom, top, every=2)
    a1.set_ylim(bottom, top)

    a2.loglog(x, d["TN_Tnu"], color="0.4")
    a2.set_ylabel(r"$T_N/T_\nu$")
    a2.set_xlabel(r"$x = m_N / T_\nu$")
    t = d["TN_Tnu"]
    a2.set_ylim(t.min() / 2, t.max() * 2)

    # rates per DM particle
    curves = []
    rate_cols = [n for n in d.dtype.names if n.startswith("fwd:")]
    lx = np.log(x)
    rmax = H.max()
    for n in rate_cols:
        name = n.split(":", 1)[1]
        first = next(col for sp, col in FIRST if name.startswith(sp))
        r = d[n] * d[first] / Yt   # over H
        moved = np.sum(0.5 * (np.abs(r[1:]) + np.abs(r[:-1])) * np.diff(lx))
        if moved < min_moved:
            continue
        c = RATE_COLOURS.get(name, "0.3")
        a3.loglog(x, np.where(r > 0, r * H, np.nan), color=c, zorder=-4)
        curves.append((x, np.abs(r) * H, LABELS.get(name, name), c))
        rmax = max(rmax, np.nanmax(np.abs(r) * H))
    a3.loglog(x, H, color=C_H, lw=2.0, zorder=0)
    curves.insert(0, (x, H, r"$H$", C_H))
    rlo, rhi = 1e-2 * H.min(), 30 * rmax
    a3.set_xlim(*xlim)
    a3.set_ylim(rlo, rhi)
    a3.set_xlabel(r"$x = m_N / T_\nu$")
    a3.set_ylabel(r"$\text{Rate per DM particle}\;\;[\mathrm{keV}]$")
    a3.yaxis.set_label_position("right")
    a3.yaxis.set_ticks_position("right")
    log_ticks(a3, "y", rlo, rhi, every=4)
    a3.set_ylim(rlo, rhi)
    # a decade at the shared edge is labelled on the right panel only
    log_ticks(a2, "x", *xlim, blank_at=xlim[1])
    log_ticks(a3, "x", *xlim)
    for a in (a2, a3):
        a.set_xlim(*xlim)

    fig.suptitle(rf"$m_N = {m_keV:g}\,\mathrm{{keV}},\ m_{{A'}} = 2.5\, m_N,\ m_\phi = {mphi:g}\, m_N,\ "
                 rf"g = {sci_tex(g)},\ \sin^2 2\theta = {sci_tex(s2)}$", y=0.99)
    fig.tight_layout()
    place_labels(a1, a1_curves)
    place_labels(a3, curves)
    out = os.path.join(fig_dir, os.path.basename(path)[:-4] + ".pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    min_moved = 1e-2
    if "--min" in args:
        i = args.index("--min")
        min_moved = float(args[i + 1])
        del args[i:i + 2]
    src = args[0] if len(args) > 0 else "tmp/benchmarks"
    fig_dir = args[1] if len(args) > 1 else "figures/production"
    os.makedirs(fig_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, "*.csv"))) if os.path.isdir(src) else [src]:
        plot(p, fig_dir, min_moved)

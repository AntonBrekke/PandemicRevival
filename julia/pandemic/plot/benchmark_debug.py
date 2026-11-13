"""
Debugging figures of the benchmark histories of src/run_benchmarks.jl (per-process detail; the
benchmark figures themselves are plot/benchmark.py): one PDF per point, for either solver.

    python plot/benchmark_debug.py [csv | dir] [fig_dir] [--hide-invalid]

Defaults: tmp/benchmarks -> figures/benchmarks_debug. Panels, against x = m_N/T_nu, one y axis each:
1. the contribution of each species to Omega h^2, counting the N it decays into (A' -> 2 N,
   phi -> w_phi N; N2 dashed, it often lies on N1), and their sum Omega h^2 (black dashed;
   dotted: 0.12);
2. T_N/T_nu;
3. (m_N - mu_N1)/T_N;
4. the rates the solver computes, in GeV, with H as the thick black line (one colour per
   reaction, RATE_COLOURS, the same for both solvers and in benchmark.py):
   network: forward rate per particle of every reaction (solid: dark sector, dashed:
            with a neutrino, and N2N2 -> A'A', which often lies on N1N1 -> A'A');
   fluid:   the collision term of every process over n~ at the solver's steps (solid:
            produces n~, dashed: removes it; n~ = n_N1 + n_N2 + 2 n_A' + 4 n_phi), and
            in black the net rate of change of n~ that the solver integrated. For a
            process in equilibrium (rates >> H) the term at a stored state is the residual
            of a near-cancellation: only its size is meaningful;
   both:    Gamma_kd (thick grey), the momentum exchange that sets the coupled phase
            of the Lyman-alpha lengths.
With an N1-N2 splitting (`_delta` CSVs): the history continues past the solver's end (x = 100,
dash-dotted) with the late conversions, up to x = 1e5; the relaxation rate lambda of N2/N1
through N1 N1 <-> N2 N2 (dotted), whose last crossing of H is x_fo (thin dotted line).

Validity (`validity`, also used by benchmark.py). The solvers start at T_dw with the whole DW
yield in a thermal spectrum at one T_N. Neither holds at first, so every line is drawn over the
whole solution but faded where its assumptions do not hold yet (`--hide-invalid`: not drawn there),
solid from then on:
- grey band, labelled "not thermal" ("never thermal" if x_c is absent): the dark sector is not
  thermal yet (x < x_c,
  the first Gamma_kd > H, Gamma_kd momentum-transfer weighted; the DW spectrum is colder than
  thermal);
- H and Omega h^2 = 0.12: always;
- N1 (and Omega h^2) from x_DW, the end of DW production (`dw_end_x`, 90% of the yield; not marked).
  Before x_c the number is the DW number plus what the portal reactions add with thermal rates
  (2% at 20 keV, g = 1e-3; 0.2% at 10 keV, g = 2e-4);
- N2 as N1, but not before N1 N1 <-> N2 N2 relaxes N2/N1 faster than H (conv_H > 1). That relaxation
  counts every conversion, also the forward ones that exchange no momentum, and sets
  f_N2(p) = f_N1(p) whatever the spectrum, so N2 holds before thermalisation where it is fast; where it
  becomes fast only after x_c, N2 waits for x_c in the network (its N2 then comes from the thermal
  rates). The fluid splits n~ by mu_N1 = mu_N2, so there N1 too waits for conv_H > 1;
- A', phi (at their quasi-steady values, fed by the N spectrum), T_N, the gap and all rates
  (including lambda): from x_c.
If the dark sector never thermalises (x_c absent), everything but N1, N2, Omega h^2 and H stays faded.
The bands lie under every line (zorder BAND_Z).
"""
import sys
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter

LABELS = {
    "N1N1_N2N2": r"$N_1N_1\to N_2N_2$", "N1N1_AA": r"$N_1N_1\to A'A'$", "N2N2_AA": r"$N_2N_2\to A'A'$",
    "N1N2_Aphi": r"$N_1N_2\to A'\phi$", "AA_phiphi": r"$A'A'\to\phi\phi$", "N1N1_phiphi": r"$N_1N_1\to\phi\phi$",
    "A_N1N2": r"$A'\to N_1N_2$", "phi_AA": r"$\phi\to A'A'$", "phi_N1N1": r"$\phi\to N_1N_1$",
    "N2nu_A": r"$N_2\nu\to A'$", "N1nu_phi": r"$N_1\nu\to\phi$", "N2nu_N1N2": r"$N_2\nu\to N_1N_2$ (RIS)",
    "N1nu_N2N2": r"$N_1\nu\to N_2N_2$", "N1nu_AA": r"$N_1\nu\to A'A'$ (RIS)", "N2nu_Aphi": r"$N_2\nu\to A'\phi$",
    "N1nu_phiphi": r"$N_1\nu\to\phi\phi$", "NN_AA": r"$NN\to A'A'$",
}
SPECIES_COLOURS = {"N1": "#7bc043", "N2": "#2e7d32", "A": "#f37736", "phi": "#8e44ad"}
SPECIES_LS = {"N2": "--"}   # N2 often lies on N1
RATE_LS = {"N2N2_AA": "--"}   # N2N2 -> A'A' often lies on N1N1 -> A'A'
# one colour per reaction, the same for both solvers and in every figure (the first five are the
# draft's); NN_AA is the fluid's N1N1_AA + N1N2_AA + N2N2_AA
RATE_COLOURS = {
    "A_N1N2": "#797ef6", "N2nu_A": "#1aa7ec", "NN_AA": "#4adede", "N1N1_AA": "#4adede",
    "N2N2_AA": "#1e2f97", "N1N1_N2N2": "#ffa62b", "N1nu_phi": "#2a9d8f", "phi_AA": "#9b5de5",
    "AA_phiphi": "#e76f51", "N1N2_Aphi": "#8d6e63", "N1nu_N2N2": "#606c38", "N2nu_N1N2": "#bc6c25",
    "N1nu_AA": "#6d597a", "N2nu_Aphi": "#b5838d", "N1nu_phiphi": "#264653", "phi_N1N1": "#a8dadc",
    "N1N1_phiphi": "#457b9d",
}
# computed with exact quantum statistics in the fluid (the others come from the MB tables)
FLUID_EXACT = {"N2nu_A", "N1nu_phi", "NN_AA"}
PORTAL = {"N2nu_A", "N1nu_phi", "N2nu_N1N2", "N1nu_N2N2", "N1nu_AA", "N2nu_Aphi", "N1nu_phiphi"}
FLOOR = 1e-6   # rate curves that never exceed this (in units of H) are left out
W_PHI = 4      # N per phi for m_phi > 2 m_A' (`phi_weight` in network.jl)
FADE = 0.28    # opacity of a line where its assumptions do not hold yet
HIDE_INVALID = False   # True (--hide-invalid): draw nothing there
BAND_Z = -100   # the validity bands, under every line
C_NT = "0.9"   # band: not thermal yet


def dw_end_x(m_keV, frac=0.9):
    """
    x = m_N/T_nu at which DW production has made `frac` of its yield. The DW rate per Hubble time,
    d ln Y/d ln T ∝ y^3/(1 + y^6/3)^2 with y = T/T_dw (thermal potential ∝ p T^5 against
    m^2; one momentum, constant g*), peaks at T_dw = 133 MeV (m_N/keV)^(1/3) (T_d_dw in
    constants_functions.jl, the solvers' start). With w = y^3 the yield still to come below y is
    ∫_0^w dw'/(1 + w'^2/3)^2 over its total 3^(1/2) pi/4; frac = 0.9 is y = 0.515.
    """
    a = np.sqrt(3.0)

    def below(w):   # ∫_0^w dw/(1 + w^2/3)^2
        return a / 2 * (np.arctan(w / a) + (w / a) / (1 + (w / a) ** 2))
    total = a * np.pi / 4
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if below(mid) < (1 - frac) * total else (lo, mid)
    y = lo ** (1 / 3)
    return m_keV * 1e-6 / (y * 0.133 * m_keV ** (1 / 3))


def _first_above(x, r):
    i = np.flatnonzero(r >= 1)
    return x[i[0]] if len(i) else None


def validity(d, title, solver):
    """
    Where each line of a benchmark holds (see the header): (bands, start) with bands =
    (x0, x_DW, x_c, x_kd) (x_c, x_kd None if absent) and start[key] the x from which line `key`
    ("N1", "N2", "A", "phi", "omega", "T", "rates") is valid (np.inf: never).
    """
    x = d["x"]
    m_keV = float(re.search(r"m_N = ([\d.eE+-]+) keV", title).group(1))
    x0, x_dw = x[0], dw_end_x(m_keV)
    kd = d["Gamma_kd_H"]
    x_c = _first_above(x, kd)
    x_kd = None
    if x_c is not None:
        on = np.flatnonzero(kd >= 1)
        x_kd = x[on[-1] + 1] if on[-1] + 1 < len(x) else None
    # nothing holds before DW production has finished: the state there assumes it has
    therm = max(x_c, x_dw) if x_c is not None else np.inf
    n_start = max(x_dw, x0)
    start = {"omega": n_start, "N1": n_start, "N2": therm, "A": therm, "phi": therm, "T": therm, "rates": therm}
    if "conv_H" in d.dtype.names:   # (CSVs before 2026-11-05 lack it: N2 from x_c as before)
        x_conv = _first_above(x, d["conv_H"])
        x_conv = np.inf if x_conv is None else x_conv
        if solver == "fluid":   # its N1/N2 split assumes the relaxation
            start["N1"] = start["N2"] = max(n_start, x_conv)
        else:
            start["N2"] = max(n_start, min(x_conv, therm))
    return (x0, x_dw, x_c, x_kd), start


def plot_valid(ax, x, y, x_from, **kw):
    """
    y against x: faded (FADE) before x_from (HIDE_INVALID: not drawn), as given from it on (both
    include the joint).
    """
    if x_from <= x[0]:
        return ax.plot(x, y, **kw)
    i = np.searchsorted(x, x_from)
    if not HIDE_INVALID:
        kw_f = dict(kw, alpha=FADE * kw.get("alpha", 1.0))
        kw_f.pop("label", None)
        ax.plot(x[:min(i + 1, len(x))], y[:min(i + 1, len(x))], **kw_f)
    if i < len(x):
        return ax.plot(x[max(i - 1, 0):], y[max(i - 1, 0):], **kw)
    return ax.plot([], [], **kw)   # never valid: only the legend entry


def shade_validity(axes, bands, x_end, label_ax=None):
    """The not-yet-thermal band on `axes`, under every line; labels on label_ax."""
    x0, x_dw, x_c, x_kd = bands
    nt_end = x_c if x_c is not None else x_end
    for a in axes:
        if nt_end > x0:
            a.axvspan(x0, nt_end, color=C_NT, lw=0, zorder=BAND_Z)
    if label_ax is None:
        return
    tr = label_ax.get_xaxis_transform()
    kw = dict(transform=tr, fontsize=7, color="0.3", va="top")
    if nt_end > x0:
        label_ax.text(np.sqrt(x0 * nt_end), 0.97, "not thermal" if x_c is not None else "never thermal",
                      ha="center", **kw)


def omega_parts(d):
    """
    Contributions of N1, N2, A', phi to Omega h^2 = k (Y_N1 + Y_N2 + 2 Y_A' + w_phi Y_phi),
    k = m_N s0 / rho_crit0_h2. k and w_phi are fitted to the solver's omega_h2 column (exact to
    the CSV's digits); w_phi only where phi carries a visible share, W_PHI otherwise.
    """
    om = d["omega_h2"]
    A = np.stack([d["Y_N1"] + d["Y_N2"] + 2 * d["Y_A"], d["Y_phi"]], 1) / om[:, None]
    (k, kw), *_ = np.linalg.lstsq(A, np.ones_like(om), rcond=None)
    if np.max(kw * A[:, 1]) < 1e-3:
        k = np.median(1 / A[:, 0])
        kw = W_PHI * k
    return k * d["Y_N1"], k * d["Y_N2"], 2 * k * d["Y_A"], kw * d["Y_phi"], kw / k


def plot(path, fig_dir):
    title = open(path).readline().lstrip("# ").strip()
    d = np.genfromtxt(path, delimiter=",", names=True, skip_header=1, deletechars="")
    solver = title.split()[0]
    x = d["x"]
    H = d["H_GeV"]
    fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(9, 12.5), sharex=True,
                                         gridspec_kw={"height_ratios": [1.2, 0.5, 0.5, 1.5]})

    bands, start = validity(d, title, solver)
    shade_validity((a1, a2, a3, a4), bands, x[-1], label_ax=a1)
    om1, om2, omA, omphi, w_phi = omega_parts(d)
    for o, key, lab in ((om1, "N1", r"$N_1$"), (om2, "N2", r"$N_2$"), (omA, "A", r"$A'$ ($\times 2\,N$)"),
                        (omphi, "phi", rf"$\phi$ ($\times {w_phi:.3g}\,N$)")):
        plot_valid(a1, x, o, start[key], color=SPECIES_COLOURS[key], ls=SPECIES_LS.get(key, "-"), label=lab)
    plot_valid(a1, x, d["omega_h2"], start["omega"], color="k", ls="--", lw=1.3, label=r"total $\Omega h^2$")
    a1.axhline(0.12, color="0.5", ls=":", lw=1)
    o_all = np.concatenate([om1, om2, omA])
    a1.set_ylim(max(o_all[o_all > 0].min(), 1e-30 * d["omega_h2"].max()), 30 * d["omega_h2"].max())
    a1.set_yscale("log")
    a1.set_ylabel(r"contribution to $\Omega h^2$")
    a1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    a1.set_title(f"{title}   ($\\Omega h^2$ = {d['omega_h2'][-1]:.3f})".replace("sin^2 2theta", r"$\sin^2 2\theta$"),
                 fontsize=9)

    plot_valid(a2, x, d["TN_Tnu"], start["T"], color="k", ls="-", lw=1.2)
    a2.set_yscale("log")
    a2.set_ylabel(r"$T_N/T_\nu$")

    g = d["gap_N1"]
    plot_valid(a3, x, g, start["T"], color="k", ls="-", lw=1.2)
    a3.set_yscale("symlog", linthresh=1.0)
    # the gap can be < 0 (degenerate N) in the fluid; label 1-2-5 steps within the data
    ticks = [t * 10**e * sg for sg in (-1, 1) for e in range(-1, 4) for t in (1, 2, 5)] + [0.0]
    a3.yaxis.set_major_locator(FixedLocator(sorted(t for t in ticks if g.min() / 1.5 <= t <= 1.5 * g.max()
                                                   or g.min() * 1.5 <= t <= g.max() / 1.5)))
    a3.yaxis.set_major_formatter(ScalarFormatter())
    a3.set_ylim(g.min() / 1.5 if g.min() > 0 else 1.5 * g.min(), 1.5 * g.max())
    a3.set_ylabel(r"$(m_N - \mu_{N_1})/T_N$")

    # the CSV stores rates over H; plotted in GeV, dropping points below FLOOR H
    # per process: the network's forward rates, the fluid's collision terms (what it integrates)
    rate_cols = [n for n in d.dtype.names if n.startswith("fwd:" if solver == "network" else "net:")]
    for n in rate_cols:
        name = n.split(":", 1)[1]
        r = d[n]
        if not np.any(np.abs(r) > FLOOR):
            continue
        lab = LABELS.get(name, name) + (" (exact)" if solver == "fluid" and name in FLUID_EXACT else "")
        c = RATE_COLOURS.get(name, "0.3")
        if solver == "network":
            plot_valid(a4, x, np.where(r > FLOOR, r * H, np.nan), start["rates"], ls=RATE_LS.get(name, "--" if name in PORTAL else "-"),
                       color=c, lw=1.2, label=lab)
        else:
            plot_valid(a4, x, np.where(r > FLOOR, r * H, np.nan), start["rates"], ls="-", color=c, lw=0.9, alpha=0.8, label=lab)
            plot_valid(a4, x, np.where(r < -FLOOR, -r * H, np.nan), start["rates"], ls="--", color=c, lw=0.9, alpha=0.8)
    if solver == "fluid":
        tot = d["net_total"]
        plot_valid(a4, x, np.where(tot > FLOOR, tot * H, np.nan), start["rates"], color="k", ls="-", lw=1.5,
                   label=r"net $\dot{\tilde n}$ (solution)")
        plot_valid(a4, x, np.where(tot < -FLOOR, -tot * H, np.nan), start["rates"], color="k", ls="--", lw=1.5)
    a4.plot(x, d["Gamma_kd_H"] * H, color="0.6", lw=4, alpha=0.7, label=r"$\Gamma_\mathrm{kd}$", zorder=1)
    a4.plot(x, H, color="k", lw=3.5, label=r"$H$", zorder=10)
    if "conv_H" in d.dtype.names and "delta =" in title:   # equal masses: a multiple of N1N1 -> N2N2
        plot_valid(a4, x, np.where(d["conv_H"] > FLOOR, d["conv_H"] * H, np.nan), start["rates"], ls=":",
                   color=RATE_COLOURS["N1N1_N2N2"], lw=2.2, label=r"$\lambda$: $N_1N_1\leftrightarrow N_2N_2$ relaxation")
    split = "delta =" in title
    if split:
        # the solver's end: after it the late conversions (with_conversions), rates of the solver absent
        x_fo = re.search(r"x_fo = ([\d.eE+-]+|NaN)", title)
        for a in (a1, a2, a3, a4):
            a.axvline(100, color="0.4", ls="-.", lw=0.8)
            if x_fo and np.isfinite(float(x_fo.group(1))):
                a.axvline(float(x_fo.group(1)), color=RATE_COLOURS["N1N1_N2N2"], ls=":", lw=1)
    a4.set_yscale("log")
    top = max(np.nanmax(np.abs(d[n]) * H) for n in rate_cols + ["Gamma_kd_H"] + (["net_total"] if solver == "fluid" else []))
    a4.set_ylim(FLOOR * H.min(), 30 * top)
    a4.set_ylabel(r"$\Gamma_\mathrm{fwd}$ per particle [GeV]" if solver == "network"
                  else r"$|C_{\tilde n}|/\tilde n$ [GeV] (dashed: $< 0$)")
    a4.set_xscale("log")
    a4.set_xlabel(r"$x = m_N/T_\nu$")
    a4.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    for a in (a1, a2, a3, a4):
        a.minorticks_off()
    fig.tight_layout()
    out = os.path.join(fig_dir, os.path.basename(path)[:-4] + ".pdf")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--hide-invalid" in args:
        args.remove("--hide-invalid")
        HIDE_INVALID = True
    src = args[0] if len(args) > 0 else "tmp/benchmarks"
    fig_dir = args[1] if len(args) > 1 else "figures/benchmarks_debug"
    os.makedirs(fig_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, "*.csv"))) if os.path.isdir(src) else [src]:
        plot(p, fig_dir)

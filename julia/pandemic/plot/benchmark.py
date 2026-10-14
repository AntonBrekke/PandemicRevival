"""
Benchmark figures from src/run_benchmarks.jl: one PDF per point, for either solver.

    python plot/benchmark.py [csv | dir] [fig_dir]

Defaults: tmp/benchmarks -> figures/benchmarks. Panels, against x = m_N/T_nu:
1. abundances Y = n/s of N1, N2, A', phi, and Omega h^2 (right axis);
2. T_N/T_nu and (m_N - mu_N1)/T_N (right axis);
3. the rates the solver computes, over H (the line at 1 is H):
   network: forward rate per particle of every reaction (solid: dark sector, dashed:
            with a neutrino);
   fluid:   the collision term of every process over n~ H at the solver's steps (solid:
            produces n~, dashed: removes it; n~ = n_N1 + n_N2 + 2 n_A' + 4 n_phi), and
            in black the net rate of change of n~ that the solver integrated. For a
            process in equilibrium (rates >> H) the term at a stored state is the residual
            of a near-cancellation: only its size is meaningful;
   both:    Gamma_kd/H (thick grey), the momentum exchange that sets the coupled phase
            of the Lyman-alpha lengths.
"""
import sys
import os
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
# computed with exact quantum statistics in the fluid (the others come from the MB tables)
FLUID_EXACT = {"N2nu_A", "N1nu_phi", "NN_AA"}
PORTAL = {"N2nu_A", "N1nu_phi", "N2nu_N1N2", "N1nu_N2N2", "N1nu_AA", "N2nu_Aphi", "N1nu_phiphi"}
FLOOR = 1e-6   # rate curves that never exceed this (in units of H) are left out


def plot(path, fig_dir):
    title = open(path).readline().lstrip("# ").strip()
    d = np.genfromtxt(path, delimiter=",", names=True, skip_header=1, deletechars="")
    solver = title.split()[0]
    x = d["x"]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 11), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.6, 1.4]})

    for col, lab, c in (("Y_N1", r"$N_1$", "C0"), ("Y_N2", r"$N_2$", "C1"), ("Y_A", r"$A'$", "C2"), ("Y_phi", r"$\phi$", "C3")):
        a1.plot(x, d[col], color=c, label=lab)
    y_all = np.concatenate([d[c] for c in ("Y_N1", "Y_N2", "Y_A")])
    a1.set_ylim(max(y_all[y_all > 0].min(), 1e-30), 3 * y_all.max())
    a1.set_yscale("log")
    a1.set_ylabel(r"$Y = n/s$")
    b1 = a1.twinx()
    b1.plot(x, d["omega_h2"], "k-.", lw=1, label=r"$\Omega h^2$")
    b1.set_yscale("log")
    # Omega h^2 can be constant to 1e-4 (all of it DW-produced): do not zoom into the rounding
    b1.set_ylim(1e-4 * d["omega_h2"][-1], 3 * d["omega_h2"][-1])
    b1.set_ylabel(r"$\Omega h^2$ (dash-dotted)")
    a1.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=8)
    a1.set_title(f"{title}   ($\\Omega h^2$ = {d['omega_h2'][-1]:.3f})".replace("sin^2 2theta", r"$\sin^2 2\theta$"),
                 fontsize=9)

    a2.plot(x, d["TN_Tnu"], "k-", lw=1.2)
    a2.set_yscale("log")
    a2.set_ylabel(r"$T_N/T_\nu$ (solid)")
    b2 = a2.twinx()
    b2.plot(x, d["gap_N1"], color="0.5", ls="--", lw=1.2)
    b2.set_yscale("symlog", linthresh=1.0)
    # the gap can be < 0 (degenerate N) in the fluid; label 1-2-5 steps within the data
    g = d["gap_N1"]
    ticks = [t * 10**e * sg for sg in (-1, 1) for e in range(-1, 4) for t in (1, 2, 5)] + [0.0]
    b2.yaxis.set_major_locator(FixedLocator(sorted(t for t in ticks if g.min() / 1.5 <= t <= 1.5 * g.max()
                                                   or g.min() * 1.5 <= t <= g.max() / 1.5)))
    b2.yaxis.set_major_formatter(ScalarFormatter())
    b2.set_ylim(g.min() / 1.5 if g.min() > 0 else 1.5 * g.min(), 1.5 * g.max())
    b2.set_ylabel(r"$(m_N - \mu_{N_1})/T_N$ (dashed)")

    cmap = plt.get_cmap("tab20")
    rate_cols = [n for n in d.dtype.names if n.startswith(("fwd:", "net:"))]   # per process
    for i, n in enumerate(rate_cols):
        name = n.split(":", 1)[1]
        r = d[n]
        if not np.any(np.abs(r) > FLOOR):
            continue
        lab = LABELS.get(name, name) + (" (exact)" if solver == "fluid" and name in FLUID_EXACT else "")
        c = cmap((2 * i) % 20 + (i // 10) % 2)
        if solver == "network":
            a3.plot(x, np.where(r > 0, r, np.nan), ls="--" if name in PORTAL else "-", color=c, lw=1.2, label=lab)
        else:
            a3.plot(x, np.where(r > 0, r, np.nan), "-", color=c, lw=0.9, alpha=0.8, label=lab)
            a3.plot(x, np.where(r < 0, -r, np.nan), "--", color=c, lw=0.9, alpha=0.8)
    if solver == "fluid":
        tot = d["net_total"]
        a3.plot(x, np.where(tot > 0, tot, np.nan), "k-", lw=2.2, label=r"net $\dot{\tilde n}$ (solution)")
        a3.plot(x, np.where(tot < 0, -tot, np.nan), "k--", lw=2.2)
    a3.plot(x, d["Gamma_kd_H"], color="0.6", lw=4, alpha=0.7, label=r"$\Gamma_\mathrm{kd}/H$", zorder=1)
    a3.axhline(1, color="k", lw=2.5, zorder=10)
    a3.set_yscale("log")
    top = max(np.nanmax(np.abs(d[n])) for n in rate_cols + ["Gamma_kd_H"] + (["net_total"] if solver == "fluid" else []))
    a3.set_ylim(FLOOR, min(30 * top, 1e26))
    a3.set_ylabel(r"$\Gamma_\mathrm{fwd}/H$ per particle" if solver == "network"
                  else r"$|C_{\tilde n}|/(\tilde n H)$ (dashed: $< 0$)")
    a3.set_xscale("log")
    a3.set_xlabel(r"$x = m_N/T_\nu$")
    a3.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    for a in (a1, a2, a3, b1, b2):
        a.minorticks_off()
    fig.tight_layout()
    out = os.path.join(fig_dir, os.path.basename(path)[:-4] + ".pdf")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


src = sys.argv[1] if len(sys.argv) > 1 else "tmp/benchmarks"
fig_dir = sys.argv[2] if len(sys.argv) > 2 else "figures/benchmarks"
os.makedirs(fig_dir, exist_ok=True)
for p in sorted(glob.glob(os.path.join(src, "*.csv"))) if os.path.isdir(src) else [src]:
    plot(p, fig_dir)

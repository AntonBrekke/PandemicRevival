"""
Benchmark figures from src/run_benchmarks.jl: one PDF per point, for either solver.

    python plot/benchmark.py [csv | dir] [fig_dir]

Defaults: tmp/benchmarks -> figures/benchmarks. Panels, against x = m_N/T_nu, one y axis each:
1. the contribution of each species to Omega h^2, counting the N it decays into (A' -> 2 N,
   phi -> w_phi N), and their sum Omega h^2 (black dashed; dotted: 0.12);
2. T_N/T_nu;
3. (m_N - mu_N1)/T_N;
4. the rates the solver computes, in GeV, with H as the thick black line:
   network: forward rate per particle of every reaction (solid: dark sector, dashed:
            with a neutrino);
   fluid:   the collision term of every process over n~ at the solver's steps (solid:
            produces n~, dashed: removes it; n~ = n_N1 + n_N2 + 2 n_A' + 4 n_phi), and
            in black the net rate of change of n~ that the solver integrated. For a
            process in equilibrium (rates >> H) the term at a stored state is the residual
            of a near-cancellation: only its size is meaningful;
   both:    Gamma_kd (thick grey), the momentum exchange that sets the coupled phase
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
W_PHI = 4      # N per phi for m_phi > 2 m_A' (`phi_weight` in network.jl)


def omega_parts(d):
    """
    Contributions of N1, N2, A', phi to Omega h^2 = k (Y_N1 + Y_N2 + 2 Y_A' + w_phi Y_phi),
    k = m_N s0 / rho_crit0_h2. k and w_phi are fitted to the solver's omega_h2 column (exact to
    the CSV's 5 digits); w_phi only where phi carries a visible share, W_PHI otherwise.
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

    om1, om2, omA, omphi, w_phi = omega_parts(d)
    for o, lab, c in ((om1, r"$N_1$", "C0"), (om2, r"$N_2$", "C1"), (omA, r"$A'$ ($\times 2\,N$)", "C2"),
                      (omphi, rf"$\phi$ ($\times {w_phi:.3g}\,N$)", "C3")):
        a1.plot(x, o, color=c, label=lab)
    a1.plot(x, d["omega_h2"], "k--", lw=1.3, label=r"total $\Omega h^2$")
    a1.axhline(0.12, color="0.5", ls=":", lw=1)
    o_all = np.concatenate([om1, om2, omA])
    a1.set_ylim(max(o_all[o_all > 0].min(), 1e-30 * d["omega_h2"].max()), 30 * d["omega_h2"].max())
    a1.set_yscale("log")
    a1.set_ylabel(r"contribution to $\Omega h^2$")
    a1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    a1.set_title(f"{title}   ($\\Omega h^2$ = {d['omega_h2'][-1]:.3f})".replace("sin^2 2theta", r"$\sin^2 2\theta$"),
                 fontsize=9)

    a2.plot(x, d["TN_Tnu"], "k-", lw=1.2)
    a2.set_yscale("log")
    a2.set_ylabel(r"$T_N/T_\nu$")

    g = d["gap_N1"]
    a3.plot(x, g, "k-", lw=1.2)
    a3.set_yscale("symlog", linthresh=1.0)
    # the gap can be < 0 (degenerate N) in the fluid; label 1-2-5 steps within the data
    ticks = [t * 10**e * sg for sg in (-1, 1) for e in range(-1, 4) for t in (1, 2, 5)] + [0.0]
    a3.yaxis.set_major_locator(FixedLocator(sorted(t for t in ticks if g.min() / 1.5 <= t <= 1.5 * g.max()
                                                   or g.min() * 1.5 <= t <= g.max() / 1.5)))
    a3.yaxis.set_major_formatter(ScalarFormatter())
    a3.set_ylim(g.min() / 1.5 if g.min() > 0 else 1.5 * g.min(), 1.5 * g.max())
    a3.set_ylabel(r"$(m_N - \mu_{N_1})/T_N$")

    # the CSV stores rates over H; plotted in GeV, dropping points below FLOOR H
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
            a4.plot(x, np.where(r > FLOOR, r * H, np.nan), ls="--" if name in PORTAL else "-", color=c, lw=1.2,
                    label=lab)
        else:
            a4.plot(x, np.where(r > FLOOR, r * H, np.nan), "-", color=c, lw=0.9, alpha=0.8, label=lab)
            a4.plot(x, np.where(r < -FLOOR, -r * H, np.nan), "--", color=c, lw=0.9, alpha=0.8)
    if solver == "fluid":
        tot = d["net_total"]
        a4.plot(x, np.where(tot > FLOOR, tot * H, np.nan), "k-", lw=1.5, label=r"net $\dot{\tilde n}$ (solution)")
        a4.plot(x, np.where(tot < -FLOOR, -tot * H, np.nan), "k--", lw=1.5)
    a4.plot(x, d["Gamma_kd_H"] * H, color="0.6", lw=4, alpha=0.7, label=r"$\Gamma_\mathrm{kd}$", zorder=1)
    a4.plot(x, H, color="k", lw=3.5, label=r"$H$", zorder=10)
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
    src = sys.argv[1] if len(sys.argv) > 1 else "tmp/benchmarks"
    fig_dir = sys.argv[2] if len(sys.argv) > 2 else "figures/benchmarks"
    os.makedirs(fig_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, "*.csv"))) if os.path.isdir(src) else [src]:
        plot(p, fig_dir)

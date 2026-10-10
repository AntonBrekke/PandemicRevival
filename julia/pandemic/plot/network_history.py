"""
Benchmark plot of the species-resolved network: abundances of N1, N2, A', phi
(top) and the forward rate per particle over H of every reaction (bottom).

    python plot/network_history.py <history.csv | dir> [fig_dir]

Input from src/run_network_history.jl. Rates: Gamma_fwd per particle of the
first dark initial species, over H; solid lines dark-sector reactions, dashed
portal reactions (with nu). Resonance-subtracted rates are shown as |.|.
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "N1N1_N2N2": r"$N_1N_1\to N_2N_2$", "N1N1_AA": r"$N_1N_1\to A'A'$", "N2N2_AA": r"$N_2N_2\to A'A'$",
    "N1N2_Aphi": r"$N_1N_2\to A'\phi$", "AA_phiphi": r"$A'A'\to\phi\phi$", "N1N1_phiphi": r"$N_1N_1\to\phi\phi$",
    "A_N1N2": r"$A'\to N_1N_2$", "phi_AA": r"$\phi\to A'A'$", "phi_N1N1": r"$\phi\to N_1N_1$",
    "N2nu_A": r"$N_2\nu\to A'$", "N1nu_phi": r"$N_1\nu\to\phi$", "N2nu_N1N2": r"$N_2\nu\to N_1N_2$ (RIS)",
    "N1nu_N2N2": r"$N_1\nu\to N_2N_2$", "N1nu_AA": r"$N_1\nu\to A'A'$ (RIS)", "N2nu_Aphi": r"$N_2\nu\to A'\phi$",
    "N1nu_phiphi": r"$N_1\nu\to\phi\phi$",
}
PORTAL = {"N2nu_A", "N1nu_phi", "N2nu_N1N2", "N1nu_N2N2", "N1nu_AA", "N2nu_Aphi", "N1nu_phiphi"}


def plot(path, fig_dir):
    d = np.genfromtxt(path, delimiter=",", names=True)
    x = d["x"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    for col, lab, c in (("Y_N1", r"$N_1$", "C0"), ("Y_N2", r"$N_2$", "C1"), ("Y_A", r"$A'$", "C2"), ("Y_phi", r"$\phi$", "C3")):
        a1.plot(x, d[col], color=c, label=lab)
    a1.set_yscale("log")
    y_all = np.concatenate([d[c] for c in ("Y_N1", "Y_N2", "Y_A")])
    a1.set_ylim(max(y_all[y_all > 0].min(), 1e-30), 3 * y_all.max())
    a1.set_ylabel(r"$Y = n/s$")
    ax2 = a1.twinx()
    ax2.plot(x, d["TN_Tnu"], "k:", lw=1)
    ax2.set_ylabel(r"$T_N/T_\nu$ (dotted)")
    ax2.set_yscale("log")
    a1.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=8)
    base = os.path.basename(path)[:-4]
    a1.set_title(base.replace("_", " ") + f"   ($\\Omega h^2$ = {d['omega_h2'][-1]:.3f})", fontsize=9)
    cmap = plt.get_cmap("tab20")
    names = [n for n in d.dtype.names if n in LABELS]
    for i, n in enumerate(names):
        r = d[n]
        if not np.any(r > 1e-6):
            continue
        a2.plot(x, np.where(r > 0, r, np.nan), ls="--" if n in PORTAL else "-", color=cmap(i % 20), label=LABELS[n], lw=1.2)
    a2.axhline(1, color="k", lw=0.8)
    a2.set_yscale("log")
    a2.set_ylim(1e-6, 1e14)
    a2.set_ylabel(r"$\Gamma/H$ per particle")
    a2.set_xscale("log")
    a2.set_xlabel(r"$x = m_N/T_\nu$")
    a2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    for a in (a1, a2, ax2):
        a.minorticks_off()
    fig.tight_layout()
    out = os.path.join(fig_dir, "network_history_" + base + ".pdf")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


src = sys.argv[1]
fig_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(fig_dir, exist_ok=True)
for p in sorted(glob.glob(os.path.join(src, "*.csv"))) if os.path.isdir(src) else [src]:
    plot(p, fig_dir)

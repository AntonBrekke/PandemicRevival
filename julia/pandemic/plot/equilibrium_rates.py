#! /usr/bin/env python3
"""
Gamma/H along benchmark histories, from src/run_equilibrium_check.jl --benchmarks.

    conda run -n pandemic python plot/equilibrium_rates.py [tmp/equilibrium] [figures/equilibrium]

One figure per benchmark CSV, three panels against x = m_N/T_nu:
  (a) dark sector (assumptions A1-A3): N1N1 -> N2N2, N1N2 -> N1N2 (with and
      without its on-shell A'), A' -> N1N2 and its inverse, A'A' <-> NN;
  (b) portal (A5): the included N2 nu -> A' and every neglected portal process,
      weighted by the change |Delta n~| it makes;
  (c) phi production per dark particle (A6), and T_N/T_nu, m_N/T_N.
The grey band marks Gamma/H < 1; the dotted line the end of the production
window (x_99, from the .summary file).
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DELTA = {"N1nu_N2N2": 1, "N2nu_N1N2": 1, "N1nu_AA": 3, "N2nu_Aphi": 1, "N1nu_phiphi": 1, "N1nu_phi": 1}
LABELS = {
    "conv": r"$N_1N_1\to N_2N_2$", "el12": r"$N_1N_2\to N_1N_2$",
    "el12_offshell": r"$N_1N_2\to N_1N_2$ (off-shell)", "A_dec": r"$A'\to N_1N_2$",
    "A_inv": r"$N_1N_2\to A'$ (per $N_1$)", "NN_AA": r"$N_1N_1\to A'A'$", "AA_NN": r"$A'A'\to N_1N_1$",
    "portal_in": r"$N_2\nu\to A'$ (in solver)", "N1nu_N2N2": r"$N_1\nu\to N_2N_2$",
    "N2nu_N1N2": r"$N_2\nu\to N_1N_2$ (off-shell)", "N1nu_AA": r"$N_1\nu\to A'A'$ ($\times3$)",
    "N2nu_Aphi": r"$N_2\nu\to A'\phi$", "N1nu_phiphi": r"$N_1\nu\to\phi\phi$", "N1nu_phi": r"$N_1\nu\to\phi$",
    "phi_prod": r"$\phi$ production / dark particle",
}


def load(path):
    with open(path) as f:
        head = f.readline().strip("# \n")
    d = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
    summ = {}
    if os.path.exists(path + ".summary"):
        for line in open(path + ".summary"):
            k, v = line.strip().split(",", 1)
            try:
                summ[k] = float(v)
            except ValueError:
                pass
    return head, d, summ


def positive(y):
    y = np.array(y, dtype=float)
    y[~(y > 0)] = np.nan
    return y


def plot_one(path, out_dir):
    head, d, summ = load(path)
    x = d["x"]
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 9.0), sharex=True)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhspan(1e-300, 1, color="0.92", zorder=0)
        ax.axhline(1, color="0.5", lw=0.8)
        if "x_99" in summ:
            ax.axvline(summ["x_99"], color="0.3", ls=":", lw=1)
    for k in ("conv", "el12", "el12_offshell", "A_dec", "A_inv", "NN_AA", "AA_NN"):
        axs[0].plot(x, positive(d[k]), label=LABELS[k], ls="--" if k == "el12_offshell" else "-")
    axs[0].set_ylabel(r"$\Gamma/H$ (dark sector)")
    axs[1].plot(x, positive(d["portal_in"]), color="k", lw=2, label=LABELS["portal_in"])
    for k, w in DELTA.items():
        axs[1].plot(x, positive(w * d[k]), label=LABELS[k])
    axs[1].set_ylabel(r"$|\Delta\tilde n|\,\Gamma/H$ (portal)")
    axs[2].plot(x, positive(d["phi_prod"]), color="C3", label=LABELS["phi_prod"])
    axs[2].plot(x, d["TN_Tnu"], color="0.3", ls="--", label=r"$T_N/T_\nu$")
    axs[2].plot(x, d["xd"], color="0.6", ls="-.", label=r"$m_N/T_N$")
    axs[2].set_ylabel(r"$\Gamma_\phi/H$")
    axs[2].set_xlabel(r"$x = m_N/T_\nu$")
    for ax in axs:
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(lo, 1e-12), min(max(hi, 10), 1e30))
        ax.legend(fontsize=6.5, loc="best", ncol=2)
    axs[0].set_title(head.replace(" ", ",  "), fontsize=7)
    fig.tight_layout()
    stem = os.path.join(out_dir, os.path.basename(path)[:-4])
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    return stem


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "tmp/equilibrium"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/equilibrium"
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, "*.csv"))):
        print("wrote", plot_one(f, out) + ".pdf")

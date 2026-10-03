#! /usr/bin/env python3
"""
Maps of the equilibrium check over the relic scan (src/run_equilibrium_check.jl
--scan, <scan_dir>/equilibrium.csv), in the (m_N, g) plane of the roots.

    conda run -n pandemic python plot/equilibrium_scan.py [scan_dir] [out_stem]

Four panels, colour = log10 of
  (a) min Gamma/H of N1N1 -> N2N2 over the production window: keeps
      mu_N1 = mu_N2, i.e. the N2 population the pandemic feeds on (A2);
  (b) min Gamma/H of A' -> N1N2 over the window (A3, mu_A' = 2 mu_N);
  (c) e-folds of growth from all neglected portal processes over those from
      N2 nu -> A' (A5);
  (d) F_phi, the fraction of the dark sector converted into phi (A6).
Open circles: the assumption fails (< 1 in (a), (b); > 0.1 in (c); > 0.1 in (d)).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PORTAL = ("N1nu_N2N2", "N2nu_N1N2", "N1nu_AA", "N2nu_Aphi", "N1nu_phiphi", "N1nu_phi")


def load(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def main():
    scan = sys.argv[1] if len(sys.argv) > 1 else "tmp/relic_scan_theta_merged"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/equilibrium_scan"
    d = load(os.path.join(scan, "equilibrium.csv"))
    m = d["m_N"] * 1e6
    y = d["y"]
    neg = sum(d["efolds_" + k] for k in PORTAL)
    rel = neg / np.maximum(d["efolds_portal_in"], 1e-300)
    panels = [
        (np.log10(d["min_conv"]), d["min_conv"] < 1, r"$\log_{10}\min\,\Gamma_{N_1N_1\to N_2N_2}/H$"),
        (np.log10(d["min_A_dec"]), d["min_A_dec"] < 1, r"$\log_{10}\min\,\Gamma_{A'\to N_1N_2}/H$"),
        (np.log10(np.maximum(rel, 1e-12)), rel > 0.1, r"$\log_{10}$ e-folds neglected / included portal"),
        (np.log10(np.maximum(d["F_phi"], 1e-12)), d["F_phi"] > 0.1, r"$\log_{10} F_\phi$"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True, sharey=True)
    for ax, (c, bad, lab) in zip(axs.flat, panels):
        sc = ax.scatter(m, y, c=c, s=22, cmap="viridis", vmin=np.nanpercentile(c, 2), vmax=np.nanpercentile(c, 98))
        ax.scatter(m[bad], y[bad], s=60, facecolors="none", edgecolors="r", lw=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.85).set_label(lab, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
    for ax in axs[1]:
        ax.set_xlabel(r"$m_N$ [keV]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$g$")
    fig.suptitle(r"equilibrium check, $m_\phi = 100\,m_N$, $m_\nu = 0.1$ eV, $m_0 = 1$ TeV "
                 f"({len(m)} roots; red: assumption fails)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out + ".pdf")
    print("wrote", out + ".pdf")
    # text summary
    for lab, bad in (("A2 conversions < H in window", panels[0][1]), ("A3 A' decay < H in window", panels[1][1]),
                     ("A5 neglected portal > 10%", panels[2][1]), ("A6 F_phi > 0.1", panels[3][1])):
        ys = sorted(set(np.round(np.log10(y[bad]), 2)))
        print(f"{lab}: {bad.sum()}/{len(m)} roots; log10 g in {ys[:3]}..{ys[-3:]}" if bad.any() else f"{lab}: none")


if __name__ == "__main__":
    main()

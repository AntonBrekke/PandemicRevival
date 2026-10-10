"""
N2/N1 abundance ratio today at the network's relic roots, logarithmic colour scale.

    python plot/network_n2_fraction.py [lyman_alpha_network.csv] [out_stem]

Left: in the (m_N, g) plane. Right: in the (m_N, sin^2 2theta) plane of the
relic plot. Points where the MB network does not apply (degenerate dark sector,
g >~ 5e-3) are absent. The reduced solver has N2/N1 = 1 everywhere.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

src = sys.argv[1] if len(sys.argv) > 1 else "tmp/network_scan/lyman_alpha_network.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/network_n2_over_n1"
rows = [r for r in csv.DictReader(open(src)) if np.isfinite(float(r["f_N2_network"]))]
m = np.array([1e6 * float(r["m_N"]) for r in rows])
g = np.array([float(r["y"]) for r in rows])
s2 = np.array([float(r["s2_network"]) for r in rows])
f2 = np.array([float(r["f_N2_network"]) for r in rows])
ratio = f2 / (1 - f2)
norm = LogNorm(vmin=ratio.min(), vmax=1.0)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, yv, ylab in ((a1, g, r"$g$"), (a2, s2, r"$\sin^2 2\theta$ at $\Omega h^2 = 0.12$")):
    sc = ax.scatter(m, yv, c=np.clip(ratio, norm.vmin, None), norm=norm, cmap="viridis", s=28, marker="s",
                    edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xlabel(r"$m_{N}$ [keV]"); ax.set_ylabel(ylab)
a1.axhspan(3e-4, 6e-3, color="k", alpha=0.07, lw=0)
a1.text(0.7, 1.2e-3, "reduced solver valid", fontsize=8)
a1.set_title(r"$n_{N_2}/n_{N_1}$ today (network)")
a2.set_title("same, in the relic plane")
fig.colorbar(sc, ax=(a1, a2), label=r"$n_{N_2}/n_{N_1}$")
fig.savefig(out + ".pdf")
print("written", out + ".pdf", f"(ratio range {ratio.min():.2e} - {ratio.max():.3f})")

"""
N2/N1 abundance ratio today at the network's relic roots, logarithmic colour scale.

    python plot/network_n2_fraction.py [out_stem]

From tmp/network_scan{,_fine}/lyman_alpha.csv. Left: in the (m_N, g) plane. Right: in
the (m_N, sin^2 2theta) plane of the relic plot. Points where the MB network does not
apply (degenerate dark sector, g >~ 5e-3) are absent. In the fluid (g >= 1e-3)
N2/N1 = 1 by construction.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

out = sys.argv[1] if len(sys.argv) > 1 else "figures/network_n2_over_n1"
rows = [r for p in ("tmp/network_scan/lyman_alpha.csv", "tmp/network_scan_fine/lyman_alpha.csv")
        for r in csv.DictReader(open(p)) if np.isfinite(float(r["f_N2"]))]
m = np.array([1e6 * float(r["m_N"]) for r in rows])
g = np.array([float(r["y"]) for r in rows])
s2 = np.array([float(r["sin2_2theta"]) for r in rows])
f2 = np.array([float(r["f_N2"]) for r in rows])
ratio = f2 / (1 - f2)
norm = LogNorm(vmin=ratio.min(), vmax=1.0)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, yv, ylab in ((a1, g, r"$g$"), (a2, s2, r"$\sin^2 2\theta$ at $\Omega h^2 = 0.12$")):
    sc = ax.scatter(m, yv, c=np.clip(ratio, norm.vmin, None), norm=norm, cmap="viridis", s=28, marker="s",
                    edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xlabel(r"$m_{N}$ [keV]"); ax.set_ylabel(ylab)
a1.set_title(r"$n_{N_2}/n_{N_1}$ today (network)")
a2.set_title("same, in the relic plane")
fig.colorbar(sc, ax=(a1, a2), label=r"$n_{N_2}/n_{N_1}$")
fig.savefig(out + ".pdf")
print("written", out + ".pdf", f"(ratio range {ratio.min():.2e} - {ratio.max():.3f})")

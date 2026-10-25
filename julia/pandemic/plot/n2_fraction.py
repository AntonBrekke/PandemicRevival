"""
N2/N1 abundance ratio today at the relic roots, logarithmic colour scale, one row of panels
per scan.

    python plot/n2_fraction.py [out_stem] [label=lyman_alpha.csv ...]   (split at the last "=")

Default: the equal-mass scan (tmp/combined_scan) and the scan with an N1-N2 splitting
delta = 1e-4 (tmp/delta_runs/delta1e-4/combined), both solvers (network for g < 1e-3, fluid
above; the dotted line marks the switch). Equal masses: in the fluid N2/N1 = 1 by construction.
With a splitting: N2/N1 after the late conversions N1 N1 <-> N2 N2 (`with_conversions`,
lyman_alpha.jl). Left: in the (m_N, g) plane. Right: in the (m_N, sin^2 2theta) plane of the
relic plot. Points where no solver applies are absent; ratios below VMIN are drawn at VMIN.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VMIN = 1e-8
G_SWITCH = 1e-3   # network below, fluid above
DEFAULT = [r"$m_{N_2} = m_{N_1}$=tmp/combined_scan/lyman_alpha.csv",
           r"$\delta = 10^{-4}$=tmp/delta_runs/delta1e-4/combined/lyman_alpha.csv"]


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if np.isfinite(float(r["f_N2"]))]
    col = lambda k: np.array([float(r[k]) for r in rows])
    f2 = col("f_N2")
    return 1e6 * col("m_N"), col("y"), col("sin2_2theta"), f2 / (1 - f2)


out = sys.argv[1] if len(sys.argv) > 1 else "figures/n2_over_n1"
specs = [s.rsplit("=", 1) if "=" in s else (s, s) for s in (sys.argv[2:] or DEFAULT)]
data = [(lab, load(p)) for lab, p in specs]
norm = LogNorm(vmin=VMIN, vmax=1.0)

fig, axes = plt.subplots(len(data), 2, figsize=(11, 4.2 * len(data)), squeeze=False, constrained_layout=True)
for (lab, (m, g, s2, ratio)), (a1, a2) in zip(data, axes):
    for ax, yv, ylab in ((a1, g, r"$g$"), (a2, s2, r"$\sin^2 2\theta$ at $\Omega h^2 = 0.12$")):
        sc = ax.scatter(m, yv, c=np.clip(ratio, VMIN, None), norm=norm, cmap="viridis", s=22, marker="s",
                        edgecolors="none")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
        ax.set_xlabel(r"$m_{N_1}$ [keV]"); ax.set_ylabel(ylab)
    a1.axhline(G_SWITCH, color="k", ls=":", lw=1)
    a1.set_title(rf"$n_{{N_2}}/n_{{N_1}}$ today, {lab}")
    a2.set_title(f"same, in the relic plane ({lab})")
    print(f"{lab}: {len(m)} roots, ratio {ratio.min():.2e} - {ratio.max():.3f}")
fig.colorbar(sc, ax=axes, label=r"$n_{N_2}/n_{N_1}$", shrink=0.8)
fig.savefig(out + ".pdf")
print("written", out + ".pdf")

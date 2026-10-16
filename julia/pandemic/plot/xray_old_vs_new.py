"""
The X-ray bound of the money plot, old and new, on an empty frame with the money plot's axes.

    conda run -n pandemic python plot/xray_old_vs_new.py [out_stem]

- old: the minimum over the limits used until 2026-10-13 (Chandra blank sky, dwarfs, M31,
  NuSTAR 2020, XMM blank sky, INTEGRAL 2008), recomputed here;
- new: the current combined limit (PandemicRevival/xray_constraints/overall_constraint.dat,
  written by plot_xray.py), which assumes that all the dark matter decays;
- new, rescaled: the line the money plot draws, the new limit divided by the N1 fraction of the
  dark matter at the roots of tmp/combined_scan (`scaled_xray_limit`).
Writes <out_stem>.pdf (default figures/xray_old_vs_new).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relic_scan_theta import (XRAY_DIR, M_LIM, S_LIM, COLUMNWIDTH, set_style, get_figsize, load_roots,
                              load_lyman_alpha, load_xray_limit, scaled_xray_limit)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("figures", "xray_old_vs_new")
OLD = ("CBS_2008.02283.dat", "Dwarfs_1408.3531.dat", "M31_1311.0282.dat", "NuSTAR_1908.09037.dat",
       "XMMBlankSky_2102.02207.dat", "INTEGRAL_0710.4922.dat")

m = np.logspace(np.log10(M_LIM[0]), np.log10(M_LIM[1]), 4000)   # keV
curves = [np.loadtxt(os.path.join(XRAY_DIR, f), comments="#") for f in OLD]
old = np.exp(np.min([np.interp(np.log(m * 1e-6), np.log(c[:, 0]), np.log(c[:, 1]), left=np.inf, right=np.inf)
                     for c in curves], axis=0))
m_new, s_new = load_xray_limit()
roots = load_roots("tmp/combined_scan")
_, _, mix = load_lyman_alpha("tmp/combined_scan")
m_sc, s_sc = scaled_xray_limit(roots, mix)
new_on_m = np.exp(np.interp(np.log(m), np.log(m_new), np.log(s_new), left=np.nan, right=np.nan))
sel = np.isfinite(old) & np.isfinite(new_on_m)
for lo, hi in ((2, 6), (6, 40), (40, 300)):
    r = new_on_m[sel & (m >= lo) & (m < hi)] / old[sel & (m >= lo) & (m < hi)]
    print(f"new/old {lo}-{hi} keV: median {np.median(r):.2f}, range {r.min():.2f}-{r.max():.2f}")

set_style()
fig = plt.figure(figsize=get_figsize(COLUMNWIDTH, wf=1.0, hf=0.9), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis="both", which="both", direction="in", width=0.5)
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
for s in ax.spines.values():
    s.set_linewidth(0.5)
ax.plot(m, old, color="C3", lw=1.2, label=r"old (until 2026-10-13)")
ax.plot(m_new, s_new, color="C0", lw=1.2, label=r"new, all DM decaying")
ax.plot(m_sc, s_sc, color="k", lw=1.4, ls="--", label=r"new $/\,f_{N_1}$ (money plot)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*M_LIM)
ax.set_ylim(*S_LIM)
for axis, (lo, hi) in ((ax.xaxis, M_LIM), (ax.yaxis, S_LIM)):
    decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
    axis.set_major_locator(FixedLocator(10**decades))
    axis.set_minor_locator(FixedLocator([j * 10**d for d in decades for j in range(2, 10)]))
    axis.set_major_formatter(LogFormatterMathtext())
    axis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"$m_{N_1}\;\;[\mathrm{keV}]$")
ax.set_ylabel(r"$\sin^2 (2 \theta_1)$")
ax.legend(loc="upper right", fontsize=8, frameon=True)
fig.tight_layout()
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out + ".pdf")
print("written", out + ".pdf")

"""
The X-ray bound of the money plot before and after the INTEGRAL update of
2026-10-13, on an empty frame with the money plot's axes.

    conda run -n pandemic python plot/xray_old_vs_new.py [out_stem]

Old: the minimum over the limit files of xray_constraints/plot_xray.py with
INTEGRAL from Boyarsky et al. 2008 (0710.4922); new: the same with INTEGRAL/SPI
from Fischer et al. 2023 (2211.06200, reference profile; overall_constraint.dat).
Both are recomputed here as plot_xray.py does. Also drawn: the two INTEGRAL
limits themselves, with the SPI profile band of Fig. 10 (COM to ISO).
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
from relic_scan_theta import XRAY_DIR, M_LIM, S_LIM, COLUMNWIDTH, set_style, get_figsize

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("figures", "xray_old_vs_new")


def load(name):
    return np.loadtxt(os.path.join(XRAY_DIR, name), comments="#", skiprows=0 if name.startswith("INTEGRAL_SPI") else 2)


common = [load(f) for f in ("CBS_2008.02283.dat", "Dwarfs_1408.3531.dat", "M31_1311.0282.dat",
                            "NuSTAR_1908.09037.dat", "XMMBlankSky_2102.02207.dat")]
bmn08 = load("INTEGRAL_0710.4922.dat")
spi = load("INTEGRAL_SPI_2211.06200.dat")   # m [GeV], reference, COM, ISO


def overall(m, curves):
    """Minimum over the curves, log-linear interpolation, as plot_xray.py."""
    lm = np.log(m)
    return np.exp(np.min([np.interp(lm, np.log(c[:, 0]), np.log(c[:, 1]), left=10.0, right=10.0)
                          for c in curves], axis=0))


m = np.logspace(np.log10(M_LIM[0]), np.log10(M_LIM[1]), 4000) * 1e-6   # GeV
old = overall(m, common + [bmn08])
new = overall(m, common + [spi[:, :2]])
# the shipped file must be the "new" curve
ship = np.loadtxt(os.path.join(XRAY_DIR, "overall_constraint.dat"))
inr = (m >= ship[0, 0]) & (m <= ship[-1, 0])   # the limit files start at 2 keV
chk = np.exp(np.interp(np.log(m[inr]), np.log(ship[:, 0]), np.log(ship[:, 1])))
dev = np.abs(chk / new[inr] - 1)   # the shipped file is the same minimum on a coarser grid
print(f"new vs overall_constraint.dat: |ratio - 1| median {np.median(dev):.1e}, 99% {np.quantile(dev, 0.99):.1e}, max {dev.max():.1e}")
old[~inr], new[~inr] = np.nan, np.nan
sel = m > 41e-6
print(f"new/old above 41 keV: median {np.median(new[sel] / old[sel]):.2f}, "
      f"range {np.min(new[sel] / old[sel]):.2f} - {np.max(new[sel] / old[sel]):.2f}")

set_style()
fig = plt.figure(figsize=get_figsize(COLUMNWIDTH, wf=1.0, hf=0.9), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis="both", which="both", direction="in", width=0.5)
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
for s in ax.spines.values():
    s.set_linewidth(0.5)

k = 1e6   # GeV -> keV
ax.fill_between(k * spi[:, 0], spi[:, 3], spi[:, 2], color="C0", alpha=0.15, lw=0,
                label=r"SPI 2023, MW profile range")
ax.plot(k * bmn08[:, 0], bmn08[:, 1], color="C3", lw=0.9, ls=":", label=r"INTEGRAL, BMN08 (0710.4922)")
ax.plot(k * spi[:, 0], spi[:, 1], color="C0", lw=0.9, ls=":", label=r"INTEGRAL/SPI 2023 (2211.06200)")
ax.plot(k * m, old, color="C3", lw=1.4, label=r"combined bound, old")
ax.plot(k * m, new, color="C0", lw=1.4, ls="--", label=r"combined bound, new")

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

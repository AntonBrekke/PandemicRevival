"""
Relic roots for several m_phi against m_phi = 100 m_N, from the combined scan directories.

    python plot/mphi_compare.py [out_stem]

Reads tmp/combined_scan (m_phi = 100) and tmp/mphi_scans/mphi<r>/combined (written by
plot/combined_scan_dir.py from tmp/mphi_scans/mphi<r>). Top: sin^2 2theta(m_N) at whole decades
of g, one line style per m_phi. Bottom: the ratio to m_phi = 100 at the same (m_N, g). Prints the
median ratio per g and m_phi.
"""
import sys
import os
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "figures/mphi_compare"
key = lambda r: (round(float(r["m_N"]) * 1e6, 3), round(math.log10(float(r["y"])), 3))


def load(d):
    p = os.path.join(d, "roots.csv")
    if not os.path.exists(p):
        return {}
    return {key(r): float(r["sin2_2theta"]) for r in csv.DictReader(open(p)) if r["solver"] != "failed"}


ref = load("tmp/combined_scan")
runs = {}
for r in ("50", "10", "5.5", "1.5"):
    d = f"tmp/mphi_scans/mphi{r}/combined"
    if os.path.exists(d):
        runs[r] = load(d)
styles = {"100": "-", "50": "--", "10": "-.", "5.5": ":", "1.5": (0, (1, 3))}
decades = range(-8, 0)
cmap = plt.get_cmap("viridis")
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
for i, e in enumerate(decades):
    c = cmap(i / (len(decades) - 1))
    for name, roots in [("100", ref)] + list(runs.items()):
        pts = sorted((k[0], v) for k, v in roots.items() if abs(k[1] - e) < 1e-3)
        if not pts:
            continue
        ax.plot(*zip(*pts), ls=styles[name], color=c, lw=1.3,
                label=(rf"$g=10^{{{e}}}$" if name == "100" else None))
        if name != "100":
            rat = [(m, v / ref[(m, e)]) for m, v in pts if (m, e) in ref]
            if rat:
                axr.plot(*zip(*rat), ls=styles[name], color=c, lw=1.3)
for name in ["100"] + list(runs):
    ax.plot([], [], ls=styles[name], color="k", label=rf"$m_\phi = {name}\,m_N$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
ax.set_ylabel(r"$\sin^2 2\theta$ at $\Omega h^2 = 0.12$")
ax.legend(fontsize=7, ncol=2)
axr.set_xscale("log"); axr.set_yscale("log"); axr.minorticks_off()
axr.axhline(1, color="k", lw=0.5)
axr.set_ylabel(r"$/\ (m_\phi = 100\,m_N)$")
axr.set_xlabel(r"$m_N$ [keV]")
fig.tight_layout()
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out + ".pdf")
print("written", out + ".pdf")

ys = sorted({k[1] for roots in runs.values() for k in roots})
print(f"{'g':>9} " + " ".join(f"{'m_phi=' + n:>12}" for n in runs) + "   (median sin^2 2theta / m_phi=100; n points)")
for y in ys:
    row = []
    for n, roots in runs.items():
        rat = sorted(v / ref[k] for k, v in roots.items() if abs(k[1] - y) < 1e-6 and k in ref)
        row.append(f"{rat[len(rat) // 2]:8.3g} ({len(rat):2d})" if rat else f"{'-':>12}")
    print(f"{10 ** y:9.3g} " + " ".join(row))

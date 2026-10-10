"""
Relic roots of the species-resolved network against the reduced solver.

    python plot/network_vs_reduced.py [network_roots.csv] [out_stem]

Top: sin^2 2theta(m_N) at whole decades of y, reduced solver dashed, network
solid (open circles: the MB network stopped because the dark sector became
degenerate, or did not converge). Bottom: the ratio network/reduced. The band
3e-4 < y < 6e-3 is where the reduced solver's assumptions hold
(notes/equilibrium_check.tex).
"""
import sys
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "tmp/network_scan/network_roots.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/network_vs_reduced"

rows = list(csv.DictReader(open(src)))
by_y = defaultdict(list)
for r in rows:
    by_y[float(r["y"])].append(r)

decades = sorted(y for y in by_y if abs(math.log10(y) - round(math.log10(y))) < 1e-3)
cmap = plt.get_cmap("viridis")
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
for i, y in enumerate(decades):
    c = cmap(i / max(1, len(decades) - 1))
    pts = sorted(by_y[y], key=lambda r: float(r["m_N"]))
    m = [1e6 * float(r["m_N"]) for r in pts]
    ax.plot(m, [float(r["s2_reduced"]) for r in pts], "--", color=c, lw=1)
    ok = [(1e6 * float(r["m_N"]), float(r["s2_network"]), float(r["s2_network"]) / float(r["s2_reduced"]))
          for r in pts if r["status"] == "converged"]
    bad = [1e6 * float(r["m_N"]) for r in pts if r["status"] != "converged"]
    if ok:
        ax.plot([p[0] for p in ok], [p[1] for p in ok], "-", color=c, lw=1.5, label=f"$g = 10^{{{round(math.log10(y))}}}$")
        axr.plot([p[0] for p in ok], [p[2] for p in ok], "-", color=c, lw=1.5)
    if bad:
        ax.plot(bad, [float(r["s2_reduced"]) for r in pts if r["status"] != "converged"], "o", mfc="none", color=c, ms=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\sin^2 2\theta$ at $\Omega h^2 = 0.12$")
ax.set_title("network (solid) vs reduced solver (dashed)")
ax.legend(fontsize=7, ncol=2)
ax.minorticks_off()
axr.set_xscale("log")
axr.set_yscale("log")
axr.axhline(1, color="k", lw=0.5)
axr.set_ylabel("network / reduced")
axr.set_xlabel(r"$m_N$ [keV]")
axr.minorticks_off()
fig.tight_layout()
fig.savefig(out + ".pdf")
print("written", out + ".pdf")

# summary by y: median ratio and status counts
print(f"{'y':>9} {'n':>3} {'conv':>4} {'degen':>5} {'fail':>4} {'median ratio':>12}")
for y in sorted(by_y):
    pts = by_y[y]
    conv = [float(r["s2_network"]) / float(r["s2_reduced"]) for r in pts if r["status"] == "converged"]
    conv.sort()
    med = conv[len(conv) // 2] if conv else float("nan")
    print(f"{y:9.3g} {len(pts):3d} {len(conv):4d} {sum(r['status'] == 'degenerate' for r in pts):5d} "
          f"{sum(r['status'] == 'failed' for r in pts):4d} {med:12.3f}")

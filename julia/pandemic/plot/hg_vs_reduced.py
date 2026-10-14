"""
Relic roots of the high-g one-fluid solver against the reduced solver and the
MB network.

    python plot/hg_vs_reduced.py [hg_roots.csv] [out_stem]

Top: sin^2 2theta(m_N) at whole decades of y and y = 1.778e-3, 5.623e-3:
reduced solver dashed, network dotted (where it converged), high-g solver
solid. Bottom: the ratios high-g/reduced (solid) and network/reduced (dotted).
Prints a summary by y (median ratio, failures) and the overlap high-g/network.
"""
import sys
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "tmp/hg_scan/hg_roots.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/hg_vs_reduced"

rows = list(csv.DictReader(open(src)))
by_y = defaultdict(list)
for r in rows:
    by_y[float(r["y"])].append(r)


def f(r, k):
    try:
        return float(r[k])
    except ValueError:
        return float("nan")


shown = sorted(by_y)
cmap = plt.get_cmap("viridis")
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
for i, y in enumerate(shown):
    c = cmap(i / max(1, len(shown) - 1))
    pts = sorted(by_y[y], key=lambda r: f(r, "m_N"))
    m = [1e6 * f(r, "m_N") for r in pts]
    ax.plot(m, [f(r, "s2_reduced") for r in pts], "--", color=c, lw=0.8)
    hg = [(1e6 * f(r, "m_N"), f(r, "s2_hg"), f(r, "s2_hg") / f(r, "s2_reduced")) for r in pts if r["status"] == "converged"]
    net = [(1e6 * f(r, "m_N"), f(r, "s2_network"), f(r, "s2_network") / f(r, "s2_reduced"))
           for r in pts if r["status_network"] == "converged"]
    lab = f"$g = {y:.3g}$"
    if hg:
        ax.plot([p[0] for p in hg], [p[1] for p in hg], "-", color=c, lw=1.5, label=lab)
        axr.plot([p[0] for p in hg], [p[2] for p in hg], "-", color=c, lw=1.5)
    if net:
        ax.plot([p[0] for p in net], [p[1] for p in net], ":", color=c, lw=1.5)
        axr.plot([p[0] for p in net], [p[2] for p in net], ":", color=c, lw=1.5)
    bad = [(1e6 * f(r, "m_N"), f(r, "s2_reduced")) for r in pts if r["status"] != "converged"]
    if bad:
        ax.plot([p[0] for p in bad], [p[1] for p in bad], "x", color=c, ms=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\sin^2 2\theta$ at $\Omega h^2 = 0.12$")
ax.set_title("high-g fluid (solid), network (dotted), reduced (dashed); x: high-g failed")
ax.legend(fontsize=7, ncol=2)
ax.minorticks_off()
axr.set_xscale("log")
axr.set_yscale("log")
axr.axhline(1, color="k", lw=0.5)
axr.set_ylabel("/ reduced")
axr.set_xlabel(r"$m_N$ [keV]")
axr.minorticks_off()
fig.tight_layout()
fig.savefig(out + ".pdf")
print("written", out + ".pdf")

print(f"{'y':>9} {'n':>3} {'conv':>4} {'fail':>4} {'median hg/red':>13} {'min':>7} {'max':>7} {'median hg/net':>13} {'n_ov':>4}")
for y in sorted(by_y):
    pts = by_y[y]
    rat = sorted(f(r, "s2_hg") / f(r, "s2_reduced") for r in pts if r["status"] == "converged")
    ov = sorted(f(r, "s2_hg") / f(r, "s2_network") for r in pts
                if r["status"] == "converged" and r["status_network"] == "converged")
    med = rat[len(rat) // 2] if rat else float("nan")
    mov = ov[len(ov) // 2] if ov else float("nan")
    print(f"{y:9.4g} {len(pts):3d} {len(rat):4d} {sum(r['status'] != 'converged' for r in pts):4d} {med:13.3f} "
          f"{(rat[0] if rat else float('nan')):7.3f} {(rat[-1] if rat else float('nan')):7.3f} {mov:13.3f} {len(ov):4d}")

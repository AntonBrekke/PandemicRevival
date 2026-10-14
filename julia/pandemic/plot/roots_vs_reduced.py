"""
Relic roots of the network and the fluid solver against those of the original reduced
solver (one temperature, mu_N1 = mu_N2 = mu_A'/2, thermal start, only A' <-> N2 nu as
portal, no phi; removed from the code on 2026-10-14, its roots are kept as the s2_reduced
column of the scans).

    python plot/roots_vs_reduced.py [out_stem]

Top: sin^2 2theta(m_N) at whole decades of y: reduced dashed, network dotted, fluid solid.
Bottom: the ratios to the reduced root. Prints the median ratio per y, with the network
for y < 1e-3 and the fluid from y = 1e-3 (where both exist, also fluid/network).
"""
import sys
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "figures/roots_vs_reduced"


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float("nan")


net = {(f(r, "m_N"), f(r, "y")): r for r in csv.DictReader(open("tmp/network_scan/network_roots.csv"))}
fl = {(f(r, "m_N"), f(r, "y")): r for r in csv.DictReader(open("tmp/fluid_scan/fluid_roots.csv"))}
by_y = defaultdict(list)
for k in set(net) | set(fl):
    by_y[k[1]].append(k)

cmap = plt.get_cmap("viridis")
decades = sorted(y for y in by_y if abs(math.log10(y) - round(math.log10(y))) < 1e-3)
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
for i, y in enumerate(decades):
    c = cmap(i / max(1, len(decades) - 1))
    ks = sorted(by_y[y])
    red = [(1e6 * k[0], f(net.get(k) or fl.get(k), "s2_reduced")) for k in ks]
    ax.plot(*zip(*red), "--", color=c, lw=0.8)
    for src, col, ls in ((net, "s2_network", ":"), (fl, "s2_fluid", "-")):
        pts = [(1e6 * k[0], f(src[k], col), f(src[k], col) / f(src[k], "s2_reduced")) for k in ks
               if k in src and src[k]["status"] == "converged"]
        if pts:
            lab = f"$g = 10^{{{round(math.log10(y))}}}$" if ls == "-" or y < 1e-3 else None
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=c, lw=1.5, label=lab)
            axr.plot([p[0] for p in pts], [p[2] for p in pts], ls, color=c, lw=1.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\sin^2 2\theta$ at $\Omega h^2 = 0.12$")
ax.set_title("fluid (solid), network (dotted), reduced (dashed)")
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


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


print(f"{'y':>9} {'solver':>8} {'n':>4} {'median/red':>10} {'min':>7} {'max':>7} {'fluid/net':>9}")
for y in sorted(by_y):
    src, col, name = (fl, "s2_fluid", "fluid") if y >= 1e-3 * (1 - 1e-3) else (net, "s2_network", "network")
    rat = [f(src[k], col) / f(src[k], "s2_reduced") for k in by_y[y] if k in src and src[k]["status"] == "converged"]
    ov = [f(fl[k], "s2_fluid") / f(net[k], "s2_network") for k in by_y[y] if k in fl and k in net
          and fl[k]["status"] == "converged" and net[k]["status"] == "converged"]
    if rat:
        print(f"{y:9.4g} {name:>8} {len(rat):4d} {median(rat):10.3f} {min(rat):7.3f} {max(rat):7.3f} {median(ov):9.3f}")

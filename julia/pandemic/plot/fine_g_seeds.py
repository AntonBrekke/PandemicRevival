"""
Seeds for a finer coupling grid: the geometric midpoint between every two neighbouring
y of the existing scans (4 per decade -> 8 per decade), at every mass where both
neighbours have a root, with sin^2 2theta interpolated linearly in (ln y, ln s2)
between the neighbours' roots of the same solver.

    python plot/fine_g_seeds.py [y_switch]

Writes
- tmp/network_scan_fine/seeds.csv (y < y_switch): m_N,y,sin2_2theta,omega_h2, read by
  `src/run_network_scan.jl <tables> tmp/network_scan_fine tmp/network_scan_fine/seeds.csv`.
  Seeded from the network roots, from the reduced ones where a neighbour has no network root.
- tmp/hg_scan_fine/seeds.csv (y >= y_switch): in the format of network_roots.csv, read by
  `src/run_hg_scan.jl --points tmp/hg_scan_fine/seeds.csv --out tmp/hg_scan_fine`.
  Seeded from the high-g roots (converged or bracketed). Its s2_reduced column holds the
  seed (there is no reduced root at these y).
"""
import sys
import os
import csv
import math
from collections import defaultdict

y_switch = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3


def grid(rows, s2_col, ok):
    """{m: {y: s2}} of the rows that pass ok."""
    g = defaultdict(dict)
    for r in rows:
        if ok(r):
            g[float(r["m_N"])][float(r["y"])] = float(r[s2_col])
    return g


def midpoints(ys):
    ys = sorted(ys)
    return [(a, b, math.sqrt(a * b)) for a, b in zip(ys, ys[1:])]


def interp(a, b, y, sa, sb):
    t = math.log(y / a) / math.log(b / a)
    return math.exp(math.log(sa) + t * (math.log(sb) - math.log(sa)))


net_rows = list(csv.DictReader(open("tmp/network_scan/network_roots.csv")))
hg_rows = list(csv.DictReader(open("tmp/hg_scan/hg_roots.csv")))
net = grid(net_rows, "s2_network", lambda r: r["status"] == "converged")
red = grid(net_rows, "s2_reduced", lambda r: math.isfinite(float(r["s2_reduced"])))
hg = grid(hg_rows, "s2_hg", lambda r: r["status"] in ("converged", "bracketed"))
ys = sorted({float(r["y"]) for r in net_rows})

net_seeds, hg_seeds = [], []
for a, b, y in midpoints(ys):
    for m in sorted(red):
        if y < y_switch:
            src = net if (a in net[m] and b in net[m]) else red
            if a in src[m] and b in src[m]:
                net_seeds.append((m, y, interp(a, b, y, src[m][a], src[m][b])))
        elif a in hg[m] and b in hg[m]:
            hg_seeds.append((m, y, interp(a, b, y, hg[m][a], hg[m][b])))

os.makedirs("tmp/network_scan_fine", exist_ok=True)
os.makedirs("tmp/hg_scan_fine", exist_ok=True)
with open("tmp/network_scan_fine/seeds.csv", "w") as io:
    io.write("m_N,y,sin2_2theta,omega_h2\n")
    for m, y, s2 in net_seeds:
        io.write(f"{m:.6g},{y:.4g},{s2:.6g},NaN\n")
with open("tmp/hg_scan_fine/seeds.csv", "w") as io:
    io.write("m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds\n")
    for m, y, s2 in hg_seeds:
        io.write(f"{m:.6g},{y:.4g},{s2:.6g},NaN,NaN,NaN,none,0,0\n")
print(f"{len(net_seeds)} network seeds ({len({p[1] for p in net_seeds})} y), "
      f"{len(hg_seeds)} high-g seeds ({len({p[1] for p in hg_seeds})} y)")

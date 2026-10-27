"""
Fluid seeds for m_A' = 3 m_N: the m_A' = 2.5 fluid roots (seeds/fluid_points{,_fine}.csv, from
plot/delta_seeds.py) times the ratio of the network roots, m_A' = 3 over 2.5, at the same mass in
the overlap g = 1e-3 ... 3.2e-3 (median of ln ratio over the converged pairs; 1 where there is
none). Writes seeds/fluid_points_scaled{,_fine}.csv. From julia/pandemic:

    python tmp/mA3_scan/scale_fluid_seeds.py
"""
import csv
import math
from statistics import median

D = "tmp/mA3_scan"
key = lambda r: (round(float(r["m_N"]) * 1e6, 3), round(math.log10(float(r["y"])), 3))


def roots(path):
    return {key(r): float(r["s2_network"]) for r in csv.DictReader(open(path)) if r["status"] == "converged"}


old, new = {}, {}
for s in ("", "_fine"):
    old.update(roots(f"tmp/network_scan{s}/network_roots.csv"))
    new.update(roots(f"{D}/network_scan{s}/network_roots.csv"))
lr = {}
for k in set(old) & set(new):
    if -3.001 <= k[1] <= math.log10(3.2e-3):
        lr.setdefault(k[0], []).append(math.log(new[k] / old[k]))
ratio = {m: math.exp(median(v)) for m, v in lr.items()}
print("m_A' = 3 / 2.5 network root ratio in the overlap, per mass [keV]:")
print("  " + ", ".join(f"{m:g}: {r:.3f}" for m, r in sorted(ratio.items())))
for s in ("", "_fine"):
    rows = list(csv.DictReader(open(f"{D}/seeds/fluid_points{s}.csv")))
    with open(f"{D}/seeds/fluid_points_scaled{s}.csv", "w") as io:
        io.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            r["s2_reduced"] = f"{float(r['s2_reduced']) * ratio.get(key(r)[0], 1.0):.6g}"
            io.write(",".join(r.values()) + "\n")
    print(f"{D}/seeds/fluid_points_scaled{s}.csv: {len(rows)} points")

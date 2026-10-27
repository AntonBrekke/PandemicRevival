"""
Second pass of the fluid seeds for m_A' = 3 m_N (after scale_fluid_seeds.py): the root ratio
r = s2(m_A' = 3)/s2(m_A' = 2.5) changes with g, from r_ov(m) of the network overlap (g ~ 2e-3,
0.95 - 1.14) to ~1.3 at g = 0.1778 (the first fluid roots). Seeds: the m_A' = 2.5 fluid root times
r interpolated linearly in ln g between G_OV and G_TOP (r_top per mass where a fluid root at
G_TOP exists, else their median), r_ov below G_OV. Points already in <done> are left out.
From julia/pandemic:

    python tmp/mA3_scan/rescale_fluid_seeds_g.py <done fluid_roots(_partial).csv> [suffix]

Writes seeds/fluid_points_scaled2<suffix>.csv.
"""
import csv
import math
import statistics
import sys

D = "tmp/mA3_scan"
G_OV, G_TOP = 2e-3, 0.1778
done_path = sys.argv[1]
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
key = lambda r: (round(float(r["m_N"]) * 1e6, 3), round(math.log10(float(r["y"])), 3))

base = {key(r): float(r["s2_reduced"]) for r in csv.DictReader(open(f"{D}/seeds/fluid_points.csv"))}
base.update({key(r): float(r["s2_reduced"]) for r in csv.DictReader(open(f"{D}/seeds/fluid_points_fine.csv"))})
scaled = {}
for s in ("", "_fine"):
    for r in csv.DictReader(open(f"{D}/seeds/fluid_points_scaled{s}.csv")):
        scaled[key(r)] = float(r["s2_reduced"])
r_ov = {k[0]: scaled[k] / base[k] for k in scaled}          # per mass (the same for every g)
done = [r for r in csv.DictReader(open(done_path)) if r["status"] == "converged"]
done_keys = {key(r) for r in done}
r_top = {key(r)[0]: float(r["s2_fluid"]) / base[key(r)] for r in done
         if abs(float(r["y"]) / G_TOP - 1) < 1e-3 and key(r) in base}
r_top_med = statistics.median(r_top.values())
print(f"r at g = {G_TOP}: " + ", ".join(f"{m:g}: {v:.3f}" for m, v in sorted(r_top.items()))
      + f"; median {r_top_med:.3f}")

rows = list(csv.DictReader(open(f"{D}/seeds/fluid_points{suffix}.csv")))
n = 0
with open(f"{D}/seeds/fluid_points_scaled2{suffix}.csv", "w") as io:
    io.write(",".join(rows[0].keys()) + "\n")
    for r in rows:
        k = key(r)
        if k in done_keys:
            continue
        m, g = k[0], float(r["y"])
        lo, hi = math.log(r_ov.get(m, 1.0)), math.log(r_top.get(m, r_top_med))
        w = min(max(math.log(g / G_OV) / math.log(G_TOP / G_OV), 0.0), 1.0)
        r["s2_reduced"] = f"{base[k] * math.exp(lo + w * (hi - lo)):.6g}"
        io.write(",".join(r.values()) + "\n")
        n += 1
print(f"{D}/seeds/fluid_points_scaled2{suffix}.csv: {n} points")

"""
Roots at m_A' = 3 m_N against m_A' = 2.5 m_N, per g: status counts and the median / range of the
ratio of sin^2 2theta roots where both converged. From julia/pandemic:

    python tmp/mA3_scan/compare_roots.py [network|fluid] [suffix: '' or _fine]
"""
import csv
import statistics
import sys
from collections import Counter

solver = sys.argv[1] if len(sys.argv) > 1 else "network"
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
col = "s2_network" if solver == "network" else "s2_fluid"
a = list(csv.DictReader(open(f"tmp/{solver}_scan{suffix}/{solver}_roots.csv")))
b = list(csv.DictReader(open(f"tmp/mA3_scan/{solver}_scan{suffix}/{solver}_roots.csv")))
print("m_A' = 2.5:", dict(Counter(x["status"] for x in a)))
print("m_A' = 3:  ", dict(Counter(x["status"] for x in b)))
for x in b:
    if x["status"] == "failed":
        print(f"  failed: {float(x['m_N']) * 1e6:g} keV, g = {x['y']}, {x['n_solves']} solves")
A = {(x["m_N"], x["y"]): x for x in a}
for g in sorted({x["y"] for x in b}, key=float):
    rs = [float(x[col]) / float(A[(x["m_N"], x["y"])][col]) for x in b
          if x["y"] == g and x["status"] == "converged" and A.get((x["m_N"], x["y"]), {}).get("status") == "converged"]
    if rs:
        print(f"g = {float(g):.4g}: {len(rs)} pairs, s2(3)/s2(2.5) median {statistics.median(rs):.3f}, "
              f"range {min(rs):.3f} - {max(rs):.3f}")

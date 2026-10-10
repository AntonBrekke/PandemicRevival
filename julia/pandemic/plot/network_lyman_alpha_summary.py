"""
Lyman-alpha mass limit (lambda_fs = 0.24 Mpc) per y, network vs reduced solver.

    python plot/network_lyman_alpha_summary.py [lyman_alpha_network.csv]

For each y, log-interpolates lambda_fs(m_N) of both to the bound (the limit
used in relic_scan_theta.py) and prints the two masses.
"""
import sys
import csv
import math
from collections import defaultdict

LAMBDA_MAX = 0.24
src = sys.argv[1] if len(sys.argv) > 1 else "tmp/network_scan/lyman_alpha_network.csv"
rows = list(csv.DictReader(open(src)))
by_y = defaultdict(list)
for r in rows:
    by_y[float(r["y"])].append(r)


def crossing(pts):
    """m_N [keV] where lambda_fs falls through LAMBDA_MAX (log-log interpolation), or None."""
    pts = sorted((m, l) for m, l in pts if l > 0 and math.isfinite(l))
    for (m0, l0), (m1, l1) in zip(pts, pts[1:]):
        if l0 >= LAMBDA_MAX > l1:
            t = (math.log(LAMBDA_MAX) - math.log(l0)) / (math.log(l1) - math.log(l0))
            return math.exp(math.log(m0) + t * (math.log(m1) - math.log(m0)))
    return None


print(f"{'y':>9} {'m_min network [keV]':>20} {'m_min reduced [keV]':>20} {'ratio':>6}")
for y in sorted(by_y):
    pts = by_y[y]
    mn = crossing([(1e6 * float(r["m_N"]), float(r["lambda_fs_network_Mpc"])) for r in pts])
    mr = crossing([(1e6 * float(r["m_N"]), float(r["lambda_fs_reduced_Mpc"])) for r in pts])
    f = lambda v: f"{v:20.3g}" if v else f"{'-':>20}"
    rat = f"{mn / mr:6.3f}" if mn and mr else f"{'-':>6}"
    print(f"{y:9.3g} {f(mn)} {f(mr)} {rat}")

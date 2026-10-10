"""
sigma/m at the network roots (network f_N2 and splitting) vs with the reduced
solver's 50/50 mixture, per y: largest value over the masses.

    python plot/network_sigma_m_summary.py [lyman_alpha_network.csv]
"""
import sys
import csv
import math
from collections import defaultdict
sys.path.insert(0, "plot")
from self_interaction import sigma_over_m

M_A_OVER_M_N = 2.5
M_NU = 0.1e-9   # GeV, the splitting delta = sin^2(theta_1) m_N/m_nu of run_lyman_alpha.jl
src = sys.argv[1] if len(sys.argv) > 1 else "tmp/network_scan/lyman_alpha_network.csv"
by_y = defaultdict(list)
for r in csv.DictReader(open(src)):
    if not math.isfinite(float(r["f_N2_network"])):
        continue
    m, y, s2, f2 = float(r["m_N"]), float(r["y"]), float(r["s2_network"]), float(r["f_N2_network"])
    th1 = math.sin(math.asin(math.sqrt(s2)) / 2)
    delta = th1**2 * m / M_NU
    by_y[y].append((sigma_over_m(m, y, M_A_OVER_M_N, "mix", f2=f2, delta=delta),
                    sigma_over_m(m, y, M_A_OVER_M_N, "mix", f2=0.5, delta=delta)))
print(f"{'y':>9} {'max sigma/m network':>20} {'max with f_N2 = 1/2':>20}  [cm^2/g]")
for y in sorted(by_y):
    a = max(v[0] for v in by_y[y]); b = max(v[1] for v in by_y[y])
    print(f"{y:9.3g} {a:20.3g} {b:20.3g}")

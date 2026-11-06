"""Assumption checks along the m_A' = 50 benchmark histories: thermalisation x_c (Gamma_kd > H), kinetic
decoupling x_kd, the fluid's N1 N1 <-> N2 N2 rate against H (its mu_N1 = mu_N2), the share of Omega h^2
made before each of these, and the most degenerate N gap. From julia/pandemic:
    python tmp/mA50_scan/check_benchmarks.py [dir]"""
import glob
import os
import sys
import numpy as np
sys.path.insert(0, "plot")
from benchmark import validity

d_in = sys.argv[1] if len(sys.argv) > 1 else "tmp/mA50_scan/benchmarks"
for p in sorted(glob.glob(os.path.join(d_in, "*.csv"))):
    title = open(p).readline().lstrip("# ").strip()
    solver = title.split()[0]
    d = np.genfromtxt(p, delimiter=",", names=True, skip_header=1, deletechars="")
    if solver == "network":
        d = d[1:]
    x, om = d["x"], d["omega_h2"]
    (x0, x_dw, x_c, x_kd), start = validity(d, title, solver)

    def share_before(xx):   # fraction of the final Omega made before xx
        if xx is None or not np.isfinite(xx):
            return float("nan")
        return (np.interp(np.log(xx), np.log(x), om) - om[0]) / (om[-1] - om[0])
    conv = d["conv_H"] if "conv_H" in d.dtype.names else None
    # while Omega grows (10% -> 90% of its growth): smallest conversion rate over H
    grow = (om - om[0] > 0.1 * (om[-1] - om[0])) & (om - om[0] < 0.9 * (om[-1] - om[0]))
    cmin = np.min(conv[grow]) if conv is not None and grow.any() else float("nan")
    x_conv = x[np.flatnonzero(conv >= 1)[0]] if conv is not None and (conv >= 1).any() else None
    name = os.path.basename(p)[:-4].replace("_mphi150", "")
    print(f"{name:34s} x_c={x_c if x_c else float('nan'):.3g} (Omega made before: {share_before(x_c):.3f})  "
          f"x_kd={x_kd if x_kd else float('nan'):.3g}  "
          f"x_conv={x_conv if x_conv else float('nan'):.3g} (before: {share_before(x_conv):.3f})  "
          f"min conv/H while growing={cmin:.3g}  min gap_N={d['gap_N1'].min():.3g}  "
          f"N2 valid from {start['N2']:.3g}")

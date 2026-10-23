"""
Which constraints exclude each relic root, for two combined scan directories (e.g. without and
with an N1-N2 splitting), per coupling g: the number of roots excluded by X-rays (f_N1 sin^2 2theta
above the limit), Lyman-alpha (lambda_fs_kd > 0.24 Mpc or r_s > 0.34 Mpc), self-interactions
(sigma/m > 1 cm^2/g) and allowed by all three, and the median N2 fraction.

    python plot/constraint_census.py [dir_a] [dir_b]

Defaults: tmp/combined_scan and tmp/delta_runs/delta1e-4/combined. Same limits and
sigma/m as plot/relic_scan_theta.py (the X-ray limit per root, not interpolated in the plane).
"""
import os
import sys
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from relic_scan_theta import (load_roots, load_lyman_alpha, load_xray_limit, LAMBDA_FS_MAX_MPC, R_S_MAX_MPC,
                              SIGMA_M_MAX, M_A_OVER_M_N)
from self_interaction import sigma_over_m

dirs = sys.argv[1:3] if len(sys.argv) > 2 else ["tmp/combined_scan", "tmp/delta_runs/delta1e-4/combined"]
m_x, s_x = load_xray_limit()


def census(d):
    roots = load_roots(d)
    lam, r_s, mix = load_lyman_alpha(d)
    out = collections.defaultdict(lambda: collections.Counter())
    f2s = collections.defaultdict(list)
    for k, s2 in roots.items():
        if k not in mix:
            continue
        m, y = k
        f2, dm = mix[k]
        f2s[y].append(f2)
        c = out[y]
        c["n"] += 1
        xr = (1 - f2) * s2 > np.interp(np.log(m), np.log(m_x), s_x)
        ly = lam.get(k, 0) > LAMBDA_FS_MAX_MPC or r_s.get(k, 0) > R_S_MAX_MPC
        si = sigma_over_m(1e-6 * m, y, M_A_OVER_M_N, f2, dm) > SIGMA_M_MAX
        c["xray"] += xr
        c["lya"] += ly
        c["sigma"] += si
        c["allowed"] += not (xr or ly or si)
    return out, f2s


res = [census(d) for d in dirs]
print("g        | " + " | ".join(f"{os.path.basename(os.path.dirname(d + '/'))[:22]:>22}: n  xray lya  si allowed f_N2" for d in dirs))
for y in sorted(set(res[0][0]) | set(res[1][0])):
    if abs(np.log10(y) - round(np.log10(y))) > 1e-3 and abs(np.log10(y) - round(np.log10(y)) - 0.5) > 1e-3:
        continue   # half decades only
    row = []
    for out, f2s in res:
        c = out[y]
        row.append(f"{c['n']:3d} {c['xray']:4d} {c['lya']:3d} {c['sigma']:3d} {c['allowed']:4d}    {np.median(f2s[y]) if f2s[y] else float('nan'):.2g}")
    print(f"{y:8.3g} | " + " | ".join(f"{r:>48}" for r in row))

"""
Benchmark points for future X-ray searches, in the spirit of BP1/BP2 of Bringmann et al.
(2206.10630): relic roots that every current bound allows (X-rays, both Lyman-alpha lengths,
sigma/m, as in plot/relic_scan_theta.py) but that a projected X-ray sensitivity (eROSITA, Athena,
eXTP: XRAY_PROJECTIONS there) would reach. Only N1 decays to nu gamma, so a root is reached if
f_N1 sin^2 2theta exceeds the projection (which, like the limits, assumes all the DM decays).

    python plot/xray_benchmarks.py [scan_dir] [out.csv] [--mA 2.5] [--n-per-bin 1]

Writes every candidate to <out>_candidates.csv and the chosen points to <out> (default
<scan_dir>/xray_benchmarks.csv): per bin of g (G_BINS), the root deepest inside the window,
i.e. with the largest min(L/(f_N1 s2), f_N1 s2/P), L the current limit and P the best projection
at that mass. Columns: name, m_keV, g, sin2_2theta, solver, f_N2, lambda_fs_kd_Mpc, r_s_Mpc,
sigma_m_cm2_g, now_margin = L/(f_N1 s2), reach_margin = f_N1 s2/P, reached_by.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relic_scan_theta import (LAMBDA_FS_MAX_MPC, R_S_MAX_MPC, SIGMA_M_MAX, M_A_OVER_M_N, key, load_lyman_alpha,
                              load_roots, load_xray_limit, load_xray_projections, read_csv)
from self_interaction import sigma_over_m

G_BINS = [(1e-6, 1e-4), (1e-4, 3e-4), (3e-4, 1e-3), (1e-3, 1.0)]


def interp_log(m, ms, vs):
    """Limit at m [keV] (log-log interpolation; inf outside its range)."""
    return np.exp(np.interp(np.log(m), np.log(ms), np.log(vs), left=np.inf, right=np.inf))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan_dir", nargs="?", default="tmp/combined_scan")
    p.add_argument("out", nargs="?", default=None)
    p.add_argument("--mA", type=float, default=M_A_OVER_M_N)
    p.add_argument("--n-per-bin", type=int, default=1)
    a = p.parse_args()
    out = a.out or os.path.join(a.scan_dir, "xray_benchmarks.csv")

    roots = load_roots(a.scan_dir)
    solver = {key(r): str(r["solver"]) for r in read_csv(os.path.join(a.scan_dir, "roots.csv"))}
    lam, r_s, mix = load_lyman_alpha(a.scan_dir)
    m_L, L = load_xray_limit()
    proj = load_xray_projections()

    cands = []
    for k, s2 in roots.items():
        if k not in mix or k not in lam or k not in r_s:
            continue
        m, g = k
        f2, dm = mix[k]
        f1 = 1.0 - f2
        sig = sigma_over_m(m * 1e-6, g, a.mA, f2, dm)
        lim = interp_log(m, m_L, L)
        if not (f1 * s2 < lim and lam[k] < LAMBDA_FS_MAX_MPC and r_s[k] < R_S_MAX_MPC and sig < SIGMA_M_MAX):
            continue
        reach = {lab: interp_log(m, *pr) for lab, pr in proj.items()}
        by = [lab for lab, P in reach.items() if f1 * s2 > P]
        if not by:
            continue
        P = min(reach.values())
        cands.append(dict(m_keV=m, g=g, sin2_2theta=s2, solver=solver.get(k, "?"), f_N2=f2,
                          lambda_fs_kd_Mpc=lam[k], r_s_Mpc=r_s[k], sigma_m_cm2_g=sig,
                          now_margin=lim / (f1 * s2), reach_margin=f1 * s2 / P, reached_by="+".join(by)))
    cols = ["name", "m_keV", "g", "sin2_2theta", "solver", "f_N2", "lambda_fs_kd_Mpc", "r_s_Mpc", "sigma_m_cm2_g",
            "now_margin", "reach_margin", "reached_by"]

    def write(path, rows):
        with open(path, "w") as io:
            io.write(",".join(cols) + "\n")
            for r in rows:
                io.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    cands.sort(key=lambda r: (r["g"], r["m_keV"]))
    for r in cands:
        r["name"] = ""
    write(out[:-4] + "_candidates.csv", cands)
    chosen = []
    for lo, hi in G_BINS:
        inbin = [r for r in cands if lo <= r["g"] < hi]
        inbin.sort(key=lambda r: -min(r["now_margin"], r["reach_margin"]))
        chosen += inbin[:a.n_per_bin]
    chosen.sort(key=lambda r: (r["g"], r["m_keV"]))
    for i, r in enumerate(chosen):
        r["name"] = f"BP{i + 1}"
    write(out, chosen)
    print(f"{len(cands)} roots allowed now and within projected X-ray reach -> {out[:-4]}_candidates.csv")
    for r in chosen:
        print(f"  {r['name']}: m = {r['m_keV']:g} keV, g = {r['g']:.4g}, sin^2 2theta = {r['sin2_2theta']:.4g} "
              f"({r['solver']}), f_N2 = {r['f_N2']:.3g}, lambda_fs = {r['lambda_fs_kd_Mpc']:.3g}, "
              f"r_s = {r['r_s_Mpc']:.3g} Mpc, sigma/m = {r['sigma_m_cm2_g']:.2g}; below the limit x{r['now_margin']:.2f}, "
              f"above the reach x{r['reach_margin']:.2f} ({r['reached_by']})")
    print("written", out)


if __name__ == "__main__":
    main()

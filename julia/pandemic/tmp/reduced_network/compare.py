"""Reduced-network roots and Lyman-alpha lengths against the stored ones.
From julia/pandemic: python tmp/reduced_network/compare.py"""
import csv
import glob
import os

import numpy as np

D = "tmp/reduced_network"


def read(path):
    rows = list(csv.reader(open(path)))
    return rows[0], rows[1:]


def decade_bins(y):
    return np.floor(np.log10(y) + 1e-9).astype(int)


for path in sorted(glob.glob(f"{D}/*/network_roots.csv")):
    name = path.split("/")[-2]
    h, rows = read(path)
    c = {k: i for i, k in enumerate(h)}
    m = np.array([float(r[c["m_N"]]) for r in rows])
    y = np.array([float(r[c["y"]]) for r in rows])
    seed = np.array([float(r[c["s2_reduced"]]) for r in rows])
    s2 = np.array([float(r[c["s2_network"]]) for r in rows])
    st = np.array([r[c["status"]] for r in rows])
    sec = np.array([float(r[c["seconds"]]) for r in rows])
    ok = np.isin(st, ("converged", "bracketed"))
    d = np.abs(np.log(s2 / seed))
    print(f"\n== {name}: {len(rows)} points, statuses", {k: int((st == k).sum()) for k in np.unique(st)},
          f"cpu {sec.sum() / 60:.1f} min")
    if not ok.any():
        continue
    print(f"   |d ln s2|: median {np.median(d[ok]):.2e}, 95% {np.quantile(d[ok], .95):.2e}, max {d[ok].max():.2e}")
    print("   per decade of g: g, n ok/all, median, max |d ln s2|, status of the others")
    for b in np.unique(decade_bins(y)):
        s = decade_bins(y) == b
        oks = s & ok
        rest = {k: int((st[s] == k).sum()) for k in np.unique(st[s]) if k not in ("converged", "bracketed")}
        print(f"   1e{b:+d}  {oks.sum():4d}/{s.sum():4d}  " +
              (f"{np.median(d[oks]):.2e} {d[oks].max():.2e}" if oks.any() else "-") + f"  {rest if rest else ''}")
    worst = np.argsort(-np.where(ok, d, -1))[:5]
    print("   worst:", [f"m={m[i] * 1e6:.3g}keV g={y[i]:.3g} d={d[i]:.2e} {st[i]}" for i in worst])

ly = f"{D}/base_network/lyman_alpha.csv"
if os.path.exists(ly):
    h, rows = read("tmp/combined_scan/lyman_alpha.csv")
    ref = {(r[0], r[1]): r for r in rows}
    hr, rowsr = read(ly)
    print("\n== Lyman-alpha at the reduced roots vs the stored ones (relative differences)")
    print("   reduced columns:", hr)
    for k in ("lambda_fs_Mpc", "lambda_fs_kd_Mpc", "r_s_Mpc", "f_N2"):
        vals, where = [], []
        for r in rowsr:
            key = (r[0], r[1])
            if key not in ref:
                continue
            a, b = float(r[hr.index(k)]), float(ref[key][h.index(k)])
            if np.isfinite(a) and np.isfinite(b) and b != 0:
                vals.append(abs(a / b - 1))
                where.append(key)
        vals = np.array(vals)
        i = int(np.argmax(vals))
        print(f"   {k}: n={len(vals)} median {np.median(vals):.2e}, 95% {np.quantile(vals, .95):.2e}, "
              f"max {vals[i]:.2e} at m,g={where[i]}")

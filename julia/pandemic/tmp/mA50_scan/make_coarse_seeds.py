"""Seeds of the coarse m_A' = 50 m_N, m_phi = 150 m_N scan. From julia/pandemic:
    python tmp/mA50_scan/make_coarse_seeds.py
Masses: every third of the base grid (extended past 300 keV in steps of 10^0.1) plus 10.57 keV, up to
754 keV (the X-ray limit closes the region at ~670 keV for g = 1.33, pilot roots); couplings in
half-decades from 1e-8 to 1, and 1.333 (the perturbative limit, check_perturbativity.jl).
Points with g more than one half-decade above g_SI(m), where sigma/m > 1 cm^2/g for f_N2 = 1/2 at
any mixing (it does not depend on it), are left out: excluded, and only the boundary is drawn.
Network (g < 1e-3) seeds: the m_A' = 2.5 roots (tmp/combined_scan); fluid seeds: the pilot fluid
roots, interpolated in (ln m, ln g), linearly extrapolated in ln m below 10.57 keV.
Writes seeds/network_seeds.csv (m_N,y,sin2_2theta,omega_h2), seeds/fluid_seeds.csv (network_roots
format, read by run_fluid_scan.jl --points) and seeds/pruned.csv.
"""
import csv
import math
import os
import sys
import numpy as np
sys.path.insert(0, "plot")
from self_interaction import sigma_over_m

OUT = "tmp/mA50_scan/seeds"
os.makedirs(OUT, exist_ok=True)
MASSES_KEV = [0.6, 1.312, 2.869, 6.274, 10.57, 13.72, 30.0, 59.86, 119.4, 238.3, 475.5, 598.6, 753.6]
G_NET = [10 ** (e / 2) for e in range(-16, -6)]                       # 1e-8 ... 3.162e-4
G_FLU = [1e-3, 3.162e-3, 1e-2, 3.162e-2, 0.1, 0.3162, 1.0, 1.333]
R_A = 50.0


def g_si(m_kev):
    """g at which sigma/m = 1 cm^2/g (f_N2 = 1/2, no splitting); sigma/m ∝ g^4."""
    return sigma_over_m(m_kev * 1e-6, 1.0, R_A, 0.5, 0.0) ** -0.25


rows = list(csv.DictReader(open("tmp/combined_scan/roots.csv")))
old = {(float(r["m_N"]), float(r["y"])): float(r["sin2_2theta"]) for r in rows if r["sin2_2theta"] != "NaN"}


def old_seed(m, g):
    k = min(old, key=lambda k: abs(math.log(k[0] / m)) + abs(math.log(k[1] / g)))
    return old[k]


pil = {}
for r in csv.DictReader(open("tmp/mA50_scan/pilot/fluid/fluid_roots.csv")):
    pil[(float(r["m_N"]), float(r["y"]))] = float(r["s2_fluid"])
pm = sorted({k[0] for k in pil})
pg = sorted({k[1] for k in pil})


def interp1(xs, ys, x):
    """Linear in logs, linear extrapolation beyond the ends."""
    lx, ly = np.log(xs), np.log(ys)
    i = int(np.clip(np.searchsorted(lx, np.log(x)) - 1, 0, len(lx) - 2))
    t = (np.log(x) - lx[i]) / (lx[i + 1] - lx[i])
    return float(np.exp(ly[i] + t * (ly[i + 1] - ly[i])))


def fluid_seed(m, g):
    at_g = [interp1(pm, [pil[(mm, gg)] for mm in pm], m) for gg in pg]
    return interp1(pg, at_g, g)


net, flu, pruned = [], [], []
step = 10 ** 0.5
for mk in MASSES_KEV:
    m = mk * 1e-6
    gmax = g_si(mk) * step * 1.01
    for g in G_NET + G_FLU:
        if g > gmax:
            pruned.append((m, g, g_si(mk)))
            continue
        (net if g < 1e-3 else flu).append((m, g))
with open(f"{OUT}/network_seeds.csv", "w") as f:
    f.write("m_N,y,sin2_2theta,omega_h2\n")
    for m, g in net:
        f.write(f"{m:.6g},{g:.6g},{old_seed(m, g):.6g},0.12\n")
with open(f"{OUT}/fluid_seeds.csv", "w") as f:
    f.write("m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds\n")
    for m, g in flu:
        s = fluid_seed(m, g)
        f.write(f"{m:.6g},{g:.6g},{s:.6g},0.12,{s:.6g},0.12,seed,0,0\n")
with open(f"{OUT}/pruned.csv", "w") as f:
    f.write("m_N,y,g_SI\n")
    f.writelines(f"{m:.6g},{g:.6g},{gs:.4g}\n" for m, g, gs in pruned)
print(f"{len(net)} network, {len(flu)} fluid seeds; {len(pruned)} points pruned (sigma/m)")
for mk in MASSES_KEV:
    print(f"  {mk:6.4g} keV: g_SI = {g_si(mk):.3g}")

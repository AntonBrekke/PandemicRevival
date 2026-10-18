"""
Seeds for coarse scans at other m_phi: the m_phi = 100 m_N roots at a subset of the grid.

    python plot/mphi_seeds.py out_dir

Masses: every third of the scan's 26 plus 10.57 keV; couplings: half decades (1e-8, 3.16e-8, ...,
0.1). Writes <out_dir>/network_seeds.csv (m_N,y,sin2_2theta,omega_h2; network roots, g < 1e-3,
for src/run_network_scan.jl), <out_dir>/network_seeds_all.csv (the same up to g = 0.1, with the
fluid roots as seeds above 1e-3: for the overlap check and where the fluid does not apply) and
<out_dir>/fluid_seeds.csv (network_roots.csv format with the
fluid root as s2_reduced, g >= 1e-3, for src/run_fluid_scan.jl --points).
"""
import sys
import os
import csv
import math

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
net = [r for r in csv.DictReader(open("tmp/network_scan/network_roots.csv")) if r["status"] == "converged"]
fl = [r for r in csv.DictReader(open("tmp/fluid_scan/fluid_roots.csv")) if r["status"] == "converged"]
masses = sorted({float(r["m_N"]) for r in net})
keep_m = set(masses[::3]) | {m for m in masses if abs(m * 1e6 - 10.57) < 0.01}
half_decade = lambda y: abs(2 * math.log10(float(y)) - round(2 * math.log10(float(y)))) < 1e-2
sel = lambda r: float(r["m_N"]) in keep_m and half_decade(r["y"])
with open(os.path.join(out, "network_seeds.csv"), "w") as io:
    io.write("m_N,y,sin2_2theta,omega_h2\n")
    n_net = 0
    for r in net:
        if sel(r) and float(r["y"]) < 0.999e-3:
            io.write(f"{r['m_N']},{r['y']},{r['s2_network']},NaN\n"); n_net += 1
with open(os.path.join(out, "fluid_seeds.csv"), "w") as io:
    io.write("m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds\n")
    n_fl = 0
    for r in fl:
        if sel(r):
            io.write(f"{r['m_N']},{r['y']},{r['s2_fluid']},NaN,NaN,NaN,none,0,0\n"); n_fl += 1
with open(os.path.join(out, "network_seeds_all.csv"), "w") as io:
    io.write("m_N,y,sin2_2theta,omega_h2\n")
    for r in net:
        if sel(r) and float(r["y"]) < 0.999e-3:
            io.write(f"{r['m_N']},{r['y']},{r['s2_network']},NaN\n")
    for r in fl:
        if sel(r):
            io.write(f"{r['m_N']},{r['y']},{r['s2_fluid']},NaN\n")
print(f"{len(keep_m)} masses: {sorted(round(m * 1e6, 3) for m in keep_m)}; {n_net} network and {n_fl} fluid seeds")

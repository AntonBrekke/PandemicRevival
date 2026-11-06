"""Seeds of the m_A' = 50 pilot roots, from the stored m_A' = 2.5 roots (tmp/combined_scan).
Points without a stored root (m > 300 keV, g > 0.178) start from the nearest stored one.
From julia/pandemic: python tmp/mA50_scan/make_pilot_seeds.py"""
import csv
import os

D = "tmp/mA50_scan/pilot"
os.makedirs(D, exist_ok=True)
rows = list(csv.DictReader(open("tmp/combined_scan/roots.csv")))
roots = {(float(r["m_N"]), float(r["y"])): float(r["sin2_2theta"]) for r in rows if r["sin2_2theta"] != "NaN"}


def nearest(m, g):
    k = min(roots, key=lambda k: (abs(__import__("math").log(k[0] / m)) + abs(__import__("math").log(k[1] / g))))
    return roots[k]


NET = [(m, g) for m in (1.011e-6, 1.057e-5, 3e-4) for g in (1e-7, 1e-5, 1e-4, 7.499e-4, 1e-3)]
FLU = [(m, g) for m in (1.057e-5, 3e-4, 6e-4, 1e-3) for g in (1e-3, 1e-2, 0.1, 0.5, 1.0, 1.333)]
with open(f"{D}/network_seeds.csv", "w") as f:
    f.write("m_N,y,sin2_2theta,omega_h2\n")
    for m, g in NET:
        f.write(f"{m:.6g},{g:.6g},{nearest(m, g):.6g},0.12\n")
with open(f"{D}/fluid_seeds.csv", "w") as f:
    # the format of network_roots.csv, which run_fluid_scan.jl --points reads
    f.write("m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds\n")
    for m, g in FLU:
        s = nearest(m, g)
        f.write(f"{m:.6g},{g:.6g},{s:.6g},0.12,{s:.6g},0.12,seed,0,0\n")
print(len(NET), "network,", len(FLU), "fluid seeds in", D)

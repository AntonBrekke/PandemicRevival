"""
Write the network roots as a scan directory that plot/relic_scan_theta.py reads
(m_N_<keV>keV.csv per mass, lyman_alpha.csv with lambda_fs_Mpc, f_N2, delta_m).

    python plot/network_scan_dir.py [network_scan_dir] [out_dir]

There is no kinetic-decoupling r_s for the network yet: draw with
LYA_VARIANT=nokd (lambda_fs without kinetic decoupling, as computed).
"""
import sys
import os
import csv
import math

src = sys.argv[1] if len(sys.argv) > 1 else "tmp/network_scan"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "money")
os.makedirs(out, exist_ok=True)
M_NU = 0.1e-9   # GeV, as in src/run_lyman_alpha.jl

roots = list(csv.DictReader(open(os.path.join(src, "network_roots.csv"))))
lya = {(r["m_N"], r["y"]): r for r in csv.DictReader(open(os.path.join(src, "lyman_alpha_network.csv")))}
# file names as in the merged reduced scan
names = {}
for f in os.listdir("tmp/relic_scan_theta_merged"):
    if f.startswith("m_N_") and f.endswith("keV.csv"):
        names[round(float(f[4:-7]), 3)] = f
by_m = {}
for r in roots:
    by_m.setdefault(r["m_N"], []).append(r)
for m, rows in by_m.items():
    fname = names[round(float(m) * 1e6, 3)]
    with open(os.path.join(out, fname), "w") as io:
        io.write("m_N,y,sin2_2theta,omega_h2,converged,plateau_ok,retcode,branch,slope,n_failed\n")
        for r in sorted(rows, key=lambda r: float(r["y"])):
            ok = r["status"] == "converged"
            io.write(f"{m},{r['y']},{r['s2_network'] if ok else 'NaN'},{r['omega_network'] if ok else 'NaN'},"
                     f"{'true' if ok else 'false'},{'true' if ok else 'false'},{r['status']},0,{1 if ok else 0},0\n")
with open(os.path.join(out, "lyman_alpha.csv"), "w") as io:
    io.write("m_N,y,sin2_2theta,lambda_fs_Mpc,f_N2,delta_m\n")
    for (m, y), r in sorted(lya.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]))):
        s2 = float(r["s2_network"])
        th1 = math.sin(math.asin(math.sqrt(s2)) / 2)
        io.write(f"{m},{y},{s2},{r['lambda_fs_network_Mpc']},{r['f_N2_network']},{th1**2 * float(m) / M_NU}\n")
print("written", out)

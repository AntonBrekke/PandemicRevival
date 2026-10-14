"""
Write one scan directory for plot/relic_scan_theta.py in which every relic root
comes from a solver whose assumptions hold there: the MB network
(tmp/network_scan) for y < y_switch, the high-g one-fluid solver (tmp/hg_scan)
for y >= y_switch (notes/model_limitations.tex, Sec. "Large g: results"), on the
base grid and, where present, the finer one of plot/fine_g_seeds.py
(tmp/network_scan_fine, tmp/hg_scan_fine).

    python plot/combined_scan_dir.py [out_dir] [y_switch]

Output (default tmp/combined_scan): m_N_<keV>keV.csv per mass (file names as in
the merged reduced scan; retcode column = the solver: network / hg / failed) and
lyman_alpha.csv with lambda_fs_Mpc (no kinetic decoupling), lambda_fs_kd_Mpc and
r_s_Mpc (kinetic decoupling: src/run_hg_lyman_alpha.jl and src/run_network_lyman_alpha.jl,
which for the network also counts the free streaming before the sector couples), f_N2,
delta_m. Draw with the default LYA_VARIANT (fixed); LYA_VARIANT=nokd for lambda_fs_Mpc.
"""
import sys
import os
import csv
import math

out = sys.argv[1] if len(sys.argv) > 1 else "tmp/combined_scan"
y_switch = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
os.makedirs(out, exist_ok=True)
M_NU = 0.1e-9   # GeV, as in src/run_lyman_alpha.jl

key = lambda m, y: (round(float(m) * 1e6, 3), round(math.log10(float(y)), 3))


def load(*paths):
    """{(m, y): row} from all existing CSVs of `paths` (the base grid and plot/fine_g_seeds.py's finer one)."""
    return {key(r["m_N"], r["y"]): r for p in paths if os.path.exists(p) for r in csv.DictReader(open(p))}


net = load("tmp/network_scan/network_roots.csv", "tmp/network_scan_fine/network_roots.csv")
net_lya = load("tmp/network_scan/lyman_alpha_network.csv", "tmp/network_scan_fine/lyman_alpha_network.csv")
hg = load("tmp/hg_scan/hg_roots.csv", "tmp/hg_scan_fine/hg_roots.csv")
hg_lya = load("tmp/hg_scan/lyman_alpha_hg.csv", "tmp/hg_scan_fine/lyman_alpha_hg.csv")
# fine-grid points above y_switch have no network row: give them a placeholder one
for k, h in hg.items():
    net.setdefault(k, {"m_N": h["m_N"], "y": h["y"], "status": "none", "s2_network": "NaN", "omega_network": "NaN"})

def net_lya_row(m, y, s2, l):
    """A lyman_alpha.csv row from run_network_lyman_alpha.jl's output (kd columns NaN if absent)."""
    return (m, y, s2, l["lambda_fs_network_Mpc"], l.get("lambda_fs_kd_Mpc", "NaN"), l.get("r_s_Mpc", "NaN"),
            l["f_N2_network"])


names = {}
for f in os.listdir("tmp/relic_scan_theta_merged"):
    if f.startswith("m_N_") and f.endswith("keV.csv"):
        names[round(float(f[4:-7]), 3)] = f

by_m = {}
lya_rows = []
for k, r in net.items():
    m, y = r["m_N"], r["y"]
    if float(y) >= y_switch * (1 - 1e-3):
        h = hg.get(k)
        ok = h is not None and h["status"] == "converged"
        s2, om, solver = (h["s2_hg"], h["omega_hg"], "hg") if ok else ("NaN", "NaN", "failed")
        if not ok and r["status"] == "converged":
            # high-g search unsettled, but the network (valid in the overlap) has a root
            ok = True
            s2, om, solver = r["s2_network"], r["omega_network"], "network"
            if k in net_lya:
                l = net_lya[k]
                lya_rows.append(net_lya_row(m, y, s2, l))
        elif ok and k in hg_lya:
            l = hg_lya[k]
            lya_rows.append((m, y, s2, l["lambda_fs_Mpc"], l["lambda_fs_kd_Mpc"], l["r_s_Mpc"], "0.5"))
    else:
        ok = r["status"] == "converged"
        s2, om, solver = (r["s2_network"], r["omega_network"], "network") if ok else ("NaN", "NaN", "failed")
        if ok and k in net_lya:
            l = net_lya[k]
            lya_rows.append(net_lya_row(m, y, s2, l))
    by_m.setdefault(m, []).append((y, s2, om, ok, solver))

for m, rows in by_m.items():
    with open(os.path.join(out, names[round(float(m) * 1e6, 3)]), "w") as io:
        io.write("m_N,y,sin2_2theta,omega_h2,converged,plateau_ok,retcode,branch,slope,n_failed\n")
        for y, s2, om, ok, solver in sorted(rows, key=lambda t: float(t[0])):
            t = "true" if ok else "false"
            io.write(f"{m},{y},{s2},{om},{t},{t},{solver},0,{1 if ok else 0},0\n")
with open(os.path.join(out, "lyman_alpha.csv"), "w") as io:
    io.write("m_N,y,sin2_2theta,lambda_fs_Mpc,lambda_fs_kd_Mpc,r_s_Mpc,f_N2,delta_m\n")
    for m, y, s2, lfs, lkd, rs, f2 in sorted(lya_rows, key=lambda t: (float(t[0]), float(t[1]))):
        th1 = math.sin(math.asin(math.sqrt(float(s2))) / 2)
        io.write(f"{m},{y},{s2},{lfs},{lkd},{rs},{f2},{th1**2 * float(m) / M_NU}\n")
n_hg = sum(1 for rows in by_m.values() for t in rows if t[4] == "hg")
n_net = sum(1 for rows in by_m.values() for t in rows if t[4] == "network")
n_fail = sum(1 for rows in by_m.values() for t in rows if t[4] == "failed")
print(f"written {out}: {n_net} network roots, {n_hg} high-g roots, {n_fail} failed; {len(lya_rows)} with lambda_fs")

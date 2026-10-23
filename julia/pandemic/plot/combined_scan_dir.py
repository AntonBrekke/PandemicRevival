"""
Merge the relic roots of the two solvers into one scan directory for
plot/relic_scan_theta.py: the MB network (tmp/network_scan) for y < y_switch and the
one-fluid solver (tmp/fluid_scan) for y >= y_switch, on the base grid and, where present,
the finer one of plot/fine_g_seeds.py (tmp/*_scan_fine). Where the fluid root search did
not settle but the network (valid in the overlap) has a root, the network root is used.

    python plot/combined_scan_dir.py [out_dir] [y_switch] [scan_root] [delta]

scan_root (default tmp) holds the <solver>_scan{,_fine} directories, e.g. tmp/mphi_scans/mphi10.

Output (default tmp/combined_scan):
- roots.csv: m_N,y,sin2_2theta,omega_h2,solver (network / fluid / failed);
- lyman_alpha.csv: m_N,y,sin2_2theta,lambda_fs_Mpc,lambda_fs_kd_Mpc,r_s_Mpc,f_N2,delta_m,
  from the solvers' lyman_alpha.csv (src/run_*_lyman_alpha.jl), with the N1-N2 mass
  splitting delta_m = sin^2(theta_1) m_N/m_nu, or the constant `delta` of tables with an N1-N2
  splitting (rate_params; src/run_build_rate_tables.jl --delta).
"""
import sys
import os
import csv
import math

out = sys.argv[1] if len(sys.argv) > 1 else "tmp/combined_scan"
y_switch = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
root = sys.argv[3] if len(sys.argv) > 3 else "tmp"
delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
os.makedirs(out, exist_ok=True)
M_NU = 0.1e-9   # GeV, m_nu of rate_params (src/thermal_rates.jl)

key = lambda m, y: (round(float(m) * 1e6, 3), round(math.log10(float(y)), 3))


def load(solver, name):
    """{(m, y): row} from <root>/<solver>_scan/<name> and <root>/<solver>_scan_fine/<name>."""
    paths = [f"{root}/{solver}_scan/{name}", f"{root}/{solver}_scan_fine/{name}"]
    return {key(r["m_N"], r["y"]): r for p in paths if os.path.exists(p) for r in csv.DictReader(open(p))}


net, net_lya = load("network", "network_roots.csv"), load("network", "lyman_alpha.csv")
fl, fl_lya = load("fluid", "fluid_roots.csv"), load("fluid", "lyman_alpha.csv")

roots, lya = [], []
for k in sorted(set(net) | set(fl)):
    n, f = net.get(k), fl.get(k)
    m, y = (n or f)["m_N"], (n or f)["y"]
    n_ok = n is not None and n["status"] == "converged"
    f_ok = f is not None and f["status"] == "converged"
    if float(y) >= y_switch * (1 - 1e-3) and f_ok:
        solver, s2, om, l = "fluid", f["s2_fluid"], f["omega_fluid"], fl_lya.get(k)
    elif n_ok and (float(y) < y_switch * (1 - 1e-3) or not f_ok):
        solver, s2, om, l = "network", n["s2_network"], n["omega_network"], net_lya.get(k)
    else:
        solver, s2, om, l = "failed", "NaN", "NaN", None
    roots.append((m, y, s2, om, solver))
    if l is not None:
        th1 = math.sin(math.asin(math.sqrt(float(s2))) / 2)
        lya.append((m, y, s2, l["lambda_fs_Mpc"], l["lambda_fs_kd_Mpc"], l["r_s_Mpc"], l["f_N2"],
                    f"{delta if delta > 0 else th1**2 * float(m) / M_NU:.6g}"))

with open(os.path.join(out, "roots.csv"), "w") as io:
    io.write("m_N,y,sin2_2theta,omega_h2,solver\n")
    for r in roots:
        io.write(",".join(r) + "\n")
with open(os.path.join(out, "lyman_alpha.csv"), "w") as io:
    io.write("m_N,y,sin2_2theta,lambda_fs_Mpc,lambda_fs_kd_Mpc,r_s_Mpc,f_N2,delta_m\n")
    for r in lya:
        io.write(",".join(r) + "\n")
count = {s: sum(r[4] == s for r in roots) for s in ("network", "fluid", "failed")}
print(f"written {out}: {count['network']} network roots, {count['fluid']} fluid roots, {count['failed']} failed; "
      f"{len(lya)} with Lyman-alpha lengths")

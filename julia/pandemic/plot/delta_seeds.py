"""
Seeds for scans with an N1-N2 splitting (tables built with --delta): the roots of the
equal-mass scans, which the splitting moves only a little.

    python plot/delta_seeds.py [out_dir]

From tmp/{network,fluid}_scan{,_fine}/ (delta = 0) to <out_dir> (default tmp/delta_runs/seeds):
- network_seeds{,_fine}.csv for src/run_network_scan.jl (m_N,y,sin2_2theta,omega_h2): the
  network root where it converged, else the scan's own seed (s2_reduced), so that the point set
  is the same;
- fluid_points{,_fine}.csv for src/run_fluid_scan.jl --points (the network_roots.csv columns,
  with s2_reduced replaced by the fluid root where it converged).
"""
import csv
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "tmp/delta_runs/seeds"
os.makedirs(out, exist_ok=True)
for suffix in ("", "_fine"):
    net = list(csv.DictReader(open(f"tmp/network_scan{suffix}/network_roots.csv")))
    with open(f"{out}/network_seeds{suffix}.csv", "w") as io:
        io.write("m_N,y,sin2_2theta,omega_h2\n")
        for r in net:
            ok = r["status"] == "converged"
            io.write(",".join((r["m_N"], r["y"], r["s2_network"] if ok else r["s2_reduced"],
                               r["omega_network"] if ok else r["omega_reduced"])) + "\n")
    fl = list(csv.DictReader(open(f"tmp/fluid_scan{suffix}/fluid_roots.csv")))
    with open(f"{out}/fluid_points{suffix}.csv", "w") as io:
        io.write("m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status\n")
        for r in fl:
            ok = r["status"] == "converged"
            io.write(",".join((r["m_N"], r["y"], r["s2_fluid"] if ok else r["s2_reduced"],
                               r["omega_fluid"] if ok else r["omega_reduced"], r["s2_network"], "NaN",
                               r["status_network"])) + "\n")
    print(f"{out}: {len(net)} network seeds, {len(fl)} fluid points{suffix and ' (fine)'}")

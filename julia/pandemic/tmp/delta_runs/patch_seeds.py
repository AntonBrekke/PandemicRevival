"""
Repair of the delta = 1e-4 base network scan, whose first pass seeded every solve with
dz_seed = 1e-9 (fails at small g; solve_network now falls back to it only after a take-off).

    python tmp/delta_runs/patch_seeds.py write    -> tmp/delta_runs/seeds/network_seeds_patch.csv
    python tmp/delta_runs/patch_seeds.py merge    -> rows of .../network_scan_patch replace those of .../network_scan

Re-solved: points converged at delta = 0 but not in the first pass, and all of 1e-3 <= g <= 3.2e-3.
"""
import csv
import sys

base = "tmp/delta_runs/delta1e-4/network_scan/network_roots.csv"
patch = "tmp/delta_runs/delta1e-4/network_scan_patch/network_roots.csv"
key = lambda r: (r["m_N"], r["y"])
if sys.argv[1] == "write":
    d0 = {key(r): r for r in csv.DictReader(open("tmp/network_scan/network_roots.csv"))}
    d1 = list(csv.DictReader(open(base)))
    seeds = {key(r): r for r in csv.DictReader(open("tmp/delta_runs/seeds/network_seeds.csv"))}
    pick = [r for r in d1 if (d0[key(r)]["status"] == "converged" and r["status"] != "converged")
            or 1e-3 * (1 - 1e-3) <= float(r["y"]) <= 3.2e-3]
    with open("tmp/delta_runs/seeds/network_seeds_patch.csv", "w") as io:
        io.write("m_N,y,sin2_2theta,omega_h2\n")
        for r in pick:
            s = seeds[key(r)]
            io.write(",".join((s["m_N"], s["y"], s["sin2_2theta"], s["omega_h2"])) + "\n")
    print(len(pick), "points to re-solve")
else:
    rows = list(csv.DictReader(open(base)))
    fields = list(rows[0].keys())
    new = {key(r): r for r in csv.DictReader(open(patch))}
    rows = [new.get(key(r), r) for r in rows]
    with open(base, "w") as io:
        io.write(",".join(fields) + "\n")
        for r in rows:
            io.write(",".join(r[f] for f in fields) + "\n")
    print(f"merged {len(new)} re-solved rows into {base}")

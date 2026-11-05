"""Seeds of the reduced-network checks: the stored roots of each scan, split by solver.
From julia/pandemic: python tmp/reduced_network/make_seeds.py"""
import os

SETS = {
    "base": "tmp/combined_scan",
    "mphi50": "tmp/mphi_scans/mphi50/combined",
    "mphi10": "tmp/mphi_scans/mphi10/combined",
    "mphi5.5": "tmp/mphi_scans/mphi5.5/combined",
    "mA3": "tmp/mA3_scan/combined",
    "delta1e-4": "tmp/delta_runs/delta1e-4/combined",
}
out = "tmp/reduced_network/seeds"
os.makedirs(out, exist_ok=True)
for name, d in SETS.items():
    rows = [l.strip().split(",") for l in open(f"{d}/roots.csv")]
    hdr, rows = rows[0], rows[1:]
    i = hdr.index("solver")
    for solver in ("network", "fluid"):
        sel = [r for r in rows if r[i] == solver]
        with open(f"{out}/{name}_{solver}.csv", "w") as f:
            f.write(",".join(hdr[:4]) + "\n")
            f.writelines(",".join(r[:4]) + "\n" for r in sel)
        print(name, solver, len(sel))

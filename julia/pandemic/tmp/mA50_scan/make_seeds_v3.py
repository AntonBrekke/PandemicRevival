"""Seeds of the m_A' = 50 fluid rerun with the final fluid (2026-11-08), from the current roots (run from
julia/pandemic): seeds/fluid_seeds_v3.csv (every fluid point; its root, or its bracket), and
seeds/fluid_bench_seeds_v3.csv (the off-grid fluid benchmarks at 300 and 600 keV, from pauli/fluid_roots.csv)."""
import csv

D = 'tmp/mA50_scan/'
HDR = 'm_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds\n'
row = lambda m, y, s2: f"{m},{y},{s2},0.12,{s2},0.12,seed,0,0\n"

rows = list(csv.DictReader(open(D + 'fluid_scan/fluid_roots.csv')))
with open(D + 'seeds/fluid_seeds_v3.csv', 'w') as f:
    f.write(HDR)
    for r in rows:
        s2 = r['s2_fluid'] if r['s2_fluid'] not in ('', 'NaN') else r['s2_network']
        f.write(row(r['m_N'], r['y'], s2))
print(len(rows), 'scan seeds')

bench = {('0.0003', '0.1'), ('0.0003', '1'), ('0.0006', '1.333')}
with open(D + 'seeds/fluid_bench_seeds_v3.csv', 'w') as f:
    f.write(HDR)
    n = 0
    for r in csv.DictReader(open(D + 'pauli/fluid_roots.csv')):
        if (r['m_N'], r['y']) in bench:
            f.write(row(r['m_N'], r['y'], r['s2_fluid'])); n += 1
print(n, 'benchmark seeds')

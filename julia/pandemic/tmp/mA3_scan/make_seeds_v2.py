"""Seeds of the m_A' = 3 rerun (2026-11-07) from the roots of the first scan (in place; run from julia/pandemic):
seeds/network_seeds_v2{,_fine}.csv (the network root where it converged, else its seed) and
seeds/fluid_points_v2{,_fine}.csv (the fluid root, or its bracket where it only bracketed)."""
import csv

D = 'tmp/mA3_scan/'
for s in ('', '_fine'):
    rows = list(csv.DictReader(open(D + f'network_scan{s}/network_roots.csv')))
    with open(D + f'seeds/network_seeds_v2{s}.csv', 'w') as f:
        f.write('m_N,y,sin2_2theta,omega_h2\n')
        for r in rows:
            s2 = r['s2_network'] if r['status'] == 'converged' else r['s2_reduced']
            f.write(f"{r['m_N']},{r['y']},{s2},0.12\n")
    print('network', s or 'base', len(rows), 'points,', sum(r['status'] == 'converged' for r in rows), 'converged before')
    rows = list(csv.DictReader(open(D + f'fluid_scan{s}/fluid_roots.csv')))
    with open(D + f'seeds/fluid_points_v2{s}.csv', 'w') as f:
        f.write('m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds\n')
        for r in rows:
            s2 = r['s2_fluid']
            if s2 in ('', 'NaN'):
                s2 = r['s2_network'] if r['s2_network'] not in ('', 'NaN') else r['s2_reduced']
            f.write(f"{r['m_N']},{r['y']},{s2},0.12,{s2},0.12,seed,0,0\n")
    print('fluid', s or 'base', len(rows), 'points,', sum(r['status'] == 'converged' for r in rows), 'converged before')

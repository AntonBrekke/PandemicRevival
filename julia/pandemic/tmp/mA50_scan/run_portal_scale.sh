#!/bin/bash
# Sensitivity of the large-g fluid roots to the portal 2 -> 2 table rates (MB, no Pauli blocking of the
# final N, which are nearly degenerate there): roots with --portal-scale 0.5 and 0.25 at three pilot
# points, seeded with the pilot roots. From julia/pandemic.
T=tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls
D=tmp/mA50_scan/portal_scale
mkdir -p $D
# seeds in the network_roots format: the pilot fluid roots
awk -F, 'NR == 1 {print "m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds"; next}
         {printf "%s,%s,%s,0.12,%s,0.12,seed,0,0\n", $1, $2, $7, $7}' tmp/mA50_scan/pilot/fluid/fluid_roots.csv > $D/seeds.csv
ONLY=10.57:1,300:1,10.57:0.1
for s in 0.5 0.25; do
    julia --project=. src/run_fluid_scan.jl --nproc 3 --tables $T --points $D/seeds.csv --only $ONLY \
        --portal-scale $s --out $D/scale$s > $D/scale$s.log 2>&1 &
done
wait
echo portal scale done

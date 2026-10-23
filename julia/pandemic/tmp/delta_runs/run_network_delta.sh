#!/bin/bash
# Network scans (base and fine grid) and their Lyman-alpha / f_N2 at delta = $1 (default 1e-4),
# seeded with the delta = 0 roots (plot/delta_seeds.py). From julia/pandemic.
set -e
D=${1:-1e-4}
T=tmp/rate_tables/rate_tables_mA2.5_mphi100.0_delta$(julia -e "print(parse(Float64, \"$D\"))").jls
OUT=tmp/delta_runs/delta$D
for s in "" _fine; do
    julia --project=. -t 12 src/run_network_scan.jl $T $OUT/network_scan$s tmp/delta_runs/seeds/network_seeds$s.csv
    julia --project=. -t 12 src/run_network_lyman_alpha.jl $OUT/network_scan$s/network_roots.csv $OUT/network_scan$s/lyman_alpha.csv $T
done
echo "network delta=$D done"

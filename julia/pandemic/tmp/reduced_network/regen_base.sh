#!/bin/bash
# Base network scans (both grids) and their Lyman-alpha lengths with the reduced network, then the
# combined scan. From julia/pandemic. Same seeds as the stored full-network scans.
set -e
THREADS=${THREADS:-8}
T=tmp/rate_tables/rate_tables_mA2.5_mphi100.0.jls
julia --project=. -t "$THREADS" src/run_network_scan.jl $T tmp/network_scan 2>&1 | tail -1
julia --project=. -t "$THREADS" src/run_network_scan.jl $T tmp/network_scan_fine tmp/network_scan_fine/seeds.csv 2>&1 | tail -1
for d in network_scan network_scan_fine; do
    julia --project=. -t "$THREADS" src/run_network_lyman_alpha.jl tmp/$d/network_roots.csv tmp/$d/lyman_alpha.csv 2>&1 | tail -1
done
conda run -n pandemic python plot/combined_scan_dir.py 2>&1 | tail -1
echo regen done

#!/bin/bash
# delta = 1e-4 network: re-solve the base points of patch_seeds.py, merge, then Lyman-alpha of
# the base grid, the fine grid scan and its Lyman-alpha. From julia/pandemic.
set -e
T=tmp/rate_tables/rate_tables_mA2.5_mphi100.0_delta0.0001.jls
OUT=tmp/delta_runs/delta1e-4
julia --project=. -t 12 src/run_network_scan.jl $T $OUT/network_scan_patch tmp/delta_runs/seeds/network_seeds_patch.csv
conda run -n pandemic python tmp/delta_runs/patch_seeds.py merge
julia --project=. -t 12 src/run_network_lyman_alpha.jl $OUT/network_scan/network_roots.csv $OUT/network_scan/lyman_alpha.csv $T
julia --project=. -t 12 src/run_network_scan.jl $T $OUT/network_scan_fine tmp/delta_runs/seeds/network_seeds_fine.csv
julia --project=. -t 12 src/run_network_lyman_alpha.jl $OUT/network_scan_fine/network_roots.csv $OUT/network_scan_fine/lyman_alpha.csv $T
echo "network delta=1e-4 done"

#!/bin/bash
# Network Lyman-alpha (base and fine grid) at delta = 1e-4, after lyman_alpha_network stopped at
# x = 100 for delta > 0 (the late conversions from with_conversions). From julia/pandemic.
set -e
T=tmp/rate_tables/rate_tables_mA2.5_mphi100.0_delta0.0001.jls
OUT=tmp/delta_runs/delta1e-4
for s in "" _fine; do
    julia --project=. -t 12 src/run_network_lyman_alpha.jl $OUT/network_scan$s/network_roots.csv $OUT/network_scan$s/lyman_alpha.csv $T
done
echo "network lyman delta=1e-4 redone"

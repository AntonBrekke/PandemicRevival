#!/bin/bash
# Reduced-network checks: relic roots from the stored roots of every scan, on their own tables.
# From julia/pandemic, after make_seeds.py. Roots land in <set>_<solver>/network_roots.csv, whose
# s2_reduced column is the stored root (the seed) and s2_network the reduced network's.
THREADS=${THREADS:-8}
D=tmp/reduced_network
T=tmp/rate_tables
scan() { # name tables
    julia --project=. -t "$THREADS" src/run_network_scan.jl "$2" "$D/$1" "$D/seeds/$1.csv" --reduced 2>&1 | tail -1
}
scan base_network "$T/rate_tables_mA2.5_mphi100.0.jls"
julia --project=. -t "$THREADS" src/run_network_lyman_alpha.jl "$D/base_network/network_roots.csv" "$D/base_network/lyman_alpha.csv" \
    "$T/rate_tables_mA2.5_mphi100.0.jls" 0.1 --reduced 2>&1 | tail -1
scan mphi50_network "$T/rate_tables_mA2.5_mphi50.0.jls"
scan mphi10_network "$T/rate_tables_mA2.5_mphi10.0.jls"
scan mphi5.5_network "$T/rate_tables_mA2.5_mphi5.5.jls"
scan mA3_network "$T/rate_tables_mA3.0_mphi100.0.jls"
scan delta1e-4_network "$T/rate_tables_mA2.5_mphi100.0_delta0.0001.jls"
scan base_fluid "$T/rate_tables_mA2.5_mphi100.0.jls"
echo all done

#!/bin/bash
# Pilot roots at m_A' = 50 m_N, m_phi = 150 m_N, before any scan: reduced network and fluid at a
# few (m, g) (make_pilot_seeds.py). From julia/pandemic.
T=tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls
D=tmp/mA50_scan/pilot
julia --project=. -t 8 src/run_network_scan.jl $T $D/network $D/network_seeds.csv > $D/network.log 2>&1 &
julia --project=. src/run_fluid_scan.jl --nproc 10 --tables $T --points $D/fluid_seeds.csv --out $D/fluid > $D/fluid.log 2>&1 &
wait
echo pilot done

#!/bin/bash
# Roots, Lyman-alpha lengths and histories of the benchmark points (bps.csv) with the current solvers,
# m_A' = 3 m_N, m_phi = 100 m_N, m_nu = 0.01 eV (M_NU). ~10 min. From julia/pandemic.
# Root searches start from the seeds in bp*/network_seeds.csv (BP2 also bp2_20keV/fluid_points.csv).
# After it, put the new roots into bps.csv and the README, then ./tmp/mA3_scan/make_figures.sh.
set -e
T=tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls
B=tmp/mA3_scan/bp_round
julia --project=. -t 2 src/run_network_scan.jl $T $B/bp1_10keV $B/bp1_10keV/network_seeds.csv
julia --project=. -t 2 src/run_network_scan.jl $T $B/bp2_20keV $B/bp2_20keV/network_seeds.csv
julia --project=. src/run_fluid_scan.jl --nproc 1 --points $B/bp2_20keV/fluid_points.csv --out $B/bp2_20keV --tables $T
julia --project=. -t 2 src/run_network_lyman_alpha.jl $B/bp1_10keV/network_roots.csv $B/bp1_10keV/lyman_alpha_network.csv
julia --project=. -t 2 src/run_network_lyman_alpha.jl $B/bp2_20keV/network_roots.csv $B/bp2_20keV/lyman_alpha_network.csv
julia --project=. src/run_fluid_lyman_alpha.jl --nproc 1 --roots $B/bp2_20keV/fluid_roots.csv \
    --out $B/bp2_20keV/lyman_alpha_fluid.csv --tables $T
# histories at the roots (sin^2 2theta from the files just written)
s2() { awk -F, -v c="$2" 'NR==1{for(i=1;i<=NF;i++) if($i==c) k=i} NR==2{print $k}' "$1"; }
S1=$(s2 $B/bp1_10keV/network_roots.csv s2_network)
S2N=$(s2 $B/bp2_20keV/network_roots.csv s2_network)
S2F=$(s2 $B/bp2_20keV/fluid_roots.csv s2_fluid)
julia --project=. -t 3 src/run_benchmarks.jl network:10:2e-4:$S1,network:20:1e-3:$S2N,fluid:20:1e-3:$S2F $B/benchmarks $T
echo "BP roots: BP1 $S1, BP2 network $S2N, fluid $S2F"

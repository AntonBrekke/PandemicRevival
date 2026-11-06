#!/bin/bash
# Coarse m_A' = 50 m_N, m_phi = 150 m_N scan, fluid part (g >= 1e-3), then its Lyman-alpha lengths and
# the combined scan. Seeds: make_coarse_seeds.py. $1 workers (default 9). From julia/pandemic.
set -e
T=tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls
D=tmp/mA50_scan
NP=${1:-9}
julia --project=. src/run_fluid_scan.jl --nproc $NP --tables $T --points $D/seeds/fluid_seeds.csv --out $D/fluid_scan > $D/fluid_scan.log 2>&1
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --tables $T --roots $D/fluid_scan/fluid_roots.csv \
    --out $D/fluid_scan/lyman_alpha.csv > $D/fluid_lya.log 2>&1
conda run -n pandemic python plot/combined_scan_dir.py $D/combined 1e-3 $D > $D/combined.log 2>&1
echo coarse fluid done

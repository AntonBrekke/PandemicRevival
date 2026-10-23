#!/bin/bash
# Fluid root scans (base and fine grid) at delta = $1 (default 1e-4) with $2 workers (default 10),
# seeded with the delta = 0 fluid roots (plot/delta_seeds.py). From julia/pandemic.
set -e
D=${1:-1e-4}
NP=${2:-10}
T=tmp/rate_tables/rate_tables_mA2.5_mphi100.0_delta$(julia -e "print(parse(Float64, \"$D\"))").jls
OUT=tmp/delta_runs/delta$D
for s in "" _fine; do
    julia --project=. src/run_fluid_scan.jl --nproc $NP --points tmp/delta_runs/seeds/fluid_points$s.csv \
        --out $OUT/fluid_scan$s --tables $T
done
echo "fluid roots delta=$D done"

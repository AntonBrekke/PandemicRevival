#!/bin/bash
# After run_fluid_delta.sh and run_network_delta.sh: fluid Lyman-alpha (base and fine grid), the
# combined scan directory and the figures, at delta = $1 (default 1e-4), $2 workers. From julia/pandemic.
set -e
D=${1:-1e-4}
NP=${2:-10}
T=tmp/rate_tables/rate_tables_mA2.5_mphi100.0_delta$(julia -e "print(parse(Float64, \"$D\"))").jls
OUT=tmp/delta_runs/delta$D
for s in "" _fine; do
    julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan$s/fluid_roots.csv \
        --out $OUT/fluid_scan$s/lyman_alpha.csv --tables $T
done
conda run -n pandemic python plot/combined_scan_dir.py $OUT/combined 1e-3 $OUT $D
mkdir -p figures/delta$D
conda run -n pandemic python plot/relic_scan_theta.py $OUT/combined figures/delta$D/relic_contours
conda run -n pandemic python plot/n2_fraction.py figures/delta$D/n2_over_n1 "delta = $D=$OUT/combined/lyman_alpha.csv"
echo "post delta=$D done"

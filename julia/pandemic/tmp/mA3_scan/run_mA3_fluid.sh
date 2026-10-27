#!/bin/bash
# Fluid part of the m_A' = 3 scan, second pass (after run_mA3.sh's network part and 6 fluid roots
# at g = 0.1778 from the first pass, fluid_scan/fluid_roots_pass1.csv): seeds from
# rescale_fluid_seeds_g.py, $1 workers (default 10). From julia/pandemic.
set -e
T=tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls
OUT=tmp/mA3_scan
NP=${1:-10}
FIG=figures/mA3
mkdir -p $FIG

figures() {
    conda run -n pandemic python plot/combined_scan_dir.py $OUT/combined 1e-3 $OUT
    conda run -n pandemic python plot/relic_scan_theta.py $OUT/combined $FIG/relic_contours --mA 3
    conda run -n pandemic python plot/relic_scan_theta.py $OUT/combined $FIG/relic_contours --mA 3 --xray-proj
    conda run -n pandemic python plot/relic_scan_theta.py $OUT/combined $FIG/relic_contours --mA 3 --debug
}

julia --project=. src/run_fluid_scan.jl --nproc $NP --points $OUT/seeds/fluid_points_scaled2.csv \
    --out $OUT/fluid_scan --tables $T --min-step 0.03
tail -n +2 $OUT/fluid_scan/fluid_roots_pass1.csv >> $OUT/fluid_scan/fluid_roots.csv
until [ -f $OUT/network_scan_fine/lyman_alpha.csv ]; do sleep 30; done
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan/fluid_roots.csv \
    --out $OUT/fluid_scan/lyman_alpha.csv --tables $T
figures
echo "base grid done $(date)"
julia --project=. src/run_fluid_scan.jl --nproc $NP --points $OUT/seeds/fluid_points_scaled2_fine.csv \
    --out $OUT/fluid_scan_fine --tables $T --min-step 0.03
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan_fine/fluid_roots.csv \
    --out $OUT/fluid_scan_fine/lyman_alpha.csv --tables $T
figures
echo "m_A' = 3 scan done $(date)"

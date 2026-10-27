#!/bin/bash
# Full scan at m_A' = 3 m_N (m_phi = 100 m_N), seeded from the m_A' = 2.5 roots, with $1 fluid
# workers (default 10). From julia/pandemic, after
#     julia --project=. -t 12 src/run_build_rate_tables.jl --mA 3
#     conda run -n pandemic python plot/delta_seeds.py tmp/mA3_scan/seeds
# Order: network roots (both grids, side by side: a few slow points hold up the end of each);
# fluid roots and Lyman-alpha on the base grid (network Lyman-alpha alongside) and the figures;
# then the fine fluid grid and the figures again. SKIP_BASE_NETWORK=1 skips the base network
# scan (when it is already running or done).
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

julia --project=. -t 10 src/run_network_scan.jl $T $OUT/network_scan_fine $OUT/seeds/network_seeds_fine.csv &
[ -n "$SKIP_BASE_NETWORK" ] || julia --project=. -t 6 src/run_network_scan.jl $T $OUT/network_scan $OUT/seeds/network_seeds.csv
wait
until [ -f $OUT/network_scan/network_roots.csv ]; do sleep 30; done
conda run -n pandemic python $OUT/scale_fluid_seeds.py
(
    for s in "" _fine; do
        julia --project=. -t 4 src/run_network_lyman_alpha.jl $OUT/network_scan$s/network_roots.csv \
            $OUT/network_scan$s/lyman_alpha.csv $T
    done
) > $OUT/network_lyman_alpha.log 2>&1 &
julia --project=. src/run_fluid_scan.jl --nproc $NP --points $OUT/seeds/fluid_points_scaled.csv \
    --out $OUT/fluid_scan --tables $T --min-step 0.03
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan/fluid_roots.csv \
    --out $OUT/fluid_scan/lyman_alpha.csv --tables $T
wait
figures
echo "base grid done $(date)"
julia --project=. src/run_fluid_scan.jl --nproc $NP --points $OUT/seeds/fluid_points_scaled_fine.csv \
    --out $OUT/fluid_scan_fine --tables $T --min-step 0.03
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan_fine/fluid_roots.csv \
    --out $OUT/fluid_scan_fine/lyman_alpha.csv --tables $T
figures
echo "m_A' = 3 scan done $(date)"

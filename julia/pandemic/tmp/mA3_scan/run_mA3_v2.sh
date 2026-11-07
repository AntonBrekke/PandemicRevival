#!/bin/bash
# Rerun of the m_A' = 3 m_N scan (both grids) with the current solvers (2026-11-07): reduced network,
# m_nu = 0.01 eV, fluid with the Pauli-corrected portal rates and the cold-temperature fix. Seeds: the
# roots of the first scan (make_seeds_v2.py). Overwrites network_scan{,_fine}, fluid_scan{,_fine},
# combined (the first scan is in git history). $1 fluid workers (default 10). From julia/pandemic.
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

echo "start $(date)"
julia --project=. -t 6 src/run_network_scan.jl $T $OUT/network_scan_fine $OUT/seeds/network_seeds_v2_fine.csv > $OUT/network_fine.log 2>&1 &
julia --project=. -t 6 src/run_network_scan.jl $T $OUT/network_scan $OUT/seeds/network_seeds_v2.csv > $OUT/network.log 2>&1
wait
echo "network done $(date)"
(
    for s in "" _fine; do
        julia --project=. -t 4 src/run_network_lyman_alpha.jl $OUT/network_scan$s/network_roots.csv \
            $OUT/network_scan$s/lyman_alpha.csv $T
    done
) > $OUT/network_lyman_alpha.log 2>&1 &
julia --project=. src/run_fluid_scan.jl --nproc $NP --points $OUT/seeds/fluid_points_v2.csv \
    --out $OUT/fluid_scan --tables $T --min-step 0.03 > $OUT/fluid.log 2>&1
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan/fluid_roots.csv \
    --out $OUT/fluid_scan/lyman_alpha.csv --tables $T > $OUT/fluid_lya.log 2>&1
wait
echo "base grid done $(date)"
julia --project=. src/run_fluid_scan.jl --nproc $NP --points $OUT/seeds/fluid_points_v2_fine.csv \
    --out $OUT/fluid_scan_fine --tables $T --min-step 0.03 > $OUT/fluid_fine.log 2>&1
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan_fine/fluid_roots.csv \
    --out $OUT/fluid_scan_fine/lyman_alpha.csv --tables $T > $OUT/fluid_fine_lya.log 2>&1
figures > $OUT/figures.log 2>&1
echo "m_A' = 3 rerun done $(date)"

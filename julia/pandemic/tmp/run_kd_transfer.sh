#!/bin/bash
# Rerun of everything that uses Gamma_kd after the momentum-transfer weighting of its cross sections
# (2026-11-13, kinetic_decoupling.jl `transfer`): the Lyman-alpha lengths of all roots of the
# m_A' = 3 scan (both grids) and the m_A' = 50 scan, the combined dirs, and every benchmark history
# of both (Gamma_kd_H column, the "dark thermalisation" of the plots). Roots are unchanged (Gamma_kd
# is post-processing only). $1 fluid workers (default 10). From julia/pandemic.
set -e
NP=${1:-10}
T3=tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls
T50=tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls
S3=tmp/mA3_scan
S50=tmp/mA50_scan
echo "start $(date)"

# --- Lyman-alpha: network (threads) next to the fluid (workers)
(
    for s in "" _fine; do
        julia --project=. -t 4 src/run_network_lyman_alpha.jl $S3/network_scan$s/network_roots.csv $S3/network_scan$s/lyman_alpha.csv $T3
    done
    julia --project=. -t 4 src/run_network_lyman_alpha.jl $S50/network_scan/network_roots.csv $S50/network_scan/lyman_alpha.csv $T50
) > tmp/kd_transfer_network_lya.log 2>&1 &
for s in "" _fine; do
    julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $S3/fluid_scan$s/fluid_roots.csv \
        --out $S3/fluid_scan$s/lyman_alpha.csv --tables $T3
done > tmp/kd_transfer_fluid_lya.log 2>&1
julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --tables $T50 --roots $S50/fluid_scan/fluid_roots.csv \
    --out $S50/fluid_scan/lyman_alpha.csv >> tmp/kd_transfer_fluid_lya.log 2>&1
wait
echo "Lyman-alpha done $(date)"
conda run -n pandemic python plot/combined_scan_dir.py $S3/combined 1e-3 $S3
conda run -n pandemic python plot/combined_scan_dir.py $S50/combined 1e-3 $S50

# --- benchmark histories at the stored roots (as run_benchmarks_v2.sh, bp_round/run_bps.sh, run_benchmarks_v3.sh)
julia --project=. -t 8 src/run_benchmarks.jl \
    network:10.57:1e-7:8.43658e-10,network:10.57:1e-5:8.46427e-11,network:10.57:1e-4:1.48505e-12,network:10.57:1e-3:3.77701e-15,fluid:10.57:1e-3:3.775e-15,fluid:10.57:0.01:1.27089e-17,fluid:10.57:0.05623:7.54963e-20 \
    $S3/benchmarks $T3
julia --project=. -t 8 src/run_benchmarks.jl \
    network:10.57:1.333e-4:7.32767e-13,network:13.72:3.162e-4:1.07198e-13,network:17.81:4.217e-4:6.01572e-14 \
    $S3/benchmarks_xray $T3
julia --project=. -t 3 src/run_benchmarks.jl \
    network:10:2e-4:2.63318e-13,network:20:1e-3:5.34854e-15,fluid:20:1e-3:5.34707e-15 \
    $S3/bp_round/benchmarks $T3
julia --project=. -t 8 src/run_benchmarks.jl \
    network:10.57:1e-7:8.43658e-10,network:10.57:1e-5:4.18692e-10,network:10.57:7.499e-4:1.10052e-13,network:300:1e-4:1.6517e-12,fluid:10.57:1e-3:4.08961e-14,fluid:10.57:0.01:2.2992e-17,fluid:10.57:0.1:1.45443e-19,fluid:10.57:1:1.03263e-22,fluid:300:0.1:2.94588e-18,fluid:300:1:7.9235e-21,fluid:600:1.333:7.73039e-21 \
    $S50/benchmarks $T50
echo "benchmarks done $(date)"

# --- figures
plot/make_figures.sh mA3
plot/make_figures.sh mA50
echo "all done $(date)"
# Run on 2026-11-13 (ThinkPad P14s, 10 workers): Lyman-alpha 2 h 20 min, histories 22 min. The x_fo
# column of the m_A' = 3 network (both grids) and fluid base grid was then redone with the corrected
# relaxation rate (those processes had started before it); combined dirs and figures rebuilt after.

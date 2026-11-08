#!/bin/bash
# m_A' = 50 benchmark histories with the final code (2026-11-08). Fluid roots: grid points from fluid_scan/,
# off-grid ones (300, 600 keV; 10.57 keV at g = 1) from benchmark_roots/. Network roots as before (the
# network did not change). Figures to figures/mA50/. From julia/pandemic.
set -e
T=tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls
D=tmp/mA50_scan
julia --project=. -t 8 src/run_benchmarks.jl \
    network:10.57:1e-7:8.43658e-10,network:10.57:1e-5:4.18692e-10,network:10.57:7.499e-4:1.10052e-13,network:300:1e-4:1.6517e-12,fluid:10.57:1e-3:4.08961e-14,fluid:10.57:0.01:2.2992e-17,fluid:10.57:0.1:1.45443e-19,fluid:10.57:1:1.03263e-22,fluid:300:0.1:2.94588e-18,fluid:300:1:7.9235e-21,fluid:600:1.333:7.73039e-21 \
    $D/benchmarks $T
mkdir -p figures/mA50
conda run -n pandemic python plot/production.py $D/benchmarks figures/mA50/production --mA 50
conda run -n pandemic python plot/benchmark.py $D/benchmarks figures/mA50/benchmarks
echo "benchmarks done $(date)"

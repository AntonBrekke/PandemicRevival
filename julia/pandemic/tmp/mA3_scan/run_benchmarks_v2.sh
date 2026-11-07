#!/bin/bash
# Benchmark histories of the m_A' = 3 rerun (2026-11-07), at its roots: the standard set at 10.57 keV
# (benchmarks/) and the X-ray BPs of xray_bps.csv (benchmarks_xray/). The fluid at g = 1e-3 only
# bracketed its root (3.775e-15; network root 3.777e-15). From julia/pandemic.
set -e
T=tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls
OUT=tmp/mA3_scan
julia --project=. -t 8 src/run_benchmarks.jl \
    network:10.57:1e-7:8.43658e-10,network:10.57:1e-5:8.46427e-11,network:10.57:1e-4:1.48505e-12,network:10.57:1e-3:3.77701e-15,fluid:10.57:1e-3:3.775e-15,fluid:10.57:0.01:1.27089e-17,fluid:10.57:0.05623:7.54963e-20 \
    $OUT/benchmarks $T
julia --project=. -t 8 src/run_benchmarks.jl \
    network:10.57:1.333e-4:7.32767e-13,network:13.72:3.162e-4:1.07198e-13,network:17.81:4.217e-4:6.01572e-14 \
    $OUT/benchmarks_xray $T
mkdir -p figures/mA3
for s in "" _xray; do
    conda run -n pandemic python plot/production.py $OUT/benchmarks$s figures/mA3/production$s --mA 3
    conda run -n pandemic python plot/benchmark.py $OUT/benchmarks$s figures/mA3/benchmarks$s
done
echo "benchmarks done $(date)"

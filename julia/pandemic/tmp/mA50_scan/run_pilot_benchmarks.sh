#!/bin/bash
# Benchmarks (histories with Gamma_kd, conv_H, rates) at pilot roots of m_A' = 50 m_N, m_phi = 150 m_N.
# From julia/pandemic.
T=tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls
D=tmp/mA50_scan/benchmarks
mkdir -p $D
julia --project=. src/run_benchmarks.jl \
    network:10.57:1e-07:8.43658e-10,network:10.57:1e-05:4.18692e-10,network:10.57:0.0007499:1.10052e-13,network:300:0.0001:1.6517e-12,fluid:10.57:0.001:4.08546e-14 \
    $D $T > $D/run_a.log 2>&1 &
julia --project=. src/run_benchmarks.jl \
    fluid:10.57:0.01:2.29756e-17,fluid:10.57:0.1:1.45443e-19,fluid:10.57:1:8.62808e-23 \
    $D $T > $D/run_b.log 2>&1 &
julia --project=. src/run_benchmarks.jl \
    fluid:300:0.1:2.93003e-18,fluid:600:1.333:7.23189e-21,fluid:300:1:7.41207e-21 \
    $D $T > $D/run_c.log 2>&1 &
wait
echo benchmarks done

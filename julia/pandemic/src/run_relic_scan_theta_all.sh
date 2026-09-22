#!/usr/bin/env bash
# Relic scan over (m_N, y): root-finds sin^2(2 theta) for Omega h^2 = 0.12 at
# every mass and coupling, in parallel (src/run_relic_scan_theta_all.jl).
# Results: tmp/relic_scan_theta/m_N_<m_N>keV.csv, logs next to them.
#
# Usage (from julia/pandemic):  bash src/run_relic_scan_theta_all.sh [N_PROC]
# For other masses/couplings call the Julia driver directly (see its --help).
set -euo pipefail
cd "$(dirname "$0")/.."

N_PROC=${1:-12}
OUT=tmp/relic_scan_theta

mkdir -p "$OUT"
julia --project=. src/run_relic_scan_theta_all.jl \
    --masses 1.5,2.5,4,6.5,10,16,25,40,65,100,160,250 \
    --ys 1e-6,3.16e-6,1e-5,3.16e-5,1e-4,3.16e-4,1e-3,3.16e-3,1e-2 \
    --nproc "$N_PROC" --out "$OUT" 2>&1 | tee "$OUT/run.log"

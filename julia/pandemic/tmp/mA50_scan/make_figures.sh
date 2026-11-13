#!/bin/bash
# All figures of the m_A' = 50 m_N (m_phi = 150 m_N) coarse scan from the committed data. From julia/pandemic.
# [--debug]: also the debugging version of the benchmark figures.
set -e
S=tmp/mA50_scan
F=figures/mA50
P="conda run -n pandemic python"
FRAME="--mA 50 --mlim 1 800 --slim 1e-24 1e-8 --g-pert 1.333"
mkdir -p $F
$P plot/relic_scan_theta.py $S/combined $F/relic_contours $FRAME
$P plot/relic_scan_theta.py $S/combined $F/relic_contours $FRAME --all-points
$P plot/relic_scan_theta.py $S/combined $F/relic_contours $FRAME --n2-fraction
$P plot/relic_scan_theta.py $S/combined $F/relic_contours $FRAME --n2-fraction --dm-contours --bound-lines
$P plot/benchmark.py $S/benchmarks $F/benchmarks --mA 50
if [ "$1" = "--debug" ]; then $P plot/benchmark_debug.py $S/benchmarks $F/benchmarks_debug; fi
echo "figures in $F"

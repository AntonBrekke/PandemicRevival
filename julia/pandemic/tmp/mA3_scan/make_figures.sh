#!/bin/bash
# All figures of the m_A' = 3 m_N scan from the committed data (~3 min). From julia/pandemic.
# [--debug]: also the debugging version of the benchmark figures.
# Benchmark points: bp_round/bps.csv (BP1 10 keV g = 2e-4, BP2 20 keV g = 1e-3); xray_bps.csv holds the
# older X-ray candidates, drawn only in relic_contours_xraybps_xproj.pdf. See README.md.
set -e
S=tmp/mA3_scan
F=figures/mA3
BPS=$S/bp_round/bps.csv
P="conda run -n pandemic python"
mkdir -p $F
# money plots (--xray-proj writes only <out>_xproj.pdf, so the plain one needs its own run)
$P plot/relic_scan_theta.py $S/combined $F/relic_contours --mA 3
$P plot/relic_scan_theta.py $S/combined $F/relic_contours --mA 3 --xray-proj
$P plot/relic_scan_theta.py $S/combined $F/relic_contours --mA 3 --debug
$P plot/relic_scan_theta.py $S/combined $F/relic_contours --mA 3 --n2-fraction
# N2/N1 today with lines of constant N1-N2 splitting and the constraints as edges only
$P plot/relic_scan_theta.py $S/combined $F/relic_contours --mA 3 --n2-fraction --dm-contours --bound-lines
$P plot/relic_scan_theta.py $S/combined $F/relic_contours_bp --mA 3 --bp $BPS
$P plot/relic_scan_theta.py $S/combined $F/relic_contours_bp --mA 3 --bp $BPS --xray-proj
$P plot/relic_scan_theta.py $S/combined $F/relic_contours_xraybps --mA 3 --bp $S/xray_bps.csv --xray-proj
# benchmark figures (plot/benchmark.py): BPs, standard 10.57 keV set, X-ray candidates;
# with --debug also the per-process debugging version (plot/benchmark_debug.py) in benchmarks_debug*/
for pair in "bp_round/benchmarks:_bp_round" "benchmarks:" "benchmarks_xray:_xray"; do
    d=${pair%%:*}; s=${pair#*:}
    $P plot/benchmark.py $S/$d $F/benchmarks$s --mA 3
    if [ "$1" = "--debug" ]; then $P plot/benchmark_debug.py $S/$d $F/benchmarks_debug$s; fi
done
echo "figures in $F"

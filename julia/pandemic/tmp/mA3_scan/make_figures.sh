#!/bin/bash
# All figures of the m_A' = 3 m_N scan from the committed data (~3 min). From julia/pandemic.
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
$P plot/relic_scan_theta.py $S/combined $F/relic_contours_bp --mA 3 --bp $BPS
$P plot/relic_scan_theta.py $S/combined $F/relic_contours_bp --mA 3 --bp $BPS --xray-proj
$P plot/relic_scan_theta.py $S/combined $F/relic_contours_xraybps --mA 3 --bp $S/xray_bps.csv --xray-proj
# histories: BPs, standard 10.57 keV set, X-ray candidates
for pair in "bp_round/benchmarks:_bp_round" "benchmarks:" "benchmarks_xray:_xray"; do
    d=${pair%%:*}; s=${pair#*:}
    $P plot/production.py $S/$d $F/production$s --mA 3
    $P plot/benchmark.py $S/$d $F/benchmarks$s
done
echo "figures in $F"

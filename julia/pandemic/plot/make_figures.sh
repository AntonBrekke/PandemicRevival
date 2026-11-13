#!/bin/bash
# All figures of one scan from the committed data. From julia/pandemic:
#     plot/make_figures.sh mA3 [--debug]     # m_A' = 3 m_N (m_phi = 100 m_N), tmp/mA3_scan  -> figures/mA3 (~3 min)
#     plot/make_figures.sh mA50 [--debug]    # m_A' = 50 m_N (m_phi = 150 m_N), tmp/mA50_scan -> figures/mA50
# --debug: also the debugging version of the benchmark figures (plot/benchmark_debug.py) in benchmarks_debug*/.
#
# m_A' = 3: benchmark points are tmp/mA3_scan/bp_round/bps.csv (BP1 10 keV g = 2e-4, BP2 20 keV g = 1e-3);
# tmp/mA3_scan/xray_bps.csv holds the older X-ray candidates, drawn only in relic_contours_xraybps_xproj.pdf.
# See the README of each scan.
set -e
SCAN=$1
DEBUG=$2
case "$SCAN" in
    mA3)  FRAME="--mA 3" ;;
    mA50) FRAME="--mA 50 --mlim 1 800 --slim 1e-24 1e-8 --g-pert 1.333" ;;
    *) echo "usage: plot/make_figures.sh {mA3|mA50} [--debug]" >&2; exit 1 ;;
esac
S=tmp/${SCAN}_scan
F=figures/$SCAN
MA=${SCAN#mA}
P="conda run -n pandemic python"
mkdir -p $F

money() { $P plot/relic_scan_theta.py $S/combined "$@" $FRAME; }
benchmarks() {   # benchmarks <history dir under $S> <suffix of the figure dirs>
    $P plot/benchmark.py $S/$1 $F/benchmarks$2 --mA $MA
    if [ "$DEBUG" = "--debug" ]; then $P plot/benchmark_debug.py $S/$1 $F/benchmarks_debug$2; fi
}

if [ "$SCAN" = mA3 ]; then
    BPS=$S/bp_round/bps.csv
    # money plots (--xray-proj writes only <out>_xproj.pdf, so the plain one needs its own run)
    money $F/relic_contours
    money $F/relic_contours --xray-proj
    money $F/relic_contours --debug
    money $F/relic_contours --n2-fraction
    # N2/N1 today with lines of constant N1-N2 splitting and the constraints as edges only
    money $F/relic_contours --n2-fraction --dm-contours --bound-lines
    money $F/relic_contours_bp --bp $BPS
    money $F/relic_contours_bp --bp $BPS --xray-proj
    money $F/relic_contours_xraybps --bp $S/xray_bps.csv --xray-proj
    # benchmark figures (plot/benchmark.py): BPs, standard 10.57 keV set, X-ray candidates
    benchmarks bp_round/benchmarks _bp_round
    benchmarks benchmarks ""
    benchmarks benchmarks_xray _xray
else
    money $F/relic_contours
    money $F/relic_contours --all-points
    money $F/relic_contours --n2-fraction
    money $F/relic_contours --n2-fraction --dm-contours --bound-lines
    benchmarks benchmarks ""
fi
echo "figures in $F"

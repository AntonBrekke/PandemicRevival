#!/usr/bin/env bash
# Coarse scans at other m_phi (see tmp/mphi_scans/README): network, fluid, Lyman-alpha, combined.
# Run from julia/pandemic.
set -u
S=tmp/mphi_scans
T() { echo tmp/rate_tables/rate_tables_mA2.5_mphi$1.jls; }
log() { echo "[$(date +%H:%M)] $*"; }
for r in 50.0 10.0 5.5 1.5; do
    d=$S/mphi${r%.0}; mkdir -p $d/network_scan $d/fluid_scan
    seeds=$S/seeds/network_seeds_overlap.csv; [ $r = 1.5 ] && seeds=$S/seeds/network_seeds_all.csv
    julia --project=. -t 8 src/run_network_scan.jl $(T $r) $d/network_scan $seeds > $d/network_scan/scan.log 2>&1
    log "network $r: $(awk -F, 'NR>1{print $7}' $d/network_scan/network_roots.csv | sort | uniq -c | tr '\n' ' ')"
done
for r in 50.0 10.0 5.5; do
    d=$S/mphi${r%.0}
    julia --project=. src/run_fluid_scan.jl --nproc 12 --tables $(T $r) --points $S/seeds/fluid_seeds.csv --out $d/fluid_scan > $d/fluid_scan/scan.log 2>&1
    log "fluid $r: $(awk -F, 'NR>1{print $9}' $d/fluid_scan/fluid_roots.csv | sort | uniq -c | tr '\n' ' ')"
done
for r in 50.0 10.0 5.5 1.5; do
    d=$S/mphi${r%.0}
    julia --project=. -t 8 src/run_network_lyman_alpha.jl $d/network_scan/network_roots.csv $d/network_scan/lyman_alpha.csv $(T $r) > $d/network_scan/lya.log 2>&1
    log "network lyman $r: $(tail -n 1 $d/network_scan/lya.log)"
done
for r in 50.0 10.0 5.5; do
    d=$S/mphi${r%.0}
    julia --project=. src/run_fluid_lyman_alpha.jl --nproc 12 --tables $(T $r) --roots $d/fluid_scan/fluid_roots.csv --out $d/fluid_scan/lyman_alpha.csv > $d/fluid_scan/lya.log 2>&1
    log "fluid lyman $r: $(tail -n 1 $d/fluid_scan/lya.log)"
done
log done

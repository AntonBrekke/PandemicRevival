#!/bin/bash
# Lyman-alpha post-processing of the m_A' = 3 roots with the N1-N2 splitting at m_nu = 0.01 eV.
# The roots are those of tmp/mA3_scan (copied here unchanged for combined_scan_dir.py), not re-searched.
# $1 fluid workers (default 11). From julia/pandemic.
set -e
T=tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls
SRC=tmp/mA3_scan
OUT=$SRC/mnu0.01
MNU=0.01
NP=${1:-11}

for s in "" _fine; do
    mkdir -p $OUT/network_scan$s $OUT/fluid_scan$s
    cp $SRC/network_scan$s/network_roots.csv $OUT/network_scan$s/
    cp $SRC/fluid_scan$s/fluid_roots.csv $OUT/fluid_scan$s/
done
for s in "" _fine; do
    echo "network$s start $(date)"
    julia --project=. -t 12 src/run_network_lyman_alpha.jl $OUT/network_scan$s/network_roots.csv \
        $OUT/network_scan$s/lyman_alpha.csv $T $MNU
done
for s in "" _fine; do
    echo "fluid$s start $(date)"
    julia --project=. src/run_fluid_lyman_alpha.jl --nproc $NP --roots $OUT/fluid_scan$s/fluid_roots.csv \
        --out $OUT/fluid_scan$s/lyman_alpha.csv --tables $T --m-nu $MNU
done
conda run -n pandemic python plot/combined_scan_dir.py $OUT/combined 1e-3 $OUT
echo "ALL DONE $(date)"

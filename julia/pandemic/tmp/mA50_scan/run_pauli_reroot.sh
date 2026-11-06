#!/bin/bash
# Large-g fluid roots with the Pauli-corrected portal 2 -> 2 rates (portal_tables.jl, on by default),
# seeded with the MB pilot roots (portal_scale/seeds.csv). From julia/pandemic.
T=tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls
D=tmp/mA50_scan/pauli
mkdir -p $D
ONLY=10.57:0.1,10.57:0.5,10.57:1,10.57:1.333,300:0.1,300:1,300:1.333,600:1,600:1.333
julia --project=. src/run_fluid_scan.jl --nproc 9 --tables $T --points tmp/mA50_scan/portal_scale/seeds.csv \
    --only $ONLY --out $D > $D/reroot.log 2>&1
echo pauli reroot done

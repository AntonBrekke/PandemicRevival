#!/bin/bash
# Pauli-correction tables (portal_tables.jl) for the mass ratios in use, each followed by its checks.
# From julia/pandemic. Logs next to the tables.
D=tmp/rate_tables
for spec in "3 100" "50 150"; do
    set -- $spec
    julia --project=. -t 16 src/run_build_portal_tables.jl --mA $1 --mphi $2 > $D/portal_exact_mA$1.log 2>&1
    julia --project=. -t 16 test/check_portal_tables.jl $D/portal_exact_mA$1.0_mphi$2.0.jls 40 > $D/portal_exact_mA$1_check.log 2>&1
done
julia --project=. test/test_fluid.jl > $D/portal_exact_test_fluid.log 2>&1
echo portal tables done

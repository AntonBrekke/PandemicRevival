# Benchmark points with round m_N and g, m_A' = 3 m_N (2026-11-05)

Off the scan grid, so each has its own root search (seed interpolated from `tmp/mA3_scan/combined`);
Lyman-alpha at m_nu = 0.01 eV. `bps.csv` holds the current choice (`relic_scan_theta.py --bp`).

| | m_N | g | sin^2 2theta | files |
|---|---|---|---|---|
| BP1 (current) | 10 keV | 2e-4 | 2.63317e-13 | `bp1_10keV/` |
| BP2 (current) | 20 keV | 1e-3 | 5.34841e-15 (network), 5.34e-15 (fluid) | `bp2_20keV/` |
| earlier BP1 | 14 keV | 3e-4 | 1.2314e-13 | this directory |
| earlier BP2 | 30 keV | 3e-3 | 5.24e-16 (fluid; network degenerate) | this directory |

Peak forward rates over H (`benchmarks/`): BP1 N2 nu -> A' 15.5, NN -> A'A' 0.04; BP2 5.2 and 9.9.
Earlier BP2 replaced because NN -> A'A' (1.3e4 H) was too extreme.

Histories: `julia --project=. -t 2 src/run_benchmarks.jl network:10:2e-4:2.63317e-13,network:20:1e-3:5.34841e-15 <out> tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls`

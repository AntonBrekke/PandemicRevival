# Benchmark points with round m_N and g, m_A' = 3 m_N (chosen 2026-11-05; recomputed 2026-11-13)

**These are the benchmark points** of the m_A' = 3 scan: `bps.csv` is what `relic_scan_theta.py --bp`
takes. (`../xray_bps.csv` holds the older X-ray candidates, not the BPs; see `../README.md`.)

Off the scan grid, so each has its own root search (seed interpolated from `tmp/mA3_scan/combined`).
Recomputed on 2026-11-13 with the final solvers (reduced network, Pauli-blocked fluid with the
cold-temperature fix, m_nu = 0.01 eV) by `run_bps.sh`; the 11-05 values were from the full network
before its reservoir fix and an unsearched fluid seed.

| | m_N | g | sin^2 2theta | Omega h^2 | lambda_fs,kd / r_s [Mpc] |
|---|---|---|---|---|---|
| BP1 | 10 keV | 2e-4 | 2.63318e-13 (network; 11-05: 2.63317e-13) | 0.1200 | 0.197 / 0.058 |
| BP2 | 20 keV | 1e-3 | 5.34854e-15 (network; 11-05: 5.34841e-15), 5.34707e-15 (fluid; 11-05: seed 5.34e-15) | 0.1199 / 0.1205 | 0.056 / 0.046 |

BP2 sits at the solver switch (g = 1e-3): network and fluid roots agree to 0.03%; `bps.csv` uses
the network root (the fluid one would be the scan's choice at g >= 1e-3; indistinguishable in the plot).

Files: `bp1_10keV/`, `bp2_20keV/` (seeds, roots, Lyman-alpha lengths); `benchmarks/` (histories of
both, BP2 with both solvers; figures `figures/mA3/{benchmarks,production}_bp_round/` by
`../make_figures.sh`); `earlier/` (an earlier choice, 14 keV g = 3e-4 and 30 keV g = 3e-3, from
the 11-05 code; BP2 there replaced because NN -> A'A' at 1.3e4 H was too extreme).

Peak forward rates over H (11-05): BP1 N2 nu -> A' 15.5, NN -> A'A' 0.04; BP2 5.2 and 9.9.

Recompute: `./tmp/mA3_scan/bp_round/run_bps.sh` (~10 min), then update `bps.csv` and this table.

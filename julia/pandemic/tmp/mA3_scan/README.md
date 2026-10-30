# Scan at m_A' = 3 m_N (2026-10-27)

m_phi = 100 m_N, equal-mass tables (`tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls`, 44 min on
12 threads), post-processing with the N1-N2 splitting at m_nu = 0.1 eV (`mnu_splitting`; changes
nothing at 0.1 eV: N2 is not depleted anywhere). Everything else as the main scan.

## Contents

- `seeds/`: m_A' = 2.5 roots (`plot/delta_seeds.py tmp/mA3_scan/seeds`); fluid seeds rescaled by
  `scale_fluid_seeds.py` (network root ratio m_A' = 3/2.5 in the overlap, per mass) and then
  `rescale_fluid_seeds_g.py` (ratio interpolated in ln g up to 1.28 at g = 0.1778): with the
  first, the large-g fluid points needed 4-6 solves; with the second ~3.
- `network_scan{,_fine}/`: roots and `lyman_alpha.csv` (with `x_fo`, `delta`). 483/654 and
  420/420 converged; the others are degenerate or failed at g >~ 3e-3 (fluid domain), except
  75.36 keV at g = 1e-3.
- `fluid_scan/`: roots (base grid; 222 converged, 12 bracketed) and `lyman_alpha.csv`.
  `fluid_roots_pass1.csv`: the 6 roots of the first pass (g = 0.1778), merged into fluid_roots.csv.
- `fluid_scan_fine/` (2026-10-30, ThinkStation, 12 workers: 127 min + 36 min Lyman-alpha): roots of
  the fine grid (200 converged, 8 bracketed) and `lyman_alpha.csv`.
- `combined/`: `plot/combined_scan_dir.py tmp/mA3_scan/combined 1e-3 tmp/mA3_scan`: 852 network and
  422 fluid roots, 1274 with Lyman-alpha lengths (both grids, 8 values of g per decade throughout).
- `xray_bps.csv`: benchmark points allowed by every current bound and inside the projected reach of
  Athena/eXTP (`plot/xray_benchmarks.py tmp/mA3_scan/combined ... --mA 3`). Histories in
  `benchmarks_xray/`; standard histories at 10.57 keV in `benchmarks/`.
- `compare_roots.py`: roots against m_A' = 2.5 per g.

## Results

- Roots s2(3)/s2(2.5): 1 for g <~ 1e-7, 1.15 - 1.2 at g ~ 1e-5 - 5e-4, 0.92 - 0.95 at
  g ~ 2e-3 - 1e-2, rising to 1.28 at g = 0.1778.
- Network vs fluid in the overlap: roots agree to a median 0.25% (g = 1e-3), 0.08% (1.8e-3),
  0.02% (3.2e-3).
- Money plot (`figures/mA3/relic_contours{,_xproj,_debug}.pdf`): like m_A' = 2.5; self-int.
  region smaller (sigma/m ∝ m_A'^-4), Lyman-alpha boundaries nearly unchanged.

## Commands (from julia/pandemic)

    julia --project=. -t 12 src/run_build_rate_tables.jl --mA 3
    conda run -n pandemic python plot/delta_seeds.py tmp/mA3_scan/seeds
    ./tmp/mA3_scan/run_mA3.sh 10          # network + first fluid pass (stopped after 6 roots)
    python tmp/mA3_scan/rescale_fluid_seeds_g.py tmp/mA3_scan/fluid_scan/fluid_roots_pass1.csv
    python tmp/mA3_scan/rescale_fluid_seeds_g.py tmp/mA3_scan/fluid_scan/fluid_roots_pass1.csv _fine
    ./tmp/mA3_scan/run_mA3_fluid.sh 11    # fluid base grid, Lyman-alpha, figures (fine grid: stopped)

Fine fluid grid (~2 h + ~0.6 h Lyman-alpha on the ThinkStation with 12 workers):

    julia --project=. src/run_fluid_scan.jl --nproc 11 --points tmp/mA3_scan/seeds/fluid_points_scaled2_fine.csv \
        --out tmp/mA3_scan/fluid_scan_fine --tables tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls --min-step 0.03
    julia --project=. src/run_fluid_lyman_alpha.jl --nproc 11 --roots tmp/mA3_scan/fluid_scan_fine/fluid_roots.csv \
        --out tmp/mA3_scan/fluid_scan_fine/lyman_alpha.csv --tables tmp/rate_tables/rate_tables_mA3.0_mphi100.0.jls
    conda run -n pandemic python plot/combined_scan_dir.py tmp/mA3_scan/combined 1e-3 tmp/mA3_scan
    conda run -n pandemic python plot/relic_scan_theta.py tmp/mA3_scan/combined figures/mA3/relic_contours --mA 3 [--xray-proj --bp tmp/mA3_scan/xray_bps.csv]

Figures: `production.py tmp/mA3_scan/benchmarks{,_xray} figures/mA3/production{,_xray} --mA 3`,
`benchmark.py tmp/mA3_scan/benchmarks{,_xray} figures/mA3/benchmarks{,_xray}`.

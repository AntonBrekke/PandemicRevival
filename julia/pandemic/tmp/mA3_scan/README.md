# Scan at m_A' = 3 m_N (2026-10-27; rerun 2026-11-07, the default scan)

## Benchmark points: which file is which (read first)

| file | points | role |
|---|---|---|
| `bp_round/bps.csv` | BP1 10 keV, g = 2e-4; BP2 20 keV, g = 1e-3 | **the benchmark points** (round m_N, g; chosen 2026-11-05). Pass this to `--bp`. |
| `xray_bps.csv` | 10.57, 13.72, 17.81 keV, g ~ 1e-4 - 4e-4 | older X-ray candidates (2026-10-27): allowed now, within Athena/eXTP reach. Not the BPs. |
| `xray_bps_candidates.csv` | 15 points | all X-ray candidates |
| `benchmarks/` | 10.57 keV, g = 1e-7 - 0.056 | standard histories across g (not BPs) |

`bp_round/README.md` has the BP roots and history; `bp_round/run_bps.sh` recomputes them (roots,
Lyman-alpha, histories). Make all m_A' = 3 figures with `./tmp/mA3_scan/make_figures.sh`: the money plots
with the BPs are `figures/mA3/relic_contours_bp{,_xproj}.pdf`, their histories `figures/mA3/{benchmarks,production}_bp_round/`.
`relic_scan_theta.py --xray-proj` writes only `<out>_xproj.pdf`, not `<out>.pdf`: run it twice.

## Rerun 2026-11-07 (branch pauli-exact)

All roots and Lyman-alpha lengths of `network_scan{,_fine}`, `fluid_scan{,_fine}`, `combined` recomputed
with the current solvers: reduced network, m_nu = 0.01 eV (M_NU; 0.1 eV before), the fluid with the
exact Pauli blocking of the portal 2 <-> 2 (`portal_tables.jl`) and with the cold-temperature fix (its
T_N froze at ~1e-3 m_N late on before, so its Lyman-alpha lengths were too large). Seeds: the first
scan's roots (`make_seeds_v2.py`, `seeds/*_v2*`); driver `run_mA3_v2.sh` (ThinkStation, 10 workers:
network 27 min, fluid base 154 min, fine 103 min, with Lyman-alpha ~6 h in all). The first scan is in
git history (commit b1cb53f and before). `mnu0.01/` (post-processing of the first scan at 0.01 eV) is
superseded.

- network: 492/654 (base; the rest degenerate at g >~ 3e-3, fluid domain) and 420/420 (fine) converged.
- fluid: 226/234 and 201/208 converged (8 + 7 bracketed: Omega jumps at the take-off).
- combined: 1275 roots, all with Lyman-alpha lengths.
- Against the first scan (new/old, median [5%, 95%]): roots 1.000 for 1e-6 <= g < 1e-2; 1.063
  [1.010, 1.175] at g >= 1e-2 (Pauli); down to 0.886 at a few points with g < 1e-6 (m_nu = 0.01 eV:
  the splitting threshold of `figures/mnu_threshold.pdf`). lambda_fs,kd 0.979 [0.918, 1.006] at
  g = 1e-3 - 1e-2 and 0.836 [0.591, 1.002] at g >= 1e-2 (the temperature fix); unchanged below.
- The money plot is unchanged to the eye: all of it happens at g >= 1e-2, inside the self-interaction
  region. X-ray BPs (`xray_bps.csv`, kept): roots and margins unchanged (network, g ~ 1e-4);
  `xray_bps_candidates.csv` lists all 15 candidates. Benchmarks rerun at the new roots
  (`run_benchmarks_v2.sh`; the fluid at g = 1e-3 at its bracket, 3.775e-15).

The sections below describe the first scan.

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
    ./tmp/mA3_scan/make_figures.sh        # every m_A' = 3 figure (money plots with/without BPs, histories)

The BPs for `--bp` are `bp_round/bps.csv` (see the top of this file), not `xray_bps.csv`.

# Scans with an N1-N2 mass splitting (2026-10-23/24)

delta = (m_N2 - m_N1)/m_N1, one global value over the (m_N, g) plane; everything else as in
the main scans (m_A' = 2.5 m_N, m_phi = 100 m_N). The couplings follow from delta
(`rate_params(...; delta)`): m_nu = sin^2 theta_1 m_D/delta per point (<= 0.02 eV everywhere for
delta = 1e-4... 1e-3), m_0 = 10 GeV (y_+ perturbative). delta = 0 is the original code, bit for bit.

## Why delta = 1e-4

N2 is depleted where the conversions N1 N1 <-> N2 N2 are still faster than H once
T_N/m_N < delta: their rate per N1 is ~ C(g, m) T_N/m_N H with C ∝ g^4/m^2 (C = 2.4e4 at 10.57
keV, g = 1e-3), so N2/N1 ~ e^(-delta C). delta = 1e-3 was tried first, but in this model the
phi N1 N1 Yukawa is tied to the splitting, lambda = g delta m_N/m_A', and at delta = 1e-3 it
thermalizes the dark-sector number early through N1 N1 -> phi -> A'A' (Omega h^2 x100 at
g = 1e-4 at fixed mixing; roots -14%). At delta = 1e-4 that effect is <= 4% and the roots move
by <= 0.2%, while N2 is depleted for g >~ 1e-3 (10 keV).

## Layout

- `seeds/`: seeds from the delta = 0 roots (`plot/delta_seeds.py`); `network_seeds_patch.csv`
  (`patch_seeds.py`).
- `delta1e-4/{network,fluid}_scan{,_fine}/`: roots and `lyman_alpha.csv` (with `x_fo`, the last
  x at which the conversions are faster than H); `delta1e-4/combined/` for the plots.
- `pilot/` (delta = 1e-3) and `pilot4/` (delta = 1e-4): 10.57 keV, g = 1e-4 ... 1e-2.
- `overlap_check_delta1e-4.txt`: network vs fluid in the overlap (`test/check_delta_overlap.jl`).
- `*.log`: logs of the runs below.

The delta = 1e-3 tables are not in the repository (a copy is in
`notes/mass_splitting_scan/` of the PhD folder; or rebuild with
`src/run_build_rate_tables.jl --delta 1e-3`, 36 min on 16 threads).

## Commands (from julia/pandemic)

    julia --project=. -t 16 src/run_build_rate_tables.jl --delta 1e-4       # 36 min
    conda run -n pandemic python plot/delta_seeds.py
    tmp/delta_runs/run_network_delta.sh 1e-4      # network roots + Lyman-alpha, base and fine
    tmp/delta_runs/run_fluid_delta.sh 1e-4 10     # fluid roots, base and fine
    tmp/delta_runs/run_post_delta.sh 1e-4 12      # fluid Lyman-alpha, combined, figures

The first network pass used dz_seed = 1e-9 for every solve, which fails at small g;
`run_network_patch.sh` re-solved those points (and the overlap) after the fix in
`solve_network` (1e-9 only as a fallback after a take-off at the start).

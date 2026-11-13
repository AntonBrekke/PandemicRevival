# m_A' = 50 m_N, m_phi = 150 m_N, m_nu = 0.01 eV (2026-11-06, coarse scan)

## Gamma_kd and lambda, 2026-11-13 (branch kd-transfer)

Lyman-alpha lengths of all roots (network and fluid) and every benchmark history recomputed
(`tmp/run_kd_transfer.sh`): Gamma_kd now weights its cross sections by the momentum transfer
(kinetic_decoupling.jl), and the N2/N1 relaxation rate lambda (`relaxation_rate`, conv_H, x_fo) is
no longer half the true one. Roots unchanged.
- x_c moves later wherever the unweighted rate was set by forward t/u-channel scattering at T >> m_A':
  median x_c at g = 1e-4 - 1e-2 from ~3e-5 - 7e-5 to ~1.2e-4 - 1.8e-4 (on-shell A' takes over); unchanged at g < 1e-6.
- lambda_fs,kd: new/old median 1.000, max +1.5%; r_s down by up to 12% at a few fluid roots,
  all far from the bound; lambda_fs (no coupling), f_N2 identical. No root changes its
  verdict for either Lyman-alpha bound: the money plot is unchanged to the eye.
- x_fo: later (it was the crossing of half the rate).
- Benchmark figures: `plot/benchmark.py` (until 2026-11-13 `production.py`), validity bands from x_c;
  all figures by `plot/make_figures.sh mA50`.


Tables: `tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls`. m_phi = 100 m_N would sit exactly at the
singular phi -> A'A' threshold (2 m_A'); 150 keeps it open with the quartic lambda = g^2 (m_phi/m_A')^2/2
= 4.5 g^2 perturbative up to g ~ 1.7.

## Checks before the scan
- Perturbativity (`test/check_perturbativity.jl 50 150`): the phi width binds first, Gamma_phi/m_phi = 10%
  at g ~ 1.46 (lambda = 4 pi at 1.67). Top coupling g = 1.333 (Gamma_phi/m_phi 8%, lambda/4pi 0.63).
- Tables (`TABLE_MA=50 TABLE_MPHI=150 TABLE_ALL=1 test/test_rate_tables_eval.jl`): all 16 reactions pass.
  Narrow width vs Breit-Wigner (`test/check_narrow_width.jl`): <= 0.25% up to g = 1.33. Fluid quadrature
  (`test/check_legendre_ratio.jl`): 4e-9.
- Pilot roots (`pilot/`): reduced network 15/15, fluid 24/24 up to 1 MeV and g = 1.333; network vs fluid
  at 10.6 keV, g = 1e-3: 0.1%. The kappa~ fluid of branch reduced-network failed here once the N turn
  degenerate; the fluid is back to rho.
- Benchmarks at 11 pilot roots (`benchmarks/`, `check_benchmarks.py`): thermal from the start and
  mu_N1 = mu_N2 (N1 N1 <-> N2 N2 >= 1e3 H) during growth at every fluid root; <= 6% of Omega made before
  the end of DW production; N nearly degenerate at large g (gap 0.003 at 10.6 keV, g = 1).
- At large g the dark sector thermalises at T_dw with mu ~ 0 and T_N/T_nu ~ 1e-4 - 1e-3 (A' is light
  then), i.e. a cold relativistic gas: MB (network) stops, the fluid applies.
- Portal 2 -> 2 rates (MB tables, no Pauli blocking of the final N): `portal_scale/`. Root shift for
  portal rates x0.5 / x0.25: 10.6 keV g = 0.1: +1% / +1.5%; 10.6 keV g = 1: x1.9 / x3.6; 300 keV g = 1:
  x1.4 / x1.7. With f_N ~ 0.5 at those roots the g ~ 1 roots are likely too low by up to that much.
- Pauli blocking, exactly (branch pauli-exact, `portal_tables.jl`; `pauli/`, `run_pauli_reroot.sh`): roots
  of the fluid with the Pauli-corrected portal 2 <-> 2 rates, seeded with the MB pilot roots. sin^2 2theta
  rises by 0% (10.6 keV, g = 0.1), +20% (10.6 keV, g = 0.5 - 1.333), +0.5 / +7 / +9% (300 keV,
  g = 0.1 / 1 / 1.333), +5 / +7% (600 keV, g = 1 / 1.333). Far below the portal-scale bracket above,
  which scaled the portal rates at all times; the blocking acts while the portal reactions seed n~ in
  the cold, nearly degenerate phase (x ~ 1e-3 - 1e-2, rates down ~13%).

## Coarse scan
13 masses 0.6 - 754 keV, half-decades of g from 1e-8, plus 1.333 (`make_coarse_seeds.py`); couplings with
sigma/m > 1 cm^2/g at any mixing (f_N2 = 1/2) more than a half-decade above the bound left out.
Upper mass: the X-ray limit / f_N1 closes the region at ~670 keV for g = 1.333 (pilot roots); above
1.022 MeV N1 -> nu e+ e- opens, which the code does not include.

Rerun 2026-11-07 with the Pauli-corrected fluid (branch pauli-exact; the MB scan is in git history,
commit 67ee55b) and the fluid's cold-temperature fix (b83d8b2, da1b8a1): before it the fluid's T_N froze
at ~1e-3 m_N once the sector had cooled below that, its Lyman-alpha lengths came out too large, and
alternating fluid and network roots along g = 1e-3 - 3e-3 folded the lambda_fs boundary.
- `network_scan/`: 130/130 + 8 points where the fluid only bracketed the root (`network_scan_bracketed/`,
  appended) + 6.3 keV, g = 3.2e-3 (`fluid_redo/net/`, appended): 139 roots.
- `fluid_scan/`: 74 converged + 2 re-rooted (119, 238 keV at g = 1.333; `fluid_redo/`, seeded with the
  MB roots x 1.07, --min-step 0.03), 7 bracketed (Omega jumps at the take-off; excluded as in the other
  scans). Lyman-alpha with the fixed fluid.
- `combined/`: 135 network + 76 fluid, 2 failed (0.6 keV, g = 3e-3 and 1e-2: bracketed; the network
  stops degenerate at g = 1e-2); Lyman-alpha for all 211. Pauli shifts against the MB scan: +5 - 15% in
  sin^2 2theta at g >= 0.3, < 5% below.

Fluid roots again on 2026-11-08 with the final fluid (the 11-07 rerun had the Pauli tables but the old
fluid state; only its Lyman-alpha lengths were from the fixed fluid): `run_mA50_v3.sh`, seeds from the
roots above (`make_seeds_v3.py`), 9 workers, 23 min + 16 min Lyman-alpha.
- `fluid_scan/`: 78 converged, 5 bracketed. `combined/`: 134 network + 78 fluid, 1 failed (0.6 keV,
  g = 1e-2); 212 with Lyman-alpha lengths. Against the 11-07 scan roots agree to 0.1% and lambda_fs,kd to
  1% (medians 1.000); 33 allowed roots, as before.
- `benchmark_roots/`: the off-grid fluid benchmarks (300 keV at g = 0.1, 1; 600 keV at 1.333; 10.57 keV
  at 1), seeded from `pauli/`; all four seeds already within 1% of 0.12.
- `benchmarks/`: all 11 histories at these roots with the final code (`run_benchmarks_v3.sh`; network
  roots unchanged). The earlier ones were MB with the frozen fluid temperature.

    python plot/relic_scan_theta.py tmp/mA50_scan/combined figures/mA50/relic_contours --mA 50 --mlim 1 800 \
        --slim 1e-24 1e-8 --g-pert 1.333 [--all-points]

`--g-pert 1.333` hatches the couplings above the largest scanned one (below its line of roots) as not
computed. Allowed region: ~7 - 600 keV, g ~ 1e-3 - 1, closed by self-interactions below, X-rays above and
at high mass, Lyman-alpha at low mass. Stopped at the coarse scan (user, 2026-11-06).

Commands: `run_pilot.sh`, `run_pilot_benchmarks.sh`, `run_portal_scale.sh`, then the network scan
(`src/run_network_scan.jl <tables> tmp/mA50_scan/network_scan tmp/mA50_scan/seeds/network_seeds.csv`,
`run_network_lyman_alpha.jl`) and `run_coarse_fluid.sh`.
